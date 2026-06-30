from curses import KEY_SUSPEND
from datetime import datetime
import os
import time

from gym.spaces import Space

import numpy as np
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from itertools import chain
from algorithms.marl.utils.separated_buffer import SeparatedReplayBuffer
from utils.util import update_linear_schedule

def _t2n(x):
    return x.detach().cpu().numpy()


def _convert_legacy_mlp_state_dict(state_dict):
    """Map checkpoints saved with the older fc1/fc2 MLP names to net.* names."""
    legacy_to_current = {
        "base.mlp.fc1.0.": "base.mlp.net.0.",
        "base.mlp.fc1.2.": "base.mlp.net.2.",
        "base.mlp.fc2.0.0.": "base.mlp.net.3.",
        "base.mlp.fc2.0.2.": "base.mlp.net.5.",
        "base.mlp.fc2.1.0.": "base.mlp.net.6.",
        "base.mlp.fc2.1.2.": "base.mlp.net.8.",
    }

    converted = {}
    changed = False
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in legacy_to_current.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                changed = True
                break
        converted[new_key] = value
    return converted, changed
     
class Runner:

    def __init__(self,
                 vec_env,
                 config,
                 model_dir=""
                 ):
        self.envs = vec_env
        self.eval_envs = vec_env
        # parameters
        self.env_name = vec_env.task.cfg["env"]["env_name"]
        self.algorithm_name = config["algorithm_name"]
        self.experiment_name = config["experiment_name"]
        self.use_centralized_V = config["use_centralized_V"]
        self.use_obs_instead_of_state = config["use_obs_instead_of_state"]
        self.num_env_steps = config["num_env_steps"]
        self.episode_length = config["episode_length"]
        self.n_rollout_threads = config["n_rollout_threads"]
        self.n_eval_rollout_threads = config["n_eval_rollout_threads"]
        self.use_linear_lr_decay = config["use_linear_lr_decay"]
        self.hidden_size = config["hidden_size"]
        self.use_render = config["use_render"]
        self.recurrent_N = config["recurrent_N"]
        self.use_single_network = config["use_single_network"]
        # interval
        self.save_interval = config["save_interval"]
        self.use_eval = config["use_eval"]
        self.eval_interval = config["eval_interval"]
        self.eval_episodes = config["eval_episodes"]
        self.log_interval = config["log_interval"]
        self.freeze_policy = config.get("freeze_policy", False)
        self.deterministic_rollout = config.get("deterministic_rollout", False)

        self.seed = self.envs.task.cfg["seed"]
        self.model_dir = model_dir

        self.num_agents = self.envs.num_agents
        self.device = self.envs.rl_device
        config["device"] = self.envs.rl_device

        torch.autograd.set_detect_anomaly(True)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir + '/' + self.env_name + '/' + self.algorithm_name +'/logs_seed{}'.format(self.seed))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir + '/' + self.env_name + '/' + self.algorithm_name + '/models_seed{}'.format(self.seed))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.best_metrics = {
            "episode_reward": -float("inf"),
            "goal_success_rate": -float("inf"),
            "hit_rate": -float("inf"),
            "catch_success_rate": -float("inf"),
        }

        if self.algorithm_name == "happo":
            from algorithms.marl.happo_trainer import HAPPO as TrainAlgo
            from algorithms.marl.happo_policy import HAPPO_Policy as Policy
        if self.algorithm_name == "hatrpo":
            from algorithms.marl.hatrpo_trainer import HATRPO as TrainAlgo
            from algorithms.marl.hatrpo_policy import HATRPO_Policy as Policy
        if self.algorithm_name == "mappo":
            from algorithms.marl.mappo_trainer import MAPPO as TrainAlgo
            from algorithms.marl.mappo_policy import MAPPO_Policy as Policy

        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            # policy network
            po = Policy(config,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device = self.device)
            self.policy.append(po)

        if self.model_dir != "":
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(config, self.policy[agent_id], device = self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(config,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = torch.zeros(1, self.n_rollout_threads, device=self.device)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            done_episodes_rewards = []
            done_episode_successes = []
            done_episode_hit_successes = []
            done_episode_catch_successes = []

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, _ = self.envs.step(actions)
                episode_success = None
                episode_hit_success = None
                episode_catch_success = None
                if isinstance(infos, dict):
                    if "episode_success" in infos:
                        episode_success = infos["episode_success"].to(self.device).flatten()
                    if "episode_hit_success" in infos:
                        episode_hit_success = infos["episode_hit_success"].to(self.device).flatten()
                    if "episode_catch_success" in infos:
                        episode_catch_success = infos["episode_catch_success"].to(self.device).flatten()
                
                dones_env = torch.all(dones, dim=1)

                reward_env = torch.mean(rewards, dim=1).flatten()

                train_episode_rewards += reward_env

                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[:, t].clone())
                        if episode_success is not None:
                            done_episode_successes.append(episode_success[t].clone())
                        if episode_hit_success is not None:
                            done_episode_hit_successes.append(episode_hit_success[t].clone())
                        if episode_catch_success is not None:
                            done_episode_catch_successes.append(episode_catch_success[t].clone())
                        train_episode_rewards[:, t] = 0

                data = obs, share_obs, rewards, dones, infos, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            if self.freeze_policy:
                train_infos = [{} for _ in range(self.num_agents)]
                for agent_id in range(self.num_agents):
                    self.buffer[agent_id].after_update()
            else:
                self.compute()
                train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if not self.freeze_policy and (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\nAlgo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.algorithm_name,
                              self.experiment_name,
                              episode, 
                              episodes, 
                              total_num_steps, 
                              self.num_env_steps, 
                              int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

            if len(done_episodes_rewards) != 0:
                aver_episode_rewards = torch.stack(done_episodes_rewards).mean().cpu().numpy().tolist()
                print("some episodes done, average rewards: ", aver_episode_rewards)
                self.writter.add_scalar("train_episode_rewards", aver_episode_rewards,
                                            total_num_steps)
                self.writter.add_scalar("train_episode_fps", int(total_num_steps / (end - start)),
                                            total_num_steps)
                self._maybe_save_best("episode_reward", aver_episode_rewards, episode, total_num_steps)

            if len(done_episode_successes) != 0:
                aver_success_rate = torch.stack(done_episode_successes).float().mean().cpu().numpy().tolist()
                print("some episodes done, average goal success rate: ", aver_success_rate)
                self.writter.add_scalar("train_episode_success_rate", aver_success_rate, total_num_steps)
                self._maybe_save_best("goal_success_rate", aver_success_rate, episode, total_num_steps)

            if len(done_episode_hit_successes) != 0:
                aver_hit_rate = torch.stack(done_episode_hit_successes).float().mean().cpu().numpy().tolist()
                print("some episodes done, average hit rate: ", aver_hit_rate)
                self.writter.add_scalar("train_episode_hr", aver_hit_rate, total_num_steps)
                self._maybe_save_best("hit_rate", aver_hit_rate, episode, total_num_steps)

            if len(done_episode_catch_successes) != 0:
                aver_catch_success_rate = torch.stack(done_episode_catch_successes).float().mean().cpu().numpy().tolist()
                print("some episodes done, average catch success rate: ", aver_catch_success_rate)
                self.writter.add_scalar("train_episode_sr", aver_catch_success_rate, total_num_steps)
                self._maybe_save_best("catch_success_rate", aver_catch_success_rate, episode, total_num_steps)

            if isinstance(infos, dict) and "mean_goal_dist" in infos:
                self.writter.add_scalar("train_mean_goal_dist", infos["mean_goal_dist"].mean().item(), total_num_steps)

            if isinstance(infos, dict) and "mean_catcher_palm_dist" in infos:
                self.writter.add_scalar("train_mean_catcher_palm_dist", infos["mean_catcher_palm_dist"].mean().item(), total_num_steps)

            if isinstance(infos, dict) and "mean_catcher_contact_force" in infos:
                self.writter.add_scalar("train_mean_catcher_contact_force", infos["mean_catcher_contact_force"].mean().item(), total_num_steps)

            if isinstance(infos, dict) and "mean_object_palm_relative_speed" in infos:
                self.writter.add_scalar("train_mean_object_palm_relative_speed", infos["mean_object_palm_relative_speed"].mean().item(), total_num_steps)

            if isinstance(infos, dict):
                for key in [
                    "debug_hrsr_hit_dist_rate",
                    "debug_hrsr_catch_dist_rate",
                    "debug_hrsr_hit_contact_rate",
                    "debug_hrsr_catch_contact_rate",
                    "debug_hrsr_catch_speed_rate",
                    "debug_hrsr_above_ground_rate",
                    "debug_hrsr_hit_now_rate",
                    "debug_hrsr_catch_condition_rate",
                    "debug_hrsr_catch_now_rate",
                    "debug_hrsr_min_palm_dist",
                    "debug_hrsr_max_catcher_contact_force",
                    "debug_hrsr_palm_contact_rate",
                    "debug_hrsr_finger_contact_rate",
                    "debug_hrsr_max_palm_contact_force",
                    "debug_hrsr_max_finger_contact_force",
                    "debug_hrsr_max_all_contact_force",
                    "debug_hrsr_min_object_palm_relative_speed",
                ]:
                    if key in infos:
                        self.writter.add_scalar(key, infos[key].mean().item(), total_num_steps)

            if isinstance(infos, dict) and "pos_loss" in infos:
                self.writter.add_scalar("train_traj_estimator_pos_loss", infos["pos_loss"].mean().item(), total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0].copy_(share_obs[:, agent_id])
            self.buffer[agent_id].obs[0].copy_(obs[:, agent_id])

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step],
                                                            deterministic=self.deterministic_rollout)
            value_collector.append(value.detach())
            action_collector.append(action.detach())
            action_log_prob_collector.append(action_log_prob.detach())
            rnn_state_collector.append(rnn_state.detach())
            rnn_state_critic_collector.append(rnn_state_critic.detach())

        # [self.envs, agents, dim]
        values = torch.transpose(torch.stack(value_collector), 1, 0)
        # TODO: 
        # actions = torch.transpose(torch.stack(action_collector), 1, 0)
        # action_log_probs = torch.transpose(torch.stack(action_log_prob_collector), 1, 0)
        rnn_states = torch.transpose(torch.stack(rnn_state_collector), 1, 0)
        rnn_states_critic = torch.transpose(torch.stack(rnn_state_critic_collector), 1, 0)

        return values, action_collector, action_log_prob_collector, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = torch.all(dones, axis=1)
        
        rnn_states[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size, device=self.device)
        rnn_states_critic[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:], device=self.device)

        masks = torch.ones(self.n_rollout_threads, self.num_agents, 1, device=self.device)
        masks[dones_env == True] = torch.zeros((dones_env == True).sum(), self.num_agents, 1, device=self.device)

        active_masks = torch.ones(self.n_rollout_threads, self.num_agents, 1, device=self.device)
        active_masks[dones == True] = torch.zeros((dones == True).sum(), 1, device=self.device)
        active_masks[dones_env == True] = torch.ones((dones_env == True).sum(), self.num_agents, 1, device=self.device)

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id], obs[:, agent_id], rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id], actions[agent_id],
                                         action_log_probs[agent_id],
                                         values[:, agent_id], rewards[:, agent_id], masks[:, agent_id], None,
                                         active_masks[:, agent_id], None)


    def train(self):
        train_infos = []
        # random update order

        action_dim = 1
        factor = torch.ones(self.episode_length, self.n_rollout_threads, action_dim, device=self.device)

        for agent_id in torch.randperm(self.num_agents):
            action_dim=self.buffer[agent_id].actions.shape[-1]

            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])

            if self.algorithm_name == "hatrpo":
                old_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            if self.algorithm_name == "hatrpo":
                new_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            action_prod = torch.exp((new_actions_logprob.detach()-old_actions_logprob.detach()).reshape(self.episode_length,self.n_rollout_threads,action_dim).sum(dim=-1, keepdim=True))
            factor = factor*action_prod.detach()
            train_infos.append(train_info)      
            self.buffer[agent_id].after_update()

        return train_infos

    def _maybe_save_best(self, metric_name, metric_value, episode, total_num_steps):
        if self.freeze_policy:
            return

        metric_value = float(metric_value)
        if metric_value <= self.best_metrics[metric_name] + 1.0e-8:
            return

        self.best_metrics[metric_name] = metric_value
        save_name = "best_{}".format(metric_name)
        self.save(episode, save_name=save_name)

        info_path = os.path.join(str(self.save_dir), save_name, "best_info.txt")
        with open(info_path, "w") as f:
            f.write("metric: {}\n".format(metric_name))
            f.write("value: {:.10f}\n".format(metric_value))
            f.write("episode_update: {}\n".format(episode))
            f.write("total_num_steps: {}\n".format(total_num_steps))

        self.writter.add_scalar("best/{}".format(metric_name), metric_value, total_num_steps)
        print("New best {}: {:.6f} at update {}, saved to {}".format(
            metric_name, metric_value, episode, os.path.join(str(self.save_dir), save_name)
        ))

    def save(self, episode, save_name=None):
        checkpoint_dir = os.path.join(str(self.save_dir), save_name if save_name is not None else str(episode))
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(policy_model.state_dict(), os.path.join(checkpoint_dir, "model_agent" + str(agent_id) + ".pt"))
            else:
                os.makedirs(checkpoint_dir, exist_ok=True)
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(policy_actor.state_dict(), os.path.join(checkpoint_dir, "actor_agent" + str(agent_id) + ".pt"))
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), os.path.join(checkpoint_dir, "critic_agent" + str(agent_id) + ".pt"))

    def restore(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
                policy_model_state_dict, converted = _convert_legacy_mlp_state_dict(policy_model_state_dict)
                if converted:
                    print("Converted legacy MLP checkpoint keys for model agent {}".format(agent_id))
                self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt', map_location=self.device)
                policy_actor_state_dict, converted_actor = _convert_legacy_mlp_state_dict(policy_actor_state_dict)
                if converted_actor:
                    print("Converted legacy MLP checkpoint keys for actor agent {}".format(agent_id))
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt', map_location=self.device)
                policy_critic_state_dict, converted_critic = _convert_legacy_mlp_state_dict(policy_critic_state_dict)
                if converted_critic:
                    print("Converted legacy MLP checkpoint keys for critic agent {}".format(agent_id))
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalar(agent_k, v, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            self.writter.add_scalars(k, {k: torch.mean(v)}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        eval_rnn_states = torch.zeros(self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size,
                                   device=self.device)
        eval_masks = torch.ones(self.n_eval_rollout_threads, self.num_agents, 1, device=self.device)

        while True:
            eval_actions_collector = []
            eval_rnn_states_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(eval_obs[:, agent_id],
                                                      eval_rnn_states[:, agent_id],
                                                      eval_masks[:, agent_id],
                                                      deterministic=True)
                eval_rnn_states[:, agent_id] = temp_rnn_state
                eval_actions_collector.append(eval_actions)

            eval_actions = eval_actions_collector

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = self.eval_envs.step(
                eval_actions)

            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            eval_dones_env = torch.all(eval_dones, dim=1)

            eval_rnn_states[eval_dones_env == True] = torch.zeros(
                (eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size, device=self.device)

            eval_masks = torch.ones(self.n_eval_rollout_threads, self.num_agents, 1, device=self.device)
            eval_masks[eval_dones_env == True] = torch.zeros((eval_dones_env == True).sum(), self.num_agents, 1,
                                                          device=self.device)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(torch.sum(torch.cat(one_episode_rewards[eval_i]), dim=0))
                    one_episode_rewards[eval_i] = []

            if eval_episode >= self.eval_episodes:
                eval_episode_rewards = torch.cat(eval_episode_rewards,dim=-1)
                eval_env_infos = {'eval_average_episode_rewards': torch.mean(eval_episode_rewards),
                                  'eval_max_episode_rewards': torch.max(eval_episode_rewards)}
                print(eval_env_infos)
                self.log_env(eval_env_infos, total_num_steps)
                print("eval_average_episode_rewards is {}.".format(torch.mean(eval_episode_rewards)))
                break

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = next_value.detach()
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)
