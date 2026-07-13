from datetime import datetime
import os
import time

from gym.spaces import Space

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from algorithms.sarl.ppo import RolloutStorage


class PPO:

    def __init__(self,
                 vec_env,
                 actor_critic_class,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 init_noise_std=1.0,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=None,
                 model_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 eval_episodes=1000,
                 freeze_policy=False,
                 save_interval=500,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False
                 ):

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = device
        self.asymmetric = asymmetric

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.step_size = learning_rate

        # PPO components
        self.vec_env = vec_env
        self.actor_critic = actor_critic_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                                               init_noise_std, model_cfg, asymmetric=asymmetric)
        self.actor_critic.to(self.device)
        self.storage = RolloutStorage(self.vec_env.num_envs, num_transitions_per_env, self.observation_space.shape,
                                      self.state_space.shape, self.action_space.shape, self.device, sampler)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.eval_episodes = eval_episodes
        self.freeze_policy = freeze_policy
        self.save_interval = save_interval
        self.current_learning_iteration = 0
        self.best_metrics = {
            "episode_reward": -float("inf"),
            "goal_success_rate": -float("inf"),
            "hit_rate": -float("inf"),
            "catch_success_rate": -float("inf"),
        }

        self.apply_reset = apply_reset

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location='cuda:0'))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        filename = os.path.basename(path)
        if filename.startswith("model_") and filename.endswith(".pt"):
            self.current_learning_iteration = int(filename.split("_")[-1].split(".")[0])
        else:
            self.current_learning_iteration = 0
        self.actor_critic.train()

    def load_for_rollout(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location='cuda:0'))
        self.current_learning_iteration = 0
        self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        if self.is_testing:
            self.evaluate(self.eval_episodes, current_obs)
            return
        else:
            if self.freeze_policy:
                print("Frozen PPO stochastic rollout: policy updates and checkpoint saving are disabled.")

            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []
                done_episode_rewards = []
                done_episode_lengths = []
                done_episode_successes = []
                done_episode_hit_successes = []
                done_episode_catch_successes = []

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    # Compute the action
                    if self.freeze_policy:
                        with torch.no_grad():
                            actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs, current_states)
                    else:
                        actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs, current_states)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    next_states = self.vec_env.get_state()
                    # Record the transition
                    self.storage.add_transitions(current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma)
                    current_obs.copy_(next_obs)
                    current_states.copy_(next_states)
                    # Book keeping
                    ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews.to(self.device).view(-1)
                        cur_episode_length[:] += 1

                        done_ids = (dones.to(self.device).view(-1) > 0).nonzero(as_tuple=False).flatten()
                        if done_ids.numel() > 0:
                            done_episode_rewards.extend(
                                cur_reward_sum[done_ids].detach().cpu().numpy().tolist()
                            )
                            done_episode_lengths.extend(
                                cur_episode_length[done_ids].detach().cpu().numpy().tolist()
                            )
                        if done_ids.numel() > 0 and isinstance(infos, dict):
                            if "episode_success" in infos:
                                done_episode_successes.extend(
                                    infos["episode_success"].to(self.device).flatten()[done_ids].cpu().numpy().tolist()
                                )
                            if "episode_hit_success" in infos:
                                done_episode_hit_successes.extend(
                                    infos["episode_hit_success"].to(self.device).flatten()[done_ids].cpu().numpy().tolist()
                                )
                            if "episode_catch_success" in infos:
                                done_episode_catch_successes.extend(
                                    infos["episode_catch_success"].to(self.device).flatten()[done_ids].cpu().numpy().tolist()
                                )
                        cur_reward_sum[done_ids] = 0
                        cur_episode_length[done_ids] = 0

                total_num_steps = (it + 1) * self.num_transitions_per_env * self.vec_env.num_envs
                rollout_metrics = {}
                if self.print_log:
                    if len(done_episode_rewards) != 0:
                        rollout_metrics["episode_reward"] = float(np.mean(done_episode_rewards))
                    if len(done_episode_lengths) != 0:
                        rollout_metrics["episode_length"] = float(np.mean(done_episode_lengths))
                    if len(done_episode_successes) != 0:
                        rollout_metrics["goal_success_rate"] = float(np.mean(done_episode_successes))
                    if len(done_episode_hit_successes) != 0:
                        rollout_metrics["hit_rate"] = float(np.mean(done_episode_hit_successes))
                    if len(done_episode_catch_successes) != 0:
                        rollout_metrics["catch_success_rate"] = float(np.mean(done_episode_catch_successes))

                # Match MAPPO: best checkpoints reflect the policy that produced
                # the rollout, so save them before PPO updates on that rollout.
                if not self.freeze_policy:
                    for metric_name, metric_value in rollout_metrics.items():
                        if metric_name in self.best_metrics:
                            self._maybe_save_best(metric_name, metric_value, it, total_num_steps)

                if self.freeze_policy:
                    with torch.no_grad():
                        _, _, last_values, _, _ = self.actor_critic.act(current_obs, current_states)
                else:
                    _, _, last_values, _, _ = self.actor_critic.act(current_obs, current_states)
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                if self.freeze_policy:
                    mean_value_loss, mean_surrogate_loss = 0.0, 0.0
                else:
                    self.storage.compute_returns(last_values, self.gamma, self.lam)
                    mean_value_loss, mean_surrogate_loss = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if not self.freeze_policy and it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                ep_infos.clear()

            if not self.freeze_policy:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))

    def evaluate(self, num_eval_episodes=1000, current_obs=None):
        if current_obs is None:
            current_obs = self.vec_env.reset()

        self.actor_critic.eval()
        num_eval_episodes = int(num_eval_episodes)
        cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
        episode_rewards = []
        goal_successes = []
        hit_successes = []
        catch_successes = []

        print("PPO eval episodes: {}".format(num_eval_episodes))
        start = time.time()

        with torch.no_grad():
            while len(episode_rewards) < num_eval_episodes:
                if self.apply_reset:
                    current_obs = self.vec_env.reset()

                actions = self.actor_critic.act_inference(current_obs)
                next_obs, rews, dones, infos = self.vec_env.step(actions)
                current_obs.copy_(next_obs)

                cur_reward_sum += rews.to(self.device).view(-1)
                done_ids = (dones.to(self.device).view(-1) > 0).nonzero(as_tuple=False).flatten()

                if done_ids.numel() > 0:
                    remaining = num_eval_episodes - len(episode_rewards)
                    done_ids = done_ids[:remaining]

                    episode_rewards.extend(cur_reward_sum[done_ids].detach().cpu().numpy().tolist())

                    if isinstance(infos, dict):
                        if "episode_success" in infos:
                            goal_successes.extend(
                                infos["episode_success"].to(self.device).view(-1)[done_ids].detach().cpu().numpy().tolist()
                            )
                        if "episode_hit_success" in infos:
                            hit_successes.extend(
                                infos["episode_hit_success"].to(self.device).view(-1)[done_ids].detach().cpu().numpy().tolist()
                            )
                        if "episode_catch_success" in infos:
                            catch_successes.extend(
                                infos["episode_catch_success"].to(self.device).view(-1)[done_ids].detach().cpu().numpy().tolist()
                            )

                    cur_reward_sum[done_ids] = 0

                    if len(episode_rewards) % 1000 == 0 or len(episode_rewards) >= num_eval_episodes:
                        print("PPO eval progress: {}/{} episodes".format(len(episode_rewards), num_eval_episodes))

        elapsed = time.time() - start
        avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        max_reward = float(np.max(episode_rewards)) if episode_rewards else 0.0
        goal_rate = float(np.mean(goal_successes)) if goal_successes else 0.0
        hit_rate = float(np.mean(hit_successes)) if hit_successes else 0.0
        catch_rate = float(np.mean(catch_successes)) if catch_successes else 0.0

        print({
            "eval_average_episode_rewards": avg_reward,
            "eval_max_episode_rewards": max_reward,
            "eval_goal_success_rate": goal_rate,
            "eval_hit_rate": hit_rate,
            "eval_catch_success_rate": catch_rate,
            "eval_episodes": len(episode_rewards),
            "eval_time_sec": elapsed,
        })
        print("eval_average_episode_rewards is {}.".format(avg_reward))
        print("eval_max_episode_rewards is {}.".format(max_reward))
        print("eval_goal_success_rate is {}.".format(goal_rate))
        print("eval_hit_rate is {}.".format(hit_rate))
        print("eval_catch_success_rate is {}.".format(catch_rate))
        print("eval_episodes is {}.".format(len(episode_rewards)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        rollout_metrics = locs.get("rollout_metrics", {})

        ep_string = f''
        rollout_success_keys = {
            "episode_success",
            "episode_hit_success",
            "episode_catch_success",
        }
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                if key in rollout_success_keys:
                    continue
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.log_std.exp().mean()

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        if "episode_reward" in rollout_metrics:
            self.writer.add_scalar('Train/mean_reward', rollout_metrics["episode_reward"], locs['it'])
            self.writer.add_scalar('train_episode_rewards', rollout_metrics["episode_reward"], self.tot_timesteps)
            self.writer.add_scalar('Train/FPS',fps,locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', rollout_metrics["episode_reward"], self.tot_time)
        if "episode_length" in rollout_metrics:
            self.writer.add_scalar('Train/mean_episode_length', rollout_metrics["episode_length"], locs['it'])
            self.writer.add_scalar('Train/mean_episode_length/time', rollout_metrics["episode_length"], self.tot_time)
        if "goal_success_rate" in rollout_metrics:
            self.writer.add_scalar('train_episode_success_rate', rollout_metrics["goal_success_rate"], self.tot_timesteps)
        if "hit_rate" in rollout_metrics:
            self.writer.add_scalar('train_episode_hr', rollout_metrics["hit_rate"], self.tot_timesteps)
        if "catch_success_rate" in rollout_metrics:
            self.writer.add_scalar('train_episode_sr', rollout_metrics["catch_success_rate"], self.tot_timesteps)

        if locs['ep_infos']:
            estimator_tags = {
                "pos_loss": "train_traj_estimator_pos_loss",
                "traj_estimator_success_sample_rate": "train_traj_estimator_success_sample_rate",
                "traj_estimator_success_sample_count": "train_traj_estimator_success_sample_count",
            }
            for info_key, tag in estimator_tags.items():
                if info_key not in locs['ep_infos'][0]:
                    continue
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[info_key].to(self.device).flatten()))
                self.writer.add_scalar(tag, torch.mean(infotensor).item(), self.tot_timesteps)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        # fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if "episode_reward" in rollout_metrics:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {rollout_metrics['episode_reward']:.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {rollout_metrics.get('episode_length', 0.0):.2f}\n"""
                          f"""{'Mean goal success rate:':>{pad}} {rollout_metrics.get('goal_success_rate', 0.0):.4f}\n"""
                          f"""{'Mean hit rate:':>{pad}} {rollout_metrics.get('hit_rate', 0.0):.4f}\n"""
                          f"""{'Mean catch success rate:':>{pad}} {rollout_metrics.get('catch_success_rate', 0.0):.4f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def _maybe_save_best(self, metric_name, metric_value, iteration, total_num_steps):
        if self.freeze_policy:
            return

        metric_value = float(metric_value)
        if metric_value <= self.best_metrics[metric_name] + 1.0e-8:
            return

        self.best_metrics[metric_name] = metric_value
        save_name = "best_{}".format(metric_name)
        checkpoint_dir = os.path.join(self.log_dir, save_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.save(os.path.join(checkpoint_dir, "model.pt"))

        info_path = os.path.join(checkpoint_dir, "best_info.txt")
        with open(info_path, "w") as f:
            f.write("metric: {}\n".format(metric_name))
            f.write("value: {:.10f}\n".format(metric_value))
            f.write("iteration: {}\n".format(iteration))
            f.write("total_num_steps: {}\n".format(total_num_steps))

        self.writer.add_scalar("best/{}".format(metric_name), metric_value, total_num_steps)
        print("New best {}: {:.6f} at iteration {}, saved to {}".format(
            metric_name, metric_value, iteration, checkpoint_dir
        ))

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                if self.asymmetric:
                    states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                else:
                    states_batch = None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch,
                                                                                                                       states_batch,
                                                                                                                       actions_batch)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':

                    kl = torch.sum(
                        sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.step_size

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio

                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss
