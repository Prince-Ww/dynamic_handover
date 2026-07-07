import os

import cv2
import numpy as np
import torch

from isaacgym import gymapi

from train import apply_training_mode
from utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
from utils.parse_task import parse_task
from utils.process_marl import get_AgentIndex, process_MultiAgentRL


def capture_rgb_frame(task, env_id):
    task.gym.fetch_results(task.sim, True)
    task.gym.step_graphics(task.sim)
    task.gym.render_all_camera_sensors(task.sim)

    image = task.gym.get_camera_image(
        task.sim, task.envs[env_id], task.cameras[env_id], gymapi.IMAGE_COLOR
    )
    frame = np.asarray(image, dtype=np.uint8).reshape(
        task.camera_props.height, task.camera_props.width, 4
    )
    return frame[:, :, :3].copy()


def main():
    set_np_formatting()
    args = get_args()
    if args.model_dir == "":
        raise ValueError("--model_dir is required for video recording")
    if args.algo not in ["mappo", "happo", "hatrpo"]:
        raise ValueError("record_policy_video.py currently supports on-policy MARL algorithms")

    args.play = True
    args.train = False
    args.headless = True
    args.task_type = "MultiAgent"

    cfg, cfg_train, _ = load_cfg(args)
    cfg["env"]["enableCameraSensors"] = True
    cfg["env"]["cameraWidth"] = args.video_width
    cfg["env"]["cameraHeight"] = args.video_height
    cfg["headless"] = True
    cfg_train["run_dir"] = os.path.join(os.path.dirname(args.video_path), "_record_policy_logs")
    cfg_train["experiment_name"] = "record_policy_video"
    cfg_train["prevent_log_overwrite"] = False
    cfg_train["num_env_steps"] = max(1, int(args.video_steps))

    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))

    print("Algorithm: ", args.algo)
    apply_training_mode(args, cfg, cfg_train)
    agent_index = get_AgentIndex(cfg)
    _, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

    if args.video_env_id < 0 or args.video_env_id >= env.num_envs:
        raise ValueError("--video_env_id must be in [0, num_envs)")

    runner = process_MultiAgentRL(args, env=env, config=cfg_train, model_dir=args.model_dir)

    video_dir = os.path.dirname(args.video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        args.video_path, fourcc, args.video_fps, (args.video_width, args.video_height)
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer for {}".format(args.video_path))

    obs, _, _ = env.reset()
    eval_rnn_states = torch.zeros(
        env.num_envs, runner.num_agents, runner.recurrent_N, runner.hidden_size, device=runner.device
    )
    eval_masks = torch.ones(env.num_envs, runner.num_agents, 1, device=runner.device)

    done_episode_rewards = []
    done_goal_successes = []
    done_hit_successes = []
    done_catch_successes = []
    running_rewards = torch.zeros(env.num_envs, device=runner.device)
    frames_written = 0

    try:
        for step in range(args.video_steps):
            actions = []
            with torch.no_grad():
                for agent_id in range(runner.num_agents):
                    runner.trainer[agent_id].prep_rollout()
                    action, rnn_state = runner.trainer[agent_id].policy.act(
                        obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        deterministic=not args.video_stochastic,
                    )
                    eval_rnn_states[:, agent_id] = rnn_state
                    actions.append(action)

            obs, _, rewards, dones, infos, _ = env.step(actions)
            dones_env = torch.all(dones, dim=1)
            running_rewards += rewards.mean(dim=1).flatten()

            eval_rnn_states[dones_env] = 0
            eval_masks = torch.ones(env.num_envs, runner.num_agents, 1, device=runner.device)
            eval_masks[dones_env] = 0

            if isinstance(infos, dict):
                for env_id in torch.where(dones_env)[0].tolist():
                    done_episode_rewards.append(running_rewards[env_id].detach().clone())
                    running_rewards[env_id] = 0
                    if "episode_success" in infos:
                        done_goal_successes.append(infos["episode_success"].flatten()[env_id].detach().clone())
                    if "episode_hit_success" in infos:
                        done_hit_successes.append(infos["episode_hit_success"].flatten()[env_id].detach().clone())
                    if "episode_catch_success" in infos:
                        done_catch_successes.append(infos["episode_catch_success"].flatten()[env_id].detach().clone())

            if step % max(1, args.video_record_every) == 0:
                frame = capture_rgb_frame(env.task, args.video_env_id)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frames_written += 1

            if step % 50 == 0:
                print("record step {}/{} frames {}".format(step, args.video_steps, frames_written))
    finally:
        writer.release()

    print("Saved video to {}".format(args.video_path))
    print("Frames written:", frames_written)
    if done_episode_rewards:
        print("finished episodes:", len(done_episode_rewards))
        print("average episode reward:", torch.stack(done_episode_rewards).mean().item())
    if done_goal_successes:
        print("goal success rate:", torch.stack(done_goal_successes).float().mean().item())
    if done_hit_successes:
        print("hit rate:", torch.stack(done_hit_successes).float().mean().item())
    if done_catch_successes:
        print("catch success rate:", torch.stack(done_catch_successes).float().mean().item())


if __name__ == "__main__":
    main()
