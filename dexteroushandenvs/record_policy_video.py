import os

from isaacgym import gymapi

# Some cloud images leave this empty/invalid, which makes libgomp noisy.
if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch

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


def _tensor_pos(task, name, env_id):
    value = getattr(task, name, None)
    if value is None:
        return None
    if torch.is_tensor(value):
        return value[env_id].detach().cpu().numpy()
    return np.asarray(value[env_id])


def _draw_panel(frame, title, x_label, y_label, points, xlim, ylim, origin, size):
    ox, oy = origin
    width, height = size
    cv2.rectangle(frame, (ox, oy), (ox + width, oy + height), (245, 245, 245), -1)
    cv2.rectangle(frame, (ox, oy), (ox + width, oy + height), (80, 80, 80), 1)
    cv2.putText(frame, title, (ox + 12, oy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(frame, x_label, (ox + width - 50, oy + height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    cv2.putText(frame, y_label, (ox + 10, oy + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    for i in range(1, 5):
        x = ox + int(width * i / 5)
        y = oy + int(height * i / 5)
        cv2.line(frame, (x, oy), (x, oy + height), (220, 220, 220), 1)
        cv2.line(frame, (ox, y), (ox + width, y), (220, 220, 220), 1)

    def project(point):
        px = ox + int((point[0] - xlim[0]) / (xlim[1] - xlim[0]) * width)
        py = oy + height - int((point[1] - ylim[0]) / (ylim[1] - ylim[0]) * height)
        return px, py

    for label, point, color in points:
        if point is None:
            continue
        px, py = project(point)
        if ox <= px <= ox + width and oy <= py <= oy + height:
            cv2.circle(frame, (px, py), 7, color, -1)
            cv2.circle(frame, (px, py), 8, (30, 30, 30), 1)
            cv2.putText(frame, label, (px + 9, py - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def capture_state_frame(task, env_id, step, infos, width, height):
    object_pos = _tensor_pos(task, "object_pos", env_id)
    goal_pos = _tensor_pos(task, "goal_pos", env_id)
    throw_hand_pos = _tensor_pos(task, "allegro_right_hand_pos", env_id)
    catch_hand_pos = _tensor_pos(task, "a_hand_palm_pos", env_id)
    if catch_hand_pos is None:
        catch_hand_pos = _tensor_pos(task, "allegro_left_hand_pos", env_id)

    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    top_points = [
        ("object", None if object_pos is None else object_pos[[0, 1]], (30, 30, 230)),
        ("goal", None if goal_pos is None else goal_pos[[0, 1]], (40, 180, 40)),
        ("throw", None if throw_hand_pos is None else throw_hand_pos[[0, 1]], (230, 120, 20)),
        ("catch", None if catch_hand_pos is None else catch_hand_pos[[0, 1]], (180, 60, 180)),
    ]
    side_points = [
        ("object", None if object_pos is None else object_pos[[1, 2]], (30, 30, 230)),
        ("goal", None if goal_pos is None else goal_pos[[1, 2]], (40, 180, 40)),
        ("throw", None if throw_hand_pos is None else throw_hand_pos[[1, 2]], (230, 120, 20)),
        ("catch", None if catch_hand_pos is None else catch_hand_pos[[1, 2]], (180, 60, 180)),
    ]

    pad = 18
    panel_w = (width - pad * 3) // 2
    panel_h = height - 115
    _draw_panel(frame, "Top view", "x", "y", top_points, (-1.0, 1.0), (-1.2, 0.3), (pad, 60), (panel_w, panel_h))
    _draw_panel(frame, "Side view", "y", "z", side_points, (-1.2, 0.3), (0.0, 1.1), (pad * 2 + panel_w, 60), (panel_w, panel_h))

    cv2.putText(frame, "Isaac Gym state rollout", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2)
    cv2.putText(frame, "step {}".format(step), (width - 145, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    legend = "red=object  green=goal  orange=throw hand  purple=catch hand"
    cv2.putText(frame, legend, (18, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)

    if isinstance(infos, dict):
        metric_parts = []
        for key, label in [
            ("episode_success", "goal"),
            ("episode_hit_success", "hit"),
            ("episode_catch_success", "catch"),
        ]:
            if key in infos:
                value = infos[key].flatten()[env_id].detach().cpu().item()
                metric_parts.append("{}={:.0f}".format(label, value))
        if metric_parts:
            cv2.putText(frame, " ".join(metric_parts), (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1)

    return frame


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

    if args.video_backend not in ["state", "camera"]:
        raise ValueError("--video_backend must be either state or camera")

    cfg, cfg_train, _ = load_cfg(args)
    cfg["env"]["enableCameraSensors"] = args.video_backend == "camera"
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
                if args.video_backend == "camera":
                    frame = capture_rgb_frame(env.task, args.video_env_id)
                else:
                    frame = capture_state_frame(
                        env.task, args.video_env_id, step, infos, args.video_width, args.video_height
                    )
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
