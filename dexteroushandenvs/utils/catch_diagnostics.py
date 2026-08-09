import csv
import os

import numpy as np
import torch


class CatchDiagnosticsRecorder:
    """Collect per-frame catch signals and write one summary row per episode."""

    PREFIX = "catch_diag_"
    SUMMARY_FIELDS = [
        "episode_id", "env_id", "object_name", "object_id", "episode_steps",
        "episode_reward", "goal_success", "hit_success", "legacy_catch_success",
        "held_at_episode_end", "terminal_catch_condition", "human_label",
        "min_goal_dist", "min_palm_dist", "max_finger_contact_force",
        "max_palm_contact_force", "max_catcher_contact_force",
        "max_all_contact_force", "max_finger_contact_count",
        "max_palm_contact_count", "contact_frames", "catch_condition_frames",
        "max_consecutive_catch_condition_frames", "min_relative_speed",
        "mean_relative_speed", "terminal_relative_speed", "min_object_height",
        "terminal_object_height", "terminal_palm_dist", "min_finger_body_dist",
        "terminal_min_finger_body_dist", "terminal_finger_contact_force",
        "terminal_palm_contact_force", "terminal_catcher_contact_force",
        "terminal_finger_contact_count", "terminal_palm_contact_count",
        "terminal_object_speed", "terminal_palm_speed", "first_hit_step",
        "first_catch_condition_step", "first_legacy_catch_step",
    ]

    FINGER_BODY_COUNT = 11
    FRAME_FIELDS = [
        "episode_id", "env_id", "object_name", "object_id", "step",
        "goal_dist", "palm_dist", "finger_contact_force", "palm_contact_force",
        "catcher_contact_force", "all_contact_force", "finger_contact_count",
        "palm_contact_count", "object_speed", "palm_speed", "relative_speed",
        "object_height", "object_x", "object_y", "palm_x", "palm_y", "palm_z",
        "object_vx", "object_vy", "object_vz", "palm_vx", "palm_vy", "palm_vz",
        "relative_vx", "relative_vy", "relative_vz", "min_finger_body_dist",
        "hit_now", "catch_condition", "legacy_catch_now", "hold_steps",
    ] + ["finger_force_{:02d}".format(i) for i in range(FINGER_BODY_COUNT)] \
      + ["finger_dist_{:02d}".format(i) for i in range(FINGER_BODY_COUNT)]

    def __init__(self, output_dir, num_envs, object_names, write_frames=False):
        self.output_dir = os.path.abspath(output_dir)
        self.num_envs = int(num_envs)
        self.object_names = list(object_names)
        self.write_frames = bool(write_frames)
        self.next_episode_id = 1
        self.episode_ids = list(range(1, self.num_envs + 1))
        self.next_episode_id = self.num_envs + 1
        self.frames = [[] for _ in range(self.num_envs)]
        os.makedirs(self.output_dir, exist_ok=True)

        self.summary_path = os.path.join(self.output_dir, "episode_summary.csv")
        self.frames_path = os.path.join(self.output_dir, "frames.csv")
        self._summary_file = open(self.summary_path, "w", newline="")
        self._summary_writer = csv.DictWriter(self._summary_file, fieldnames=self.SUMMARY_FIELDS)
        self._summary_writer.writeheader()
        self._frames_file = None
        self._frames_writer = None
        if self.write_frames:
            self._frames_file = open(self.frames_path, "w", newline="")
            self._frames_writer = csv.DictWriter(self._frames_file, fieldnames=self.FRAME_FIELDS)
            self._frames_writer.writeheader()

    @staticmethod
    def _to_numpy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _first_true(frames, key):
        for frame in frames:
            if frame[key]:
                return int(frame["step"])
        return -1

    @staticmethod
    def _max_run(frames, key):
        best = current = 0
        for frame in frames:
            current = current + 1 if frame[key] else 0
            best = max(best, current)
        return best

    def record_step(self, infos):
        if not isinstance(infos, dict):
            return
        raw = {
            key[len(self.PREFIX):]: self._to_numpy(value).reshape(-1)
            for key, value in infos.items()
            if key.startswith(self.PREFIX)
        }
        required = {"progress", "goal_dist", "palm_dist", "relative_speed"}
        if not required.issubset(raw):
            return
        object_ids = self._to_numpy(infos.get("object_id", np.zeros(self.num_envs))).reshape(-1)
        for env_id in range(self.num_envs):
            object_id = int(object_ids[env_id]) if env_id < len(object_ids) else -1
            object_name = (
                self.object_names[object_id]
                if 0 <= object_id < len(self.object_names)
                else "unknown"
            )
            frame = {
                "episode_id": self.episode_ids[env_id],
                "env_id": env_id,
                "object_name": object_name,
                "object_id": object_id,
                "step": int(raw["progress"][env_id]),
            }
            for key in self.FRAME_FIELDS[5:]:
                value = raw[key][env_id]
                frame[key] = int(value) if key in {
                    "finger_contact_count", "palm_contact_count", "hit_now",
                    "catch_condition", "legacy_catch_now", "hold_steps",
                } else float(value)
            self.frames[env_id].append(frame)
            if self._frames_writer is not None:
                self._frames_writer.writerow(frame)

    def finish_episode(self, env_id, episode_reward, goal_success, hit_success,
                       legacy_catch_success):
        frames = self.frames[env_id]
        if not frames:
            return
        terminal = frames[-1]
        relative_speeds = [f["relative_speed"] for f in frames]
        contact_frames = sum(
            f["finger_contact_count"] > 0 or f["palm_contact_count"] > 0
            for f in frames
        )
        row = {
            "episode_id": self.episode_ids[env_id],
            "env_id": env_id,
            "object_name": terminal["object_name"],
            "object_id": terminal["object_id"],
            "episode_steps": len(frames),
            "episode_reward": float(episode_reward),
            "goal_success": int(goal_success),
            "hit_success": int(hit_success),
            "legacy_catch_success": int(legacy_catch_success),
            "held_at_episode_end": int(terminal["catch_condition"]),
            "terminal_catch_condition": int(terminal["catch_condition"]),
            "human_label": "",
            "min_goal_dist": min(f["goal_dist"] for f in frames),
            "min_palm_dist": min(f["palm_dist"] for f in frames),
            "max_finger_contact_force": max(f["finger_contact_force"] for f in frames),
            "max_palm_contact_force": max(f["palm_contact_force"] for f in frames),
            "max_catcher_contact_force": max(f["catcher_contact_force"] for f in frames),
            "max_all_contact_force": max(f["all_contact_force"] for f in frames),
            "max_finger_contact_count": max(f["finger_contact_count"] for f in frames),
            "max_palm_contact_count": max(f["palm_contact_count"] for f in frames),
            "contact_frames": contact_frames,
            "catch_condition_frames": sum(f["catch_condition"] for f in frames),
            "max_consecutive_catch_condition_frames": self._max_run(frames, "catch_condition"),
            "min_relative_speed": min(relative_speeds),
            "mean_relative_speed": sum(relative_speeds) / len(relative_speeds),
            "terminal_relative_speed": terminal["relative_speed"],
            "min_object_height": min(f["object_height"] for f in frames),
            "terminal_object_height": terminal["object_height"],
            "terminal_palm_dist": terminal["palm_dist"],
            "min_finger_body_dist": min(f["min_finger_body_dist"] for f in frames),
            "terminal_min_finger_body_dist": terminal["min_finger_body_dist"],
            "terminal_finger_contact_force": terminal["finger_contact_force"],
            "terminal_palm_contact_force": terminal["palm_contact_force"],
            "terminal_catcher_contact_force": terminal["catcher_contact_force"],
            "terminal_finger_contact_count": terminal["finger_contact_count"],
            "terminal_palm_contact_count": terminal["palm_contact_count"],
            "terminal_object_speed": terminal["object_speed"],
            "terminal_palm_speed": terminal["palm_speed"],
            "first_hit_step": self._first_true(frames, "hit_now"),
            "first_catch_condition_step": self._first_true(frames, "catch_condition"),
            "first_legacy_catch_step": self._first_true(frames, "legacy_catch_now"),
        }
        self._summary_writer.writerow(row)
        print(
            "catch_diag_episode={} object={} reward={:.4f} goal={} hit={} "
            "legacy_catch={} held_at_end={}".format(
                row["episode_id"], row["object_name"], row["episode_reward"],
                row["goal_success"], row["hit_success"],
                row["legacy_catch_success"], row["held_at_episode_end"],
            )
        )
        self._summary_file.flush()
        if self._frames_file is not None:
            self._frames_file.flush()
        self.frames[env_id] = []
        self.episode_ids[env_id] = self.next_episode_id
        self.next_episode_id += 1

    def discard_episode(self, env_id):
        self.frames[env_id] = []
        self.episode_ids[env_id] = self.next_episode_id
        self.next_episode_id += 1

    def close(self):
        if not self._summary_file.closed:
            self._summary_file.close()
        if self._frames_file is not None and not self._frames_file.closed:
            self._frames_file.close()
