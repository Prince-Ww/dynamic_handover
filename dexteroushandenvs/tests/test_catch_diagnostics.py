import csv

import torch

from utils.catch_diagnostics import CatchDiagnosticsRecorder


def test_catch_diagnostics_writes_episode_summary(tmp_path):
    recorder = CatchDiagnosticsRecorder(tmp_path, 1, ["ball"], write_frames=True)
    for step, condition in enumerate([0, 1, 1], start=1):
        infos = {
            "object_id": torch.tensor([0]),
            "catch_diag_progress": torch.tensor([step]),
            "catch_diag_goal_dist": torch.tensor([0.3 - step * 0.05]),
            "catch_diag_palm_dist": torch.tensor([0.2 - step * 0.02]),
            "catch_diag_finger_contact_force": torch.tensor([float(condition)]),
            "catch_diag_palm_contact_force": torch.tensor([0.0]),
            "catch_diag_catcher_contact_force": torch.tensor([float(condition)]),
            "catch_diag_all_contact_force": torch.tensor([float(condition)]),
            "catch_diag_finger_contact_count": torch.tensor([condition]),
            "catch_diag_palm_contact_count": torch.tensor([0]),
            "catch_diag_object_speed": torch.tensor([1.0]),
            "catch_diag_palm_speed": torch.tensor([0.5]),
            "catch_diag_relative_speed": torch.tensor([0.5]),
            "catch_diag_object_height": torch.tensor([0.4]),
            "catch_diag_object_x": torch.tensor([0.0]),
            "catch_diag_object_y": torch.tensor([0.0]),
            "catch_diag_palm_x": torch.tensor([0.0]),
            "catch_diag_palm_y": torch.tensor([0.0]),
            "catch_diag_palm_z": torch.tensor([0.3]),
            "catch_diag_object_vx": torch.tensor([0.0]),
            "catch_diag_object_vy": torch.tensor([0.0]),
            "catch_diag_object_vz": torch.tensor([0.0]),
            "catch_diag_palm_vx": torch.tensor([0.0]),
            "catch_diag_palm_vy": torch.tensor([0.0]),
            "catch_diag_palm_vz": torch.tensor([0.0]),
            "catch_diag_relative_vx": torch.tensor([0.0]),
            "catch_diag_relative_vy": torch.tensor([0.0]),
            "catch_diag_relative_vz": torch.tensor([0.0]),
            "catch_diag_min_finger_body_dist": torch.tensor([0.1]),
            "catch_diag_hit_now": torch.tensor([condition]),
            "catch_diag_catch_condition": torch.tensor([condition]),
            "catch_diag_legacy_catch_now": torch.tensor([int(step == 3)]),
            "catch_diag_hold_steps": torch.tensor([max(0, step - 1)]),
        }
        for finger_i in range(11):
            infos["catch_diag_finger_force_{:02d}".format(finger_i)] = torch.tensor([0.1])
            infos["catch_diag_finger_dist_{:02d}".format(finger_i)] = torch.tensor([0.1])
        recorder.record_step(infos)
    recorder.finish_episode(0, 3.0, 1, 1, 1)
    recorder.close()

    with open(tmp_path / "episode_summary.csv", newline="") as summary_file:
        row = next(csv.DictReader(summary_file))
    assert row["object_name"] == "ball"
    assert row["held_at_episode_end"] == "1"
    assert row["max_consecutive_catch_condition_frames"] == "2"
    assert row["terminal_finger_contact_count"] == "1"
    assert row["terminal_min_finger_body_dist"] == "0.10000000149011612"
    assert row["human_label"] == ""
