#!/usr/bin/env python3
"""Export MAPPO action-standard-deviation values from actor checkpoints."""

import argparse
import csv
import re
from pathlib import Path

import torch


def checkpoint_step(path):
    match = re.fullmatch(r"\d+", path.parent.name)
    return int(match.group(0)) if match else None


def action_std(state_dict, std_x_coef, std_y_coef):
    key = next((key for key in state_dict if key.endswith("action_out.log_std")), None)
    if key is None:
        raise KeyError("Missing action_out.log_std in actor checkpoint")
    log_std = state_dict[key].float()
    return torch.sigmoid(log_std / std_x_coef) * std_y_coef


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--std_x_coef", type=float, default=1.0)
    parser.add_argument("--std_y_coef", type=float, default=0.5)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    checkpoints = []
    for actor0 in models_dir.glob("*/actor_agent0.pt"):
        step = checkpoint_step(actor0)
        actor1 = actor0.with_name("actor_agent1.pt")
        if step is not None and actor1.exists():
            checkpoints.append((step, actor0, actor1))
    checkpoints.sort(key=lambda item: item[0])
    if not checkpoints:
        raise FileNotFoundError("No numeric MAPPO checkpoints with both actor files found")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "update", "env_steps", "agent0_std_mean", "agent0_std_min",
                "agent0_std_max", "agent1_std_mean", "agent1_std_min", "agent1_std_max",
            ],
        )
        writer.writeheader()
        for update, actor0, actor1 in checkpoints:
            std0 = action_std(torch.load(actor0, map_location="cpu"), args.std_x_coef, args.std_y_coef)
            std1 = action_std(torch.load(actor1, map_location="cpu"), args.std_x_coef, args.std_y_coef)
            writer.writerow({
                "update": update,
                "env_steps": update * args.num_envs * 8,
                "agent0_std_mean": std0.mean().item(),
                "agent0_std_min": std0.min().item(),
                "agent0_std_max": std0.max().item(),
                "agent1_std_mean": std1.mean().item(),
                "agent1_std_min": std1.min().item(),
                "agent1_std_max": std1.max().item(),
            })
    print("Wrote {} checkpoints to {}".format(len(checkpoints), output_csv))


if __name__ == "__main__":
    main()
