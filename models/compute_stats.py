"""
统计数据集中的 mean/std 和 min/max，为 Normalize/Unnormalize 提供 stats。

输出文件示例（torch.save）:
{
  "observation.state": {"mean": tensor(DoF), "std": tensor(DoF), "min": tensor(DoF), "max": tensor(DoF)},
  "observation.point": {"mean": tensor(3), "std": tensor(3), "min": tensor(3), "max": tensor(3)},
  "action": {"mean": tensor(DoF), "std": tensor(DoF), "min": tensor(DoF), "max": tensor(DoF)},
}
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch


def compute_stats(data_dir: Path) -> dict:
    """计算数据集的统计信息（mean/std 和 min/max）"""
    all_states = []
    all_actions = []
    all_points = []
    all_ee_positions = []

    episodes = sorted(data_dir.glob("episode_*.h5"))
    if not episodes:
        raise FileNotFoundError(f"No episode_*.h5 found in {data_dir}")

    print(f"Computing stats from {len(episodes)} episodes...")
    
    for ep_path in episodes:
        with h5py.File(ep_path, "r") as f:
            states = f["observation/state"][:]  # (T+1, DoF)
            actions = f["observation/actions"][:]  # (T, DoF)
            point = f["observation/point"][:]  # (3,)
            eePosition = f["observation/eePosition"][:]
        
        all_states.append(torch.from_numpy(states))
        all_actions.append(torch.from_numpy(actions))
        all_points.append(torch.from_numpy(point))
        all_ee_positions.append(torch.from_numpy(eePosition))

    # 合并所有数据
    all_states = torch.cat(all_states, dim=0)  # (N, DoF)
    all_actions = torch.cat(all_actions, dim=0)  # (N, DoF)
    all_points = torch.stack(all_points, dim=0)  # (M, 3)
    all_ee_positions = torch.cat(all_ee_positions, dim=0)

    # 计算 mean/std
    state_mean = all_states.mean(dim=0)
    state_std = all_states.std(dim=0)
    state_std = torch.clamp(state_std, min=1e-2)  # 防止除零

    action_mean = all_actions.mean(dim=0)
    action_std = all_actions.std(dim=0)
    action_std = torch.clamp(action_std, min=1e-2)  # 防止除零

    point_mean = all_points.mean(dim=0)
    point_std = all_points.std(dim=0)
    point_std = torch.clamp(point_std, min=1e-2)  # 防止除零

    ee_position_mean = all_ee_positions.mean(dim=0)
    ee_position_std = all_ee_positions.std(dim=0)
    ee_position_std = torch.clamp(ee_position_std, min=1e-2)

    # 计算 min/max
    state_min = all_states.min(dim=0).values
    state_max = all_states.max(dim=0).values
    action_min = all_actions.min(dim=0).values
    action_max = all_actions.max(dim=0).values
    point_min = all_points.min(dim=0).values
    point_max = all_points.max(dim=0).values
    ee_position_min = all_ee_positions.min(dim=0).values
    ee_position_max = all_ee_positions.max(dim=0).values

    eps = 0.0001
    stats = {
        "observation.state": {
            "mean": state_mean.float(),
            "std": state_std.float(),
            "min": state_min.float() - eps,
            "max": state_max.float() + eps,
        },
        "observation.point": {
            "mean": point_mean.float(),
            "std": point_std.float(),
            "min": point_min.float() - eps,
            "max": point_max.float() + eps,
        },
        "action": {
            "mean": action_mean.float(),
            "std": action_std.float(),
            "min": action_min.float() - eps,
            "max": action_max.float() + eps,
        },
        "eePosition": {
            "mean": ee_position_mean.float(),
            "std": ee_position_std.float(),
            "min": ee_position_min.float() - eps,
            "max": ee_position_max.float() + eps,
        }
    }
    
    print(f"Stats computed:")
    print(f"  State: mean shape {state_mean.shape}, std shape {state_std.shape}")
    print(f"  Point: mean shape {point_mean.shape}, std shape {point_std.shape}")
    print(f"  Action: mean shape {action_mean.shape}, std shape {action_std.shape}")

    print(f"  State: mean {state_mean}, std shape {state_std}")
    print(f"  Point: mean {point_mean}, std shape {point_std}")
    print(f"  Action: mean {action_mean}, std shape {action_std}")
    print(f"  eePosition: mean {ee_position_mean}, std {ee_position_std}")
    print(f"  eePosition: max {ee_position_max}, min {ee_position_min}")
    
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dataset stats for normalization.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/wang/code_python_project/motion_planning/data/dataset_rollouts"),
        help="Directory with episode_*.h5 files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/wang/code_python_project/motion_planning/data/dataset_stats.pt"),
        help="Output stats file (torch.save)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stats = compute_stats(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.output)
    print(f"Saved stats to {args.output}")

