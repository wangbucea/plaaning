"""
推理脚本：
- 加载训练好的 DiffusionPolicy checkpoint
- 构建 PyBullet 机械臂环境，放置红色球体标记目标点
- 使用模型生成动作，驱动机械臂运动
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.env import RoboticArmEnv, sample_workspace_point
from models.dp_configs import DiffusionConfig
from models.diffusion_policy import DiffusionPolicy


def load_policy(ckpt_path: Path, device: torch.device, stats_path: Path | None = None) -> DiffusionPolicy:
    torch.serialization.add_safe_globals([DiffusionConfig])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", None)
    dataset_stats = None
    if stats_path is not None and Path(stats_path).exists():
        dataset_stats = torch.load(stats_path, map_location="cpu")
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_stats).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    return policy


def add_target_sphere(client_id: int, pos, radius: float = 0.03):
    import pybullet as p  # local import to avoid extra deps if only training

    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=[1, 0, 0, 1],  # red
        physicsClientId=client_id,
    )
    body = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=pos,
        physicsClientId=client_id,
    )
    return body


@torch.no_grad()
def rollout(
    policy: DiffusionPolicy,
    target_point,
    steps: int = 400,
    chunk_steps: int = 8,
    use_gui: bool = True,
    seed: int | None = 42,
):
    device = next(policy.parameters()).device
    env = RoboticArmEnv(use_gui=True, seed=seed)

    # 放置目标点可视化
    add_target_sphere(env.client_id, target_point)

    state = env.reset()

    for t in range(steps):
        # 每 chunk_steps 强制重新规划，丢弃旧队列中的剩余动作
        if t % chunk_steps == 0:
            policy.reset()

        # 组装 batch，n_obs_steps=1 -> shape (B, n_obs_steps, state_dim)
        obs_state = torch.tensor(state.q, dtype=torch.float32, device=device).unsqueeze(0)
        point = torch.tensor(target_point, dtype=torch.float32, device=device).unsqueeze(0)
        batch = {
            "observation.state": obs_state,
            "observation.point": point,
        }
        policy.eval()
        action = policy.select_action(batch)  # (B, action_dim)
        action_np = action.squeeze(0).cpu().numpy().tolist()

        state = env.step(action_np)

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for robotic arm with DiffusionPolicy.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=False,
        default=r"/home/wang/code_python_project/motion_planning/train_and_inference/ckpt/best/best_model.pt",
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=400,
        help="Number of control steps to run",
    )
    parser.add_argument(
        "--chunk_steps",
        type=int,
        default=15,
        help="Number of executed steps before re-planning (model re-run)",
    )
    parser.add_argument(
        "--use_gui",
        action="store_true",
        help="Enable PyBullet GUI",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for env",
    )
    parser.add_argument(
        "--stats_path",
        type=Path,
        default=Path("/home/wang/code_python_project/motion_planning/data/dataset_stats.pt"),
        help="Path to dataset stats for normalization",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = load_policy(args.ckpt, device, stats_path=args.stats_path)
    import random

    rng = random.Random(args.seed)
    target = sample_workspace_point(rng)
    rollout(
        policy,
        target_point=target,
        steps=args.steps,
        chunk_steps=args.chunk_steps,
        use_gui=args.use_gui,
        seed=args.seed,
    )

