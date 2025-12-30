"""
训练脚本 - 使用新的数据加载器 (EpisodicDataset)
添加：余弦学习率衰减、每50个epoch保存、最优权重保存
"""

import argparse
import random
import sys
from pathlib import Path
import numpy as np

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.load_data import load_data
from models.dp_configs import DiffusionConfig
from models.diffusion_policy import DiffusionPolicy


def make_config() -> DiffusionConfig:
    """创建模型配置"""
    cfg = DiffusionConfig()
    cfg.horizon = 8  # 模型预测的轨迹长度
    cfg.n_obs_steps = 1  # 观测序列长度
    cfg.n_action_steps = 8  # 执行动作长度
    cfg.input_shapes = {"observation.state": [6],
                        "observation.point":[3],
    }
    cfg.output_shapes = {"action": [6]}
    cfg.do_mask_loss_for_padding = True

    # 数据加载器已经进行了 mean_std 归一化，模型也使用 mean_std 模式
    cfg.input_normalization_modes = {
        "observation.state": "mean_std",
        "observation.point": "mean_std",
    }
    cfg.output_normalization_modes = {"action": "mean_std"}

    return cfg


def prepare_batch(point_data, qpos_data, action_data, is_pad,ee_position, device, n_obs_steps=1):
    """
    将 dataloader 返回的元组转换为模型期望的字典格式

    Args:
        point_data: (B, 3) 终点坐标
        qpos_data: (B, state_dim) 状态
        action_data: (B, horizon, action_dim) 动作序列
        is_pad: (B, horizon) 填充标记
        device: 目标设备
        n_obs_steps: 观测步数
    Returns:
        模型期望的 batch 字典
    """

    return {
        "observation.state": qpos_data.to(device),
        "observation.point": point_data.to(device),
        "action": action_data.to(device),
        "action_is_pad": is_pad.to(device),
        "ee_position": ee_position.to(device),
    }

def state_noise(data=None, batch_size=32, noise_ratio=0.3, noise_scale=0.1, device='cuda', stats_path=None):
    dataset_stats = torch.load(stats_path, map_location="cpu")
    # 给定的均值和标准差
    mean = dataset_stats["observation.state"]["mean"]
    std = dataset_stats["observation.state"]["std"]
    mean = torch.tensor(mean, device=device)

    original_data = data["observation.state"]

    # 2. 创建扰动掩码
    mask = torch.rand(batch_size, 6, device=device) < noise_ratio

    # 3. 生成扰动噪声
    noise = torch.randn(batch_size, 6, device=device) * mean * noise_scale

    # 4. 应用扰动
    perturbed_data = original_data.clone()
    perturbed_data[mask] += noise[mask]

    return original_data, perturbed_data, mask

def train(
        data_dir: Path,
        log_dir: Path,
        epochs: int = 400,
        steps_per_epoch: int = 100,
        batch_size_train: int = 16,
        batch_size_val: int = 4,
        lr: float = 1e-6,
        seed: int = 42,
        checkpoint_dir: Path | None = None,
        stats_path: Path | None = None,
        save_interval: int = 50,
        min_lr: float = 1e-8,
        max_workers: int = 2,  # 控制工作进程数
) -> None:
    """训练函数"""
    # 设置随机种子
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建配置
    cfg = make_config()

    # 加载数据
    print(f"\nLoading data from: {data_dir}")
    train_dataloader, val_dataloader, norm_stats = load_data(
        str(data_dir),
        batch_size_train,
        batch_size_val,
        max_workers=max_workers
    )
    print("Data loaded successfully.\n")

    # 加载数据集统计信息
    state_dim = cfg.input_shapes["observation.state"][0]
    action_dim = cfg.output_shapes["action"][0]
    point_dim = 3

    dataset_stats = torch.load(stats_path, map_location="cpu")
    # 创建模型
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_stats).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # 创建余弦学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,  # 总epoch数
        eta_min=min_lr  # 最小学习率
    )

    # 创建日志目录
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    if checkpoint_dir is None:
        checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 创建最优权重保存目录
    best_checkpoint_dir = checkpoint_dir / "best"
    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 创建迭代器
    train_iter = iter(train_dataloader)
    val_iter = iter(val_dataloader)

    print(f"Starting training for {epochs} epochs...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Batch size - Train: {batch_size_train}, Val: {batch_size_val}")
    print(f"Initial learning rate: {lr}")
    print(f"Min learning rate: {min_lr}")
    print(f"Save interval: every {save_interval} epochs")
    print(f"Best weights will be saved to: {best_checkpoint_dir}\n")

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        policy.train()
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_ee_loss = 0.0
        for scale in [100, 10, 1, 0.001]:
            print("scaling: ", scale)
            for step in range(steps_per_epoch):
                # 获取训练数据
                try:
                    point_data, qpos_data, action_data, is_pad, ee_position = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    point_data, qpos_data, action_data, is_pad, ee_position = next(train_iter)

                # 准备 batch
                batch = prepare_batch(
                    point_data, qpos_data, action_data, is_pad, ee_position, device, cfg.n_obs_steps
                )
                original_data, perturbed_data, mask = state_noise(data=batch,
                                                                  batch_size=16,
                                                                  noise_ratio=0.7,
                                                                  noise_scale=0.1,
                                                                  device='cuda',
                                                                  stats_path=stats_path)
                batch["observation.state"] = perturbed_data
                batch["observation.point"] = batch["observation.point"] * scale
                bn1d = nn.BatchNorm1d(num_features=3, device=device)
                batch["observation.point"] = bn1d(batch["observation.point"])
                # 前向传播
                optimizer.zero_grad(set_to_none=True)
                out = policy(batch)
                mse_loss = out["mse_loss"]
                # ee_loss = out["ee_loss"]
                loss = mse_loss

                # 反向传播
                loss.backward()
                optimizer.step()

                # 记录损失
                loss_val = loss.item()
                mse_loss_val = mse_loss.item()
                # ee_loss_val = ee_loss.item()

                epoch_loss += loss_val
                epoch_mse_loss += mse_loss_val
                # epoch_ee_loss += ee_loss_val

                writer.add_scalar("train/step_loss", loss_val, global_step)
                writer.add_scalar("train/step_mse_loss", mse_loss_val, global_step)
                # writer.add_scalar("train/step_ee_loss", ee_loss_val, global_step)
                global_step += 1

        # 计算平均损失
        avg_loss = epoch_loss / steps_per_epoch
        avg_mse_loss = epoch_mse_loss / steps_per_epoch
        # avg_ee_loss = epoch_ee_loss / steps_per_epoch

        # 记录当前学习率
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("train/learning_rate", current_lr, epoch)

        print(f"Epoch {epoch+1}/{epochs} [Train] "
              f"Loss: {avg_loss:.6f} | MSE: {avg_mse_loss:.6f} | LR: {current_lr:.2e}")

        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        writer.add_scalar("train/epoch_mse_loss", avg_mse_loss, epoch)
        # writer.add_scalar("train/epoch_ee_loss", avg_ee_loss, epoch)

        # ========== 验证阶段 ==========
        policy.eval()
        val_loss = 0.0
        val_mse_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            val_steps_to_run = 50  # 验证50个batch
            for step in range(val_steps_to_run):
                try:
                    point_data, qpos_data, action_data, is_pad, ee_data = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dataloader)
                    point_data, qpos_data, action_data, is_pad, ee_data = next(val_iter)

                batch = prepare_batch(
                    point_data, qpos_data, action_data, is_pad,ee_data, device, cfg.n_obs_steps
                )

                out = policy(batch)
                mse_loss = out["mse_loss"]
                loss =  mse_loss

                val_loss += loss.item()
                val_mse_loss += mse_loss.item()
                val_steps += 1

        if val_steps > 0:
            avg_val_loss = val_loss / val_steps
            avg_val_mse_loss = val_mse_loss / val_steps

            print(f"Epoch {epoch+1}/{epochs} [Val]   "
                  f"Loss: {avg_val_loss:.6f} | MSE: {avg_val_mse_loss:.6f}")

            writer.add_scalar("val/epoch_loss", avg_val_loss, epoch)
            writer.add_scalar("val/epoch_mse_loss", avg_val_mse_loss, epoch)

            # 更新最优模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_ckpt_path = best_checkpoint_dir / "best_model.pt"
                checkpoint_data = {
                    "epoch": epoch + 1,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": cfg,
                    "norm_stats": norm_stats,
                    "val_loss": avg_val_loss,
                    "best_val_loss": best_val_loss,
                }
                # 如果加载了 computed_stats，也保存原始统计信息
                if "_original_stats" in dataset_stats:
                    checkpoint_data["original_stats"] = dataset_stats["_original_stats"]
                torch.save(checkpoint_data, best_ckpt_path)
                print(f"✓ New best model saved with val loss: {avg_val_loss:.6f}")

        # 更新学习率
        scheduler.step()

        # 保存常规checkpoint
        # ckpt_path = checkpoint_dir / f"epoch_{epoch+1:04d}.pt"
        # checkpoint_data = {
        #     "epoch": epoch + 1,
        #     "model_state_dict": policy.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        #     "scheduler_state_dict": scheduler.state_dict(),
        #     "config": cfg,
        #     "norm_stats": norm_stats,
        #     "val_loss": avg_val_loss if val_steps > 0 else None,
        #     "best_val_loss": best_val_loss,
        # }
        # # 如果加载了 computed_stats，也保存原始统计信息
        # if "_original_stats" in dataset_stats:
        #     checkpoint_data["original_stats"] = dataset_stats["_original_stats"]
        # torch.save(checkpoint_data, ckpt_path)

        if (epoch + 1) % save_interval == 0:
            interval_ckpt_path = checkpoint_dir / f"epoch_{epoch+1:04d}_interval_{save_interval}.pt"
            torch.save(checkpoint_data, interval_ckpt_path)
            print(f"Interval checkpoint saved: {interval_ckpt_path}")

    # 保存最终模型
    final_ckpt_path = checkpoint_dir / "final_model.pt"
    checkpoint_data = {
        "epoch": epochs,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg,
        "norm_stats": norm_stats,
        "val_loss": avg_val_loss if val_steps > 0 else None,
        "best_val_loss": best_val_loss,
    }
    if "_original_stats" in dataset_stats:
        checkpoint_data["original_stats"] = dataset_stats["_original_stats"]
    torch.save(checkpoint_data, final_ckpt_path)
    print(f"\nFinal model saved: {final_ckpt_path}")

    writer.close()
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved at: {best_checkpoint_dir / 'best_model.pt'}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train DiffusionPolicy using EpisodicDataset")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/wang/code_python_project/motion_planning/data/dataset_rollouts"),
        help="Dataset directory containing episode_*.h5 files"
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("./runs/diffusion_policy"),
        help="TensorBoard log directory"
    )
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch")
    parser.add_argument("--batch_size_train", type=int, default=16, help="Training batch size")
    parser.add_argument("--batch_size_val", type=int, default=16, help="Validation batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default="/home/wang/code_python_project/motion_planning/train_and_inference/ckpt",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--stats_path",
        type=Path,
        default=Path("/home/wang/code_python_project/motion_planning/data/dataset_stats.pt"),
        help="Path to dataset stats file (computed by compute_stats.py)"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-8,
        help="Minimum learning rate for cosine annealing"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size_train=args.batch_size_train,
        batch_size_val=args.batch_size_val,
        lr=args.lr,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        stats_path=args.stats_path,
        save_interval=args.save_interval,
        min_lr=args.min_lr,
    )
