# """
# 输出字段形状:
#   observation.state   -> [B, DoF]
#   observation.point   -> [B, 3]
#   action              -> [B, T, DoF]
#   action_is_pad       -> [B, T]
# """
#
# import random
# from pathlib import Path
# from typing import Dict, List, Tuple
#
# import h5py
# import numpy as np
#
#
# def _load_episode_arrays(episode_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     with h5py.File(episode_path, "r") as f:
#         actions = f["observation/actions"][:]  # (T, DoF)
#         states = f["observation/state"][:]     # (T+1, DoF)
#         point = f["observation/point"][:]      # (3,)
#     return states, actions, point
#
#
# def _make_padded_actions(actions: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     给定动作序列 actions (T, DoF) 和时间步 t (基于状态 S_t), 构造长度 T 的动作序列:
#     取 A_t...A_{T-1}，若长度不足 T，则用最后一帧 A_{T-1} 进行后缀填充直到长度为 T。
#     若 t 指向最后一个状态（t == T-1，对应 A_{T-1}），则整段填充为 A_{T-1}。
#     返回:
#       padded_actions: (T, DoF)
#       is_pad: (T,) bool
#     """
#     T, dof = actions.shape
#     if t < 0 or t >= T:
#         raise ValueError(f"time index t out of range: {t}, valid [0, {T-1}]")
#
#     # 若选到最后一个状态（对应最后一个动作），整段填充为 A_{T-1}
#     if t == T - 1:
#         padded = np.repeat(actions[-1][None, :], T, axis=0)
#         is_pad = np.ones((T,), dtype=bool)
#         return padded, is_pad
#
#     tail = actions[t:]  # A_t ... A_{T-1}
#     real_len = tail.shape[0]
#     if real_len > T:
#         raise ValueError("Tail length exceeds total length, unexpected.")
#
#     pad_len = T - real_len
#     if pad_len > 0:
#         last_action = tail[-1]
#         pad_block = np.repeat(last_action[None, :], pad_len, axis=0)
#         padded = np.concatenate([tail, pad_block], axis=0)
#     else:
#         padded = tail
#
#     assert padded.shape[0] == T
#
#     is_pad = np.zeros((T,), dtype=bool)
#     if pad_len > 0:
#         is_pad[-pad_len:] = True
#     return padded, is_pad
#
#
# def load_one_sample(episode_path: Path, t: int | None = None) -> Dict[str, np.ndarray]:
#     """
#     从单个 episode 加载一个时间步的样本。
#     Args:
#         episode_path: h5 文件路径
#         t: 选取的状态索引 (0-based，对应 S_t 和 A_t)，None 则随机
#     Returns:
#         {
#             "observation.state": (DoF,),
#             "observation.point": (3,),
#             "action": (T, DoF),
#             "action_is_pad": (T,),
#         }
#     """
#     states, actions, point = _load_episode_arrays(episode_path)
#     T, dof = actions.shape
#     if t is None:
#         t = random.randint(0, T - 1)
#     if t >= states.shape[0]:
#         raise ValueError(f"time index t={t} exceeds states length {states.shape[0]}")
#
#     padded_actions, is_pad = _make_padded_actions(actions, t)
#     sample = {
#         "observation.state": states[t],
#         "observation.point": point,
#         "action": padded_actions,
#         "action_is_pad": is_pad,
#     }
#     return sample
#
#
# def load_batch(
#     data_dir: str | Path,
#     batch_size: int = 4,
#     t: int | None = None,
#     episode_indices: List[int] | None = None,
# ) -> Dict[str, np.ndarray]:
#     """
#     批量随机抽取样本。
#     Args:
#         data_dir: 包含 episode_XXXX.h5 的目录
#         batch_size: 批大小 B
#         t: 固定时间步 (可选)，None 时为每个样本随机
#         episode_indices: 限定可选的 episode 索引列表 (可选)
#     Returns:
#         同单样本，但每个字段前加 batch 维度:
#           observation.state -> (B, DoF)
#           observation.point -> (B, 3)
#           action -> (B, T, DoF)
#           action_is_pad -> (B, T)
#     """
#     data_dir = Path(data_dir)
#     episodes = sorted(data_dir.glob("episode_*.h5"))
#     if episode_indices is not None:
#         episodes = [episodes[i] for i in episode_indices]
#     if not episodes:
#         raise FileNotFoundError(f"No episode_*.h5 found in {data_dir}")
#
#     batch_states: List[np.ndarray] = []
#     batch_points: List[np.ndarray] = []
#     batch_actions: List[np.ndarray] = []
#     batch_is_pad: List[np.ndarray] = []
#
#     for _ in range(batch_size):
#         ep_path = random.choice(episodes)
#         sample = load_one_sample(ep_path, t=t)
#         batch_states.append(sample["observation.state"])
#         batch_points.append(sample["observation.point"])
#         batch_actions.append(sample["action"])
#         batch_is_pad.append(sample["action_is_pad"])
#
#     obs_state = np.stack(batch_states, axis=0)
#     obs_point = np.stack(batch_points, axis=0)
#     actions = np.stack(batch_actions, axis=0)
#     is_pad = np.stack(batch_is_pad, axis=0)
#
#     return {
#         "observation.state": obs_state,
#         "observation.point": obs_point,
#         "action": actions,
#         "action_is_pad": is_pad,
#     }
#
#
# if __name__ == "__main__":
#     dataset_dir = Path("/home/wang/code_python_project/motion_planning/data/dataset_rollouts")
#     batch = load_batch(dataset_dir, batch_size=4)
#     print("observation.state:", batch["observation.state"].shape)
#     print("observation.point:", batch["observation.point"].shape)
#     print("observation.point:", batch["observation.point"][0])
#     print("action:", batch["action"].shape)
#     print("action_is_pad:", batch["action_is_pad"].shape)
#
import threading

import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import random
from pathlib import Path

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, norm_stats, episode_len,
                 history_stack=0, max_open_files=10):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = Path(dataset_dir)
        self.norm_stats = norm_stats
        self.history_stack = history_stack
        self.max_pad_len = 8

        # 存储元数据而不是加载数据
        self.episode_metas = []
        for ep_id in episode_ids:
            ep_path = self.dataset_dir / f'episode_{ep_id:04d}.h5'
            with h5py.File(ep_path, 'r') as f:
                T = f['observation/actions'].shape[0]
                self.episode_metas.append({
                    'path': ep_path,
                    'length': T,
                    'state_dim': f['observation/state'].shape[1],
                    'action_dim': f['observation/actions'].shape[1],
                    'point_dim': f['observation/point'].shape[0],
                    'ee_dim': f['observation/eePosition'].shape[1]
                })

        self.episode_len = episode_len
        self.cumulative_len = np.cumsum(self.episode_len)

        # 文件句柄缓存
        self._file_cache = {}
        self._file_lock = threading.Lock()
        self.max_open_files = max_open_files

    def _get_file_handle(self, ep_idx):
        """获取文件句柄，带缓存"""
        with self._file_lock:
            if ep_idx in self._file_cache:
                return self._file_cache[ep_idx]

            # LRU缓存策略
            if len(self._file_cache) >= self.max_open_files:
                # 移除最久未使用的
                oldest_key = next(iter(self._file_cache))
                self._file_cache[oldest_key].close()
                del self._file_cache[oldest_key]

            ep_path = self.episode_metas[ep_idx]['path']
            f = h5py.File(ep_path, 'r')
            self._file_cache[ep_idx] = f
            return f

    def _locate_transition(self, index):
        """定位到具体episode和时间步"""
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        return episode_index, start_ts

    def __getitem__(self, ts_index):
        sample_full_episode = False

        ep_idx, start_ts = self._locate_transition(ts_index)
        meta = self.episode_metas[ep_idx]

        # 惰性加载文件
        f = self._get_file_handle(ep_idx)

        # 只读取需要的数据
        if sample_full_episode:
            start_ts = 0
        else:
            if start_ts >= meta['length']:
                start_ts = meta['length'] - 1

        # 读取当前时间步的状态
        qpos = f['/observation/state'][start_ts]
        point = f['/observation/point'][:]

        # 读取动作序列（只读取需要的部分）
        actions = f['observation/actions'][:]
        ee_positions = f['/observation/eePosition'][:]

        # 历史堆叠
        if self.history_stack > 0:
            last_indices = np.maximum(0, np.arange(start_ts - self.history_stack, start_ts)).astype(int)
            last_action = actions[last_indices, :]
            last_ee = ee_positions[last_indices, :]

        # 构造padded动作序列
        episode_len = meta['length']
        real_len = episode_len - start_ts
        if real_len <= 0:
            real_len = 1
            start_ts = episode_len - 1

        padded_action = np.zeros((self.max_pad_len, meta['action_dim']), dtype=np.float32)
        padded_ee = np.zeros((self.max_pad_len, meta['ee_dim']), dtype=np.float32)

        # 复制实际数据
        actual_len = min(real_len, self.max_pad_len)
        if actual_len > 0:
            end_idx = min(start_ts + actual_len, episode_len)
            padded_action[:actual_len] = actions[start_ts:end_idx]
            padded_ee[:actual_len] = ee_positions[start_ts:end_idx]

        # 填充剩余部分
        if actual_len < self.max_pad_len:
            padded_action[actual_len:] = actions[-1]
            padded_ee[actual_len:] = ee_positions[-1]

        is_pad = np.zeros(self.max_pad_len)
        is_pad[actual_len:] = 1

        # 转换为torch张量
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        ee_data = torch.from_numpy(padded_ee).float()
        point_data = torch.from_numpy(point).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # 归一化
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        ee_data = (ee_data - self.norm_stats["point_mean"]) / self.norm_stats["point_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        point_data = (point_data - self.norm_stats["point_mean"]) / self.norm_stats["point_std"]

        if self.history_stack > 0:
            last_action_data = torch.from_numpy(last_action).float()
            last_ee_data = torch.from_numpy(last_ee).float()
            last_action_data = (last_action_data - self.norm_stats['action_mean']) / self.norm_stats['action_std']
            last_ee_data = (last_ee_data - self.norm_stats['ee_mean']) / self.norm_stats['ee_std']
            qpos_data = torch.cat((qpos_data, last_action_data.flatten()))

        return point_data, qpos_data, action_data, is_pad, ee_data

    def __len__(self):
        return self.cumulative_len[-1] if self.cumulative_len.size > 0 else 0

    def __del__(self):
        """清理时关闭所有文件句柄"""
        with self._file_lock:
            for f in self._file_cache.values():
                try:
                    f.close()
                except:
                    pass
            self._file_cache.clear()


def get_norm_stats_streaming(dataset_dir, num_episodes, batch_size=1000):    # 初始化累加器
    action_sum = None
    action_sq_sum = None
    qpos_sum = None
    qpos_sq_sum = None
    point_sum = None
    point_sq_sum = None
    ee_sum = None
    ee_sq_sum = None

    total_action_samples = 0
    total_qpos_samples = 0
    total_ee_samples = 0
    total_episodes = 0

    example_qpos = None

    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx:04d}.h5')

        with h5py.File(dataset_path, 'r') as root:
            # 流式读取，分批次处理
            qpos = root['/observation/state'][:]
            action = root['observation/actions'][:]
            point = root['/observation/point'][:]
            ee_position = root['/observation/eePosition'][:]

            # 保存一个示例
            if example_qpos is None:
                example_qpos = qpos[:1].copy()

            # 转换并累加
            qpos_t = torch.from_numpy(qpos).float()
            action_t = torch.from_numpy(action).float()
            point_t = torch.from_numpy(point).float()
            ee_t = torch.from_numpy(ee_position).float()

            # 累加统计量
            if action_sum is None:
                action_sum = action_t.sum(dim=0)
                action_sq_sum = (action_t ** 2).sum(dim=0)
                qpos_sum = qpos_t.sum(dim=0)
                qpos_sq_sum = (qpos_t ** 2).sum(dim=0)
                point_sum = point_t
                point_sq_sum = point_t ** 2
                ee_sum = ee_t.sum(dim=0)
                ee_sq_sum = (ee_t ** 2).sum(dim=0)
            else:
                action_sum += action_t.sum(dim=0)
                action_sq_sum += (action_t ** 2).sum(dim=0)
                qpos_sum += qpos_t.sum(dim=0)
                qpos_sq_sum += (qpos_t ** 2).sum(dim=0)
                point_sum += point_t
                point_sq_sum += point_t ** 2
                ee_sum += ee_t.sum(dim=0)
                ee_sq_sum += (ee_t ** 2).sum(dim=0)

            total_action_samples += action_t.shape[0]
            total_qpos_samples += qpos_t.shape[0]
            total_ee_samples += ee_t.shape[0]
            total_episodes += 1

        # 定期清理内存
        if episode_idx % 100 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 计算均值和标准差
    action_mean = action_sum / total_action_samples
    action_std = torch.sqrt(action_sq_sum / total_action_samples - action_mean ** 2)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    qpos_mean = qpos_sum / total_qpos_samples
    qpos_std = torch.sqrt(qpos_sq_sum / total_qpos_samples - qpos_mean ** 2)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    ee_mean = ee_sum / total_ee_samples
    ee_std = torch.sqrt(ee_sq_sum / total_ee_samples - ee_mean ** 2)
    ee_std = torch.clip(ee_std, 1e-2, np.inf)

    point_mean = point_sum / total_episodes
    point_std = torch.sqrt(point_sq_sum / total_episodes - point_mean ** 2)

    # 计算最小最大值（可选）
    action_min = None
    action_max = None
    ee_min = None
    ee_max = None

    # 如果需要最小最大值，可以单独遍历计算或使用近似值
    # 这里为了内存考虑，使用mean±3*std作为近似边界
    eps = 0.0001
    action_min = (action_mean - 3 * action_std).numpy() - eps
    action_max = (action_mean + 3 * action_std).numpy() + eps
    ee_min = (ee_mean - 3 * ee_std).numpy() - eps
    ee_max = (ee_mean + 3 * ee_std).numpy() + eps

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "action_min": action_min,
        "action_max": action_max,
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "point_mean": point_mean.numpy().squeeze(),
        "point_std": point_std.numpy().squeeze(),
        "ee_mean": ee_mean.numpy().squeeze(),
        "ee_std": ee_std.numpy().squeeze(),
        "ee_min": ee_min,
        "ee_max": ee_max,
        "example_qpos": example_qpos,
    }

    return stats


def BatchSampler(batch_size, episode_len_l, sample_weights=None):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch


def find_all_processed_episodes(path):
    episodes = [f for f in os.listdir(path)]
    return episodes


class RandomBatchSampler:
    """内存高效的批采样器"""

    def __init__(self, episode_len_l, batch_size, seed=42):
        self.episode_len_l = episode_len_l
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        # 计算累积长度
        self.cumulative_len = np.cumsum([0] + episode_len_l)
        self.total_steps = self.cumulative_len[-1]

    def __iter__(self):
        # 生成一个epoch的索引
        indices = self.rng.permutation(self.total_steps)

        # 分批
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if len(batch_indices) == self.batch_size:
                yield list(batch_indices)

    def __len__(self):
        return self.total_steps // self.batch_size

def load_data(dataset_dir, batch_size_train, batch_size_val,
                   max_workers=2, prefetch_factor=2):
    print(f'\nLoading data from: {dataset_dir}\n')

    # 找到所有episode
    all_eps = [f for f in os.listdir(dataset_dir) if f.startswith('episode_')]
    num_episodes = len(all_eps)

    # 训练/验证划分
    train_ratio = 0.99
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    print(f"Total episodes: {num_episodes}")
    print(f"Train episodes: {len(train_indices)}, Val episodes: {len(val_indices)}")

    # 获取episode长度（惰性方式）
    episode_lengths = []
    for ep_idx in range(num_episodes):
        ep_path = os.path.join(dataset_dir, f'episode_{ep_idx:04d}.h5')
        with h5py.File(ep_path, 'r') as f:
            episode_lengths.append(f['observation/actions'].shape[0])

    train_episode_len_l = [episode_lengths[i] for i in train_indices]
    val_episode_len_l = [episode_lengths[i] for i in val_indices]

    # 流式计算统计量
    print("Computing normalization statistics (streaming)...")
    norm_stats = get_norm_stats_streaming(dataset_dir, num_episodes)

    # 创建数据集
    train_dataset = EpisodicDataset(
        train_indices,
        dataset_dir,
        norm_stats,
        train_episode_len_l,
        max_open_files=10  # 控制同时打开的文件数
    )

    val_dataset = EpisodicDataset(
        val_indices,
        dataset_dir,
        norm_stats,
        val_episode_len_l,
        max_open_files=5
    )

    # 使用自定义采样器
    train_sampler = RandomBatchSampler(train_episode_len_l, batch_size_train)
    val_sampler = RandomBatchSampler(val_episode_len_l, batch_size_val)

    # 配置DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=min(4, max_workers),
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=min(2, max_workers),
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=True
    )

    return train_dataloader, val_dataloader, norm_stats

# dataset_dir = "/home/wang/code_python_project/motion_planning/data/dataset_rollouts"
# batch_size_train = 12
# batch_size_val = 4
# train_dataloader, val_dataloader, norm_stats, _ = load_data(dataset_dir, batch_size_train, batch_size_val)

# point_data, qpos_data, action_data, is_pad = next(iter(train_dataloader))
# print(point_data)
# print(qpos_data)
# print(action_data)
# print(is_pad)
