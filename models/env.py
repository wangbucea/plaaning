"""
Simple PyBullet robotic arm environment for random data collection.

功能需求:
1. 构建一个机械臂环境。
2. 让机械臂以随机动作运动 200 步, 朝向一个随机目标点, 记录终点 (x, y, z)。
3. 采集三组数据: 机械臂状态序列、动作序列、终点坐标。
4. 数据顺序满足 S-A-S-A-S... 的对应关系。
"""

import dataclasses
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pybullet as p
import pybullet_data


@dataclasses.dataclass
class ArmState:
    """Robot state container (只保留关节角度)."""

    q: List[float]  # joint positions

    def as_dict(self) -> Dict[str, object]:
        return {"q": self.q}


@dataclasses.dataclass
class Transition:
    """Single S-A-S transition."""

    state: ArmState
    action: List[float]
    next_state: ArmState

    def as_dict(self) -> Dict[str, object]:
        return {
            "state": self.state.as_dict(),
            "action": self.action,
            "next_state": self.next_state.as_dict(),
        }


class RoboticArmEnv:
    def __init__(
        self,
        use_gui: bool = False,
        time_step: float = 1 / 240.0,
        ik_noise_std: float = 0.05,
        seed: int | None = None,
        init_joint_positions: List[float] | None = None,
        render_sleep: float = 0.5,  # GUI 下每步等待，便于观察
    ) -> None:
        self.use_gui = use_gui
        self.time_step = time_step
        self.ik_noise_std = ik_noise_std
        self._rng = random.Random(seed)
        self.init_joint_positions = init_joint_positions
        self.render_sleep = render_sleep

        self.client_id = p.connect(p.GUI if use_gui else p.DIRECT)
        p.resetSimulation(physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)

        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.robot_id = p.loadURDF(
            "/home/wang/code_python_project/motion_planning/rm_models-main/RM65/urdf/rm_65_b_description/urdf/rm_65_b_description.urdf",
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.client_id,
            # globalScaling=2,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 3.14159]),
        )

        self.joint_indices: List[int] = []
        self.joint_limits: List[Tuple[float, float]] = []
        for j in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                self.joint_indices.append(j)
                lower = info[8] if not math.isinf(info[8]) else -math.pi
                upper = info[9] if not math.isinf(info[9]) else math.pi
                self.joint_limits.append((lower, upper))

        if not self.joint_indices:
            raise RuntimeError("No controllable joints detected.")

        self.ee_link_index = self.joint_indices[-1]
        self.reset()

    def add_target_marker(self, pos: Tuple[float, float, float], radius: float = 0.03) -> int:
        """在 GUI 下添加红色小球作为目标点标记。返回 body id。"""
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[1, 0, 0, 1],
            physicsClientId=self.client_id,
        )
        body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=pos,
            physicsClientId=self.client_id,
        )
        return body

    def reset(self) -> ArmState:
        """Reset joint states to zero."""
        # 固定初始关节角度，可通过 init_joint_positions 传入；否则默认为 0。
        rng = random.Random()
        x = rng.uniform(0, 0.05)
        default_q = [x] * len(self.joint_indices)
        target_q = self.init_joint_positions or default_q
        if len(target_q) != len(self.joint_indices):
            raise ValueError(f"init_joint_positions length mismatch: expected {len(self.joint_indices)}, got {len(target_q)}")
        for idx, q in zip(self.joint_indices, target_q):
            p.resetJointState(self.robot_id, idx, q, targetVelocity=0.0, physicsClientId=self.client_id)
        p.stepSimulation(physicsClientId=self.client_id)
        return self.get_state()

    def close(self) -> None:
        p.disconnect(self.client_id)

    def get_state(self) -> ArmState:
        joint_states = p.getJointStates(self.robot_id, self.joint_indices, physicsClientId=self.client_id)
        q = [s[0] for s in joint_states]
        return ArmState(q=q)

    def get_end_effector_position(self) -> Tuple[float, float, float]:
        ee = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        return ee[4]

    def _clip_to_limits(self, targets: List[float]) -> List[float]:
        clipped = []
        for t, (low, high) in zip(targets, self.joint_limits):
            clipped.append(max(low, min(high, t)))
        return clipped

    def sample_noisy_ik_action(self, target_pos: Tuple[float, float, float]) -> List[float]:
        """Use IK to point toward the target and add small random noise for randomness."""
        ik_solution = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            target_pos,
            physicsClientId=self.client_id,
        )
        targets = list(ik_solution[: len(self.joint_indices)])
        noisy = [
            t + self._rng.gauss(0.0, self.ik_noise_std) for t in targets
        ]
        return self._clip_to_limits(noisy)

    def random_uniform_action(self) -> List[float]:
        """Purely random joint targets within limits."""
        return [
            self._rng.uniform(low, high) for (low, high) in self.joint_limits
        ]

    def step(self, action: List[float]) -> ArmState:
        if len(action) != len(self.joint_indices):
            raise ValueError(f"Expected {len(self.joint_indices)} actions, got {len(action)}")

        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.POSITION_CONTROL,
            targetPositions=self._clip_to_limits(action),
            physicsClientId=self.client_id,
        )
        p.stepSimulation(physicsClientId=self.client_id)
        if self.use_gui and self.render_sleep > 0:
            time.sleep(self.render_sleep)
        return self.get_state()


def sample_workspace_point(rng: random.Random) -> Tuple[float, float, float]:
    """Sample a reachable-ish workspace target."""
    # x = rng.uniform(0.35, 0.75)
    # y = rng.uniform(-0.35, 0.35)
    # z = rng.uniform(0.15, 0.65)
    range = rng.uniform(-0.3, 0.3)
    range_y = rng.uniform(-0.3, 0.3)
    range_z = rng.uniform(-0.2, 0.2)
    x = 0.0 + range
    y = 0.0 + range_y
    z = 0.45 + range_z

    # x = rng.uniform(0.3, 0.4)
    # y = rng.uniform(-0.3, 0.3)
    # z = rng.uniform(0.45, 0.45)
    return (x, y, z)


def collect_random_rollout(
    steps: int = 200,
    seed: int | None = None,
    use_gui: bool = False,
) -> Dict[str, object]:
    """
    Collect a single trajectory with S-A-S-A... pattern toward a random point.

    Returns:
        {
            "target_point": (x, y, z),
            "transitions": List[Transition],
            "final_position": (x, y, z),
            "states": List[ArmState],
            "actions": List[List[float]],
        }
    """
    env = RoboticArmEnv(use_gui=use_gui, seed=seed)
    rng = random.Random(seed)

    target = sample_workspace_point(rng)
    if env.use_gui:
        env.add_target_marker(target, radius=0.03)
    transitions: List[Transition] = []
    actions: List[List[float]] = []
    states: List[ArmState] = []
    ee_position = []

    state = env.reset()
    states.append(state)

    for _ in range(steps):
        # Prefer noisy IK toward the target to bias motion; fall back to uniform.
        try:
            action = env.sample_noisy_ik_action(target)
        except Exception:
            action = env.random_uniform_action()

        next_state = env.step(action)
        transitions.append(Transition(state=state, action=action, next_state=next_state))
        actions.append(action)
        states.append(next_state)
        state = next_state
        final_pos = env.get_end_effector_position()
        ee_position.append(final_pos)
    env.close()
    return {
        "target_point": target,
        "transitions": transitions,
        "ee_position": ee_position,
        "states": states,
        "actions": actions,
    }


def collect_dataset(
    num_episodes: int = 200,
    steps_per_episode: int = 200,
    output_dir: str | Path = "dataset_rollouts",
    seed: int | None = None,
    use_gui: bool = False,
) -> Path:
    """
    采集多条轨迹，每个 episode 单独保存为一个 H5 文件，结构:
      observation/state   -> (steps+1, DoF)  关节角度序列
      observation/actions -> (steps, DoF)
      observation/point   -> (3,)  终点 (x, y, z)
    """
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(num_episodes):
        rollout = collect_random_rollout(
            steps=steps_per_episode,
            seed=rng.randint(0, 1_000_000),
            use_gui=use_gui,
        )
        states_arr = np.array(
            [s.q for s in rollout["states"]],
            dtype=np.float32,
        )
        actions_arr = np.array(rollout["actions"], dtype=np.float32)
        point_arr = np.array(rollout["target_point"], dtype=np.float32)
        ee_position_arr = np.array(rollout["ee_position"], dtype=np.float32)

        ep_path = output_dir / f"episode_{ep:04d}.h5"
        with h5py.File(ep_path, "w") as f:
            obs = f.create_group("observation")
            obs.create_dataset("state", data=states_arr, compression="gzip")
            obs.create_dataset("actions", data=actions_arr, compression="gzip")
            obs.create_dataset("point", data=point_arr, compression="gzip")
            obs.create_dataset("eePosition", data=ee_position_arr, compression="gzip")

    return output_dir


if __name__ == "__main__":
    path = collect_dataset(
        num_episodes=100000,
        steps_per_episode=200,
        output_dir="dataset_rollouts",
        seed=2025,
        use_gui=False,
    )
    print(f"Saved dataset directory to {Path(path).resolve()}")



