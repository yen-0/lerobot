from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .kinematics import JOINT_ORDER, load_default_kinematics
from .planner import PlanStep, solve_snake_target

REPO_ROOT = Path(__file__).resolve().parent.parent
ARM_CONFIG = REPO_ROOT / ".so101_arms.json"
COUNTS_PER_DEGREE = 4096.0 / 360.0
HALF_TURN = 2047
MOTOR_IDS: dict[str, int] = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


@dataclass(frozen=True)
class RobotMovePreview:
    current_joints_deg: np.ndarray
    target_joints_deg: np.ndarray
    delta_deg: np.ndarray
    current_raw: dict[str, int]
    target_raw: dict[str, int]
    ik_error: float
    max_delta_deg: float


def _load_follower_config(config_path: Path = ARM_CONFIG) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"No arm registry at {config_path}. Run `pixi run set-port follower` first.")
    cfg = json.loads(config_path.read_text())
    if "follower" not in cfg:
        raise KeyError("Follower is not registered. Run `pixi run set-port follower` first.")
    return cfg["follower"]


def raw_to_centered_degrees(raw: dict[str, int], homing_offsets: dict[str, int]) -> np.ndarray:
    values = []
    for name in JOINT_ORDER:
        true_pos = raw[name] + homing_offsets.get(name, 0)
        values.append((true_pos - HALF_TURN) / COUNTS_PER_DEGREE)
    return np.asarray(values, dtype=np.float32)


def target_raw_from_delta(current_raw: dict[str, int], delta_deg: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, delta in zip(JOINT_ORDER, delta_deg, strict=True):
        out[name] = int(np.clip(round(current_raw[name] + float(delta) * COUNTS_PER_DEGREE), 0, 4095))
    return out


def preview_move(
    target_snake: np.ndarray,
    current_raw: dict[str, int],
    homing_offsets: dict[str, int],
    max_delta_deg: float,
) -> RobotMovePreview:
    current_deg = raw_to_centered_degrees(current_raw, homing_offsets)
    plan = solve_snake_target(current_deg, target_snake, max_delta_deg=max_delta_deg, kinematics=load_default_kinematics())
    delta = np.clip(plan.target_joints_deg - current_deg, -max_delta_deg, max_delta_deg).astype(np.float32)
    # The gripper is not part of the backbone chain; leave it fixed unless a later
    # planner explicitly adds a gripper objective.
    delta[-1] = 0.0
    target_deg = current_deg + delta
    return RobotMovePreview(
        current_joints_deg=current_deg,
        target_joints_deg=target_deg,
        delta_deg=delta,
        current_raw=current_raw,
        target_raw=target_raw_from_delta(current_raw, delta),
        ik_error=plan.ik_error,
        max_delta_deg=float(np.abs(delta).max()),
    )


def _make_bus(port: str):
    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except Exception as exc:
        raise RuntimeError("Robot execution requires LeRobot. Run this command with `pixi run snake-plan-robot`.") from exc

    motors = {name: Motor(id_, "sts3215", MotorNormMode.RANGE_M100_100) for name, id_ in MOTOR_IDS.items()}
    return FeetechMotorsBus(port=port, motors=motors)


def execute_one_step(
    target_snake: np.ndarray,
    max_delta_deg: float = 3.0,
    config_path: Path = ARM_CONFIG,
    num_retry: int = 2,
    confirm: Callable[[RobotMovePreview], bool] | None = None,
) -> RobotMovePreview:
    follower = _load_follower_config(config_path)
    bus = _make_bus(follower["port"])
    bus.connect(handshake=True)
    try:
        current_raw: dict[str, int] = {}
        homing_offsets: dict[str, int] = {}
        for name in JOINT_ORDER:
            current_raw[name] = int(bus.read("Present_Position", name, normalize=False, num_retry=num_retry))
            try:
                homing_offsets[name] = int(bus.read("Homing_Offset", name, normalize=False, num_retry=num_retry))
            except Exception:
                homing_offsets[name] = 0

        preview = preview_move(target_snake, current_raw, homing_offsets, max_delta_deg)
        if preview.max_delta_deg > max_delta_deg + 1e-3:
            raise RuntimeError(f"Refusing move: max delta {preview.max_delta_deg:.3f} exceeds cap {max_delta_deg:.3f}")
        if confirm is not None and not confirm(preview):
            raise RuntimeError("Move cancelled before writing Goal_Position.")

        for name in JOINT_ORDER:
            bus.write("Goal_Position", name, preview.target_raw[name], normalize=False, num_retry=num_retry)
        return preview
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
