from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .kinematics import SnakeKinematics, load_default_kinematics


@dataclass(frozen=True)
class PlanStep:
    target_joints_deg: np.ndarray
    ik_error: float
    max_delta_deg: float


def solve_snake_target(
    current_joints_deg: np.ndarray,
    target_snake: np.ndarray,
    max_delta_deg: float = 5.0,
    kinematics: SnakeKinematics | None = None,
) -> PlanStep:
    kin = kinematics or load_default_kinematics()
    raw, err = kin.ik(target_snake, current_joints_deg)
    current = np.asarray(current_joints_deg, dtype=np.float32)
    delta = np.clip(raw - current, -max_delta_deg, max_delta_deg)
    target = kin.clamp_degrees(current + delta).astype(np.float32)
    return PlanStep(target, err, float(np.abs(target - current).max()))


def load_target_snake(path: str | Path) -> np.ndarray:
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        key = "target_snake" if "target_snake" in arr else arr.files[0]
        return np.asarray(arr[key], dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)
