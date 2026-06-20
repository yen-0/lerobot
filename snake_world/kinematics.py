from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .urdf import JointSpec, chain_from_links, parse_urdf_joints

JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

DEFAULT_URDF = Path(__file__).resolve().parent.parent / "config" / "so101" / "so101_new_calib.urdf"


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    return _rot_z(y) @ _rot_y(p) @ _rot_x(r)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = np.cos(angle), np.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _rpy_matrix(rpy)
    out[:3, 3] = xyz
    return out


def _motion(joint: JointSpec, q_rad: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    if joint.joint_type in {"revolute", "continuous"}:
        out[:3, :3] = _axis_angle(joint.axis, q_rad)
    elif joint.joint_type == "prismatic":
        out[:3, 3] = joint.axis * q_rad
    return out


@dataclass
class SnakeKinematics:
    chain: list[JointSpec]
    joint_order: tuple[str, ...] = JOINT_ORDER

    @classmethod
    def from_urdf(
        cls,
        path: str | Path = DEFAULT_URDF,
        root_link: str = "base_link",
        tip_link: str = "gripper_frame_link",
    ) -> "SnakeKinematics":
        return cls(chain_from_links(parse_urdf_joints(path), root_link, tip_link))

    @property
    def limits_rad(self) -> np.ndarray:
        limits = []
        specs = {j.name: j for j in self.chain}
        for name in self.joint_order:
            spec = specs.get(name)
            if spec is None:
                limits.append([-np.inf, np.inf])
                continue
            limits.append(
                [
                    -np.inf if spec.lower is None else spec.lower,
                    np.inf if spec.upper is None else spec.upper,
                ]
            )
        return np.asarray(limits, dtype=np.float64)

    @property
    def limits_deg(self) -> np.ndarray:
        return np.rad2deg(self.limits_rad)

    def clamp_degrees(self, joints_deg: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(joints_deg, dtype=np.float64), *self.limits_deg.T)

    def forward(self, joints_deg: np.ndarray) -> np.ndarray:
        q_by_name = dict(zip(self.joint_order, np.deg2rad(np.asarray(joints_deg, dtype=np.float64)), strict=True))
        transform = np.eye(4, dtype=np.float64)
        points = [transform[:3, 3].copy()]
        for joint in self.chain:
            transform = transform @ _transform(joint.xyz, joint.rpy)
            if joint.name in q_by_name:
                transform = transform @ _motion(joint, q_by_name[joint.name])
                points.append(transform[:3, 3].copy())
            elif joint.joint_type == "fixed" and joint.child == "gripper_frame_link":
                points.append(transform[:3, 3].copy())
        return np.asarray(points, dtype=np.float32)

    def batch_forward(self, joints_deg: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints_deg, dtype=np.float64)
        return np.stack([self.forward(row) for row in joints], axis=0)

    def ik(
        self,
        target_snake: np.ndarray,
        initial_deg: np.ndarray,
        max_iters: int = 80,
        damping: float = 1e-3,
        step_limit_deg: float = 8.0,
    ) -> tuple[np.ndarray, float]:
        q = self.clamp_degrees(initial_deg)
        target = np.asarray(target_snake, dtype=np.float64)
        target_flat = target.reshape(-1)
        eps = 1e-2
        for _ in range(max_iters):
            current = self.forward(q).astype(np.float64).reshape(-1)
            err = target_flat - current
            if float(np.linalg.norm(err)) < 1e-4:
                break
            jac = np.zeros((current.size, q.size), dtype=np.float64)
            for j in range(q.size):
                q_eps = q.copy()
                q_eps[j] += eps
                jac[:, j] = (self.forward(q_eps).reshape(-1) - current) / eps
            lhs = jac.T @ jac + damping * np.eye(q.size)
            rhs = jac.T @ err
            delta = np.linalg.solve(lhs, rhs)
            delta = np.clip(delta, -step_limit_deg, step_limit_deg)
            q = self.clamp_degrees(q + delta)
        final_err = float(np.linalg.norm(target_flat - self.forward(q).reshape(-1)))
        return q.astype(np.float32), final_err


def load_default_kinematics() -> SnakeKinematics:
    return SnakeKinematics.from_urdf(DEFAULT_URDF)
