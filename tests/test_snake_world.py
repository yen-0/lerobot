from __future__ import annotations

import numpy as np
from typer.testing import CliRunner

from snake_world.kinematics import JOINT_ORDER, load_default_kinematics
from snake_world.cli import app
from snake_world.model import train_world_model
from snake_world.planner import solve_snake_target
from snake_world.robot import preview_move, raw_to_centered_degrees, target_raw_from_delta


def test_default_urdf_chain_has_expected_joint_order() -> None:
    kin = load_default_kinematics()
    names = [joint.name for joint in kin.chain if joint.name in JOINT_ORDER]
    assert names == list(JOINT_ORDER[:5])


def test_forward_returns_ordered_3d_snake_points() -> None:
    kin = load_default_kinematics()
    snake = kin.forward(np.zeros(6, dtype=np.float32))
    assert snake.ndim == 2
    assert snake.shape[1] == 3
    assert snake.shape[0] >= 6
    assert np.isfinite(snake).all()


def test_ik_reconstructs_fk_generated_target() -> None:
    kin = load_default_kinematics()
    start = np.zeros(6, dtype=np.float32)
    target_joints = np.array([10, -12, 15, -8, 0, 0], dtype=np.float32)
    target = kin.forward(target_joints)
    solved, err = kin.ik(target, start, max_iters=80)
    assert err < 0.04
    assert np.isfinite(solved).all()


def test_planner_caps_joint_delta() -> None:
    kin = load_default_kinematics()
    current = np.zeros(6, dtype=np.float32)
    target = kin.forward(np.array([30, -30, 20, 10, 0, 0], dtype=np.float32))
    step = solve_snake_target(current, target, max_delta_deg=3.0, kinematics=kin)
    assert step.max_delta_deg <= 3.001


def test_raw_encoder_conversion_uses_homing_offsets() -> None:
    raw = {name: 2047 for name in JOINT_ORDER}
    offsets = {name: 0 for name in JOINT_ORDER}
    offsets["shoulder_pan"] = 114
    deg = raw_to_centered_degrees(raw, offsets)
    assert abs(float(deg[0]) - 10.0195) < 0.01
    assert np.allclose(deg[1:], 0.0)


def test_robot_preview_caps_delta_and_keeps_gripper_fixed() -> None:
    kin = load_default_kinematics()
    raw = {name: 2047 for name in JOINT_ORDER}
    target = kin.forward(np.array([25, -20, 15, 5, 0, 0], dtype=np.float32))
    preview = preview_move(target, raw, {}, max_delta_deg=2.5)
    assert preview.max_delta_deg <= 2.501
    assert preview.delta_deg[-1] == 0.0
    assert preview.target_raw["gripper"] == raw["gripper"]


def test_target_raw_from_delta_clips_servo_range() -> None:
    raw = {name: 4090 for name in JOINT_ORDER}
    delta = np.full(len(JOINT_ORDER), 10.0, dtype=np.float32)
    targets = target_raw_from_delta(raw, delta)
    assert all(value == 4095 for value in targets.values())


def test_plan_robot_dry_run_cli(tmp_path) -> None:
    kin = load_default_kinematics()
    target = kin.forward(np.array([1, 0, 0, 0, 0, 0], dtype=np.float32))
    target_path = tmp_path / "target.npy"
    np.save(target_path, target)
    result = CliRunner().invoke(
        app,
        [
            "plan-robot",
            "--dry-run",
            "--current-raw",
            "2047,2047,2047,2047,2047,2047",
            "--target-snake",
            str(target_path),
            "--max-delta",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "target_raw:" in result.output


def test_train_checkpoint_uses_observed_next_state_delta(tmp_path) -> None:
    import torch

    kin = load_default_kinematics()
    states = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    arm = kin.batch_forward(states)
    path = tmp_path / "tiny.npz"
    np.savez_compressed(
        path,
        state=states,
        action=states + 10,
        arm=arm,
        next_arm=np.roll(arm, -1, axis=0),
        object=np.zeros((len(states), 1, 3), dtype=np.float32),
        next_object=np.zeros((len(states), 1, 3), dtype=np.float32),
        valid_step=np.array([True, True, True, False]),
        episode_index=np.zeros(len(states), dtype=np.int64),
    )
    ckpt_path = tmp_path / "tiny.pt"
    train_world_model(path, ckpt_path, steps=1, batch_size=2, hidden=8, device="cpu", log_every=0)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["target_mode"] == "next_state_delta"
