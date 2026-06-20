from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import flatten_world
from .kinematics import load_default_kinematics

GRIPPER_INDEX = 5


def _gripper_jaw_points(tip: np.ndarray, wrist: np.ndarray, gripper_deg: float) -> np.ndarray:
    tip = np.asarray(tip, dtype=np.float32)
    wrist = np.asarray(wrist, dtype=np.float32)
    forward = tip - wrist
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        forward = forward / norm
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    side = np.cross(forward, up)
    side_norm = float(np.linalg.norm(side))
    if side_norm < 1e-6:
        side = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        side = side / side_norm
    open01 = np.clip((float(gripper_deg) - 0.0) / 50.0, 0.0, 1.0)
    half_width = 0.006 + 0.018 * open01
    length = 0.035
    base = tip - forward * 0.01
    left_base = base + side * half_width
    right_base = base - side * half_width
    left_tip = left_base + forward * length
    right_tip = right_base + forward * length
    return np.asarray([[left_base, left_tip], [right_base, right_tip]], dtype=np.float32)


def _rollout_joints(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    checkpoint: str | Path,
    device: str,
) -> np.ndarray | None:
    torch, model, ckpt = _load_model(checkpoint, device)
    if ckpt.get("target_mode") not in {"joint_delta", "next_state_delta"}:
        return None
    arm_shape = tuple(ckpt["arm_shape"])
    object_shape = tuple(ckpt["object_shape"])
    kin = load_default_kinematics()
    joints = data["state"][indices[0]].astype(np.float32)
    arm = kin.forward(joints)
    obj = data["object"][indices[0]].astype(np.float32)
    rollout = [joints.copy()]
    for idx in indices[:-1]:
        action = data["action"][idx : idx + 1].astype(np.float32)
        x = flatten_world(arm.reshape(1, *arm_shape), obj.reshape(1, *object_shape), action)
        if "x_mean" in ckpt:
            x = (x - ckpt["x_mean"]) / ckpt["x_std"]
        with torch.no_grad():
            pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
        if "y_mean" in ckpt:
            pred = pred * ckpt["y_std"] + ckpt["y_mean"]
        joints = kin.clamp_degrees(joints + pred.reshape(-1)).astype(np.float32)
        arm = kin.forward(joints)
        rollout.append(joints.copy())
    return np.stack(rollout, axis=0)


def _load_model(checkpoint: str | Path, device: str):
    try:
        import torch
        from torch import nn
    except Exception as exc:
        raise RuntimeError("Visualization with --model requires torch.") from exc

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = nn.Sequential(
        nn.Linear(ckpt["input_dim"], ckpt["hidden"]),
        nn.SiLU(),
        nn.Linear(ckpt["hidden"], ckpt["hidden"]),
        nn.SiLU(),
        nn.Linear(ckpt["hidden"], ckpt["output_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return torch, model, ckpt


def _predict_next(data: np.lib.npyio.NpzFile, indices: np.ndarray, checkpoint: str | Path, device: str) -> np.ndarray:
    torch, model, ckpt = _load_model(checkpoint, device)
    x = flatten_world(data["arm"][indices], data["object"][indices], data["action"][indices])
    current = np.concatenate(
        [data["arm"][indices].reshape(len(indices), -1), data["object"][indices].reshape(len(indices), -1)],
        axis=1,
    ).astype(np.float32)
    if "x_mean" in ckpt:
        x = (x - ckpt["x_mean"]) / ckpt["x_std"]
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
    if "y_mean" in ckpt:
        pred = pred * ckpt["y_std"] + ckpt["y_mean"]
    if ckpt.get("target_mode") in {"joint_delta", "next_state_delta"}:
        next_joints = data["state"][indices] + pred
        return load_default_kinematics().batch_forward(next_joints)
    if ckpt.get("target_mode") == "residual":
        pred = current + pred
    arm_size = int(np.prod(ckpt["arm_shape"]))
    return pred[:, :arm_size].reshape((-1, *ckpt["arm_shape"])).astype(np.float32)


def _rollout_arm(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    checkpoint: str | Path,
    device: str,
    project_kinematics: bool = False,
) -> np.ndarray:
    torch, model, ckpt = _load_model(checkpoint, device)
    arm_shape = tuple(ckpt["arm_shape"])
    object_shape = tuple(ckpt["object_shape"])
    arm_size = int(np.prod(arm_shape))
    obj_size = int(np.prod(object_shape))
    arm = data["arm"][indices[0]].astype(np.float32)
    obj = data["object"][indices[0]].astype(np.float32)
    kin = (
        load_default_kinematics()
        if project_kinematics or ckpt.get("target_mode") in {"joint_delta", "next_state_delta"}
        else None
    )
    joints = data["state"][indices[0]].astype(np.float32) if kin is not None else None
    rollout = [arm.copy()]
    for idx in indices[:-1]:
        action = data["action"][idx : idx + 1].astype(np.float32)
        x = flatten_world(arm.reshape(1, *arm_shape), obj.reshape(1, *object_shape), action)
        current = np.concatenate([arm.reshape(1, -1), obj.reshape(1, -1)], axis=1).astype(np.float32)
        if "x_mean" in ckpt:
            x = (x - ckpt["x_mean"]) / ckpt["x_std"]
        with torch.no_grad():
            pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
        if "y_mean" in ckpt:
            pred = pred * ckpt["y_std"] + ckpt["y_mean"]
        if ckpt.get("target_mode") in {"joint_delta", "next_state_delta"}:
            if kin is None or joints is None:
                raise RuntimeError("Joint-state checkpoints require kinematics for rollout.")
            joints = kin.clamp_degrees(joints + pred.reshape(-1)).astype(np.float32)
            arm = kin.forward(joints)
            rollout.append(arm.copy())
            continue
        if ckpt.get("target_mode") == "residual":
            pred = current + pred
        predicted_arm = pred[:, :arm_size].reshape(arm_shape).astype(np.float32)
        if kin is not None and joints is not None:
            joints, _ = kin.ik(predicted_arm, joints, max_iters=25)
            arm = kin.forward(joints)
        else:
            arm = predicted_arm
        obj = pred[:, arm_size : arm_size + obj_size].reshape(object_shape).astype(np.float32)
        rollout.append(arm.copy())
    return np.stack(rollout, axis=0)


def make_snake_viewer(
    data_path: str | Path,
    output: str | Path,
    model_path: str | Path | None = None,
    episode: int = 0,
    max_frames: int = 300,
    device: str = "cpu",
) -> Path:
    data = np.load(data_path, allow_pickle=True)
    episode_indices = np.flatnonzero(data["episode_index"] == episode)
    if len(episode_indices) == 0:
        raise ValueError(f"Episode {episode} not found.")
    indices = episode_indices[:max_frames]
    arm = data["arm"][indices].astype(np.float32)
    next_arm = data["next_arm"][indices].astype(np.float32)
    valid = data["valid_step"][indices].astype(bool)
    pred_arm = _predict_next(data, indices, model_path, device) if model_path else None

    points = np.concatenate([arm.reshape(-1, 3), next_arm.reshape(-1, 3)], axis=0)
    if pred_arm is not None:
        points = np.concatenate([points, pred_arm.reshape(-1, 3)], axis=0)
    center = points.mean(axis=0)
    span = float(np.max(np.ptp(points, axis=0)))
    span = span if span > 1e-6 else 1.0

    payload = {
        "episode": int(episode),
        "center": center.tolist(),
        "span": span,
        "arm": arm.tolist(),
        "next_arm": next_arm.tolist(),
        "pred_arm": pred_arm.tolist() if pred_arm is not None else None,
        "valid": valid.tolist(),
    }
    html = _HTML.replace("__PAYLOAD__", json.dumps(payload))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def make_rollout_viewer(
    data_path: str | Path,
    output: str | Path,
    model_path: str | Path,
    episode: int = 0,
    max_frames: int = 300,
    device: str = "cpu",
    project_kinematics: bool = True,
) -> Path:
    data = np.load(data_path, allow_pickle=True)
    episode_indices = np.flatnonzero(data["episode_index"] == episode)
    if len(episode_indices) == 0:
        raise ValueError(f"Episode {episode} not found.")
    indices = episode_indices[:max_frames]
    recorded = data["arm"][indices].astype(np.float32)
    rollout = _rollout_arm(data, indices, model_path, device, project_kinematics=project_kinematics)
    rollout_joints = _rollout_joints(data, indices, model_path, device)
    recorded_gripper = data["state"][indices, GRIPPER_INDEX].astype(np.float32)
    rollout_gripper = rollout_joints[:, GRIPPER_INDEX].astype(np.float32) if rollout_joints is not None else recorded_gripper
    points = np.concatenate([recorded.reshape(-1, 3), rollout.reshape(-1, 3)], axis=0)
    center = points.mean(axis=0)
    span = float(np.max(np.ptp(points, axis=0)))
    span = span if span > 1e-6 else 1.0
    tip_err = np.linalg.norm(rollout[:, -1] - recorded[:, -1], axis=-1)
    mean_curve_err = np.linalg.norm(rollout - recorded, axis=-1).mean(axis=-1)

    payload = {
        "episode": int(episode),
        "center": center.tolist(),
        "span": span,
        "recorded": recorded.tolist(),
        "rollout": rollout.tolist(),
        "recorded_gripper": recorded_gripper.tolist(),
        "rollout_gripper": rollout_gripper.tolist(),
        "tip_err": tip_err.tolist(),
        "mean_curve_err": mean_curve_err.tolist(),
    }
    html = _ROLLOUT_HTML.replace("__PAYLOAD__", json.dumps(payload))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def make_rerun_recording(
    data_path: str | Path,
    output: str | Path,
    model_path: str | Path | None = None,
    episode: int = 0,
    max_frames: int = 300,
    device: str = "cpu",
    spawn: bool = False,
    project_kinematics: bool = True,
) -> Path:
    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("Rerun visualization requires the `rerun-sdk` package.") from exc

    data = np.load(data_path, allow_pickle=True)
    episode_indices = np.flatnonzero(data["episode_index"] == episode)
    if len(episode_indices) == 0:
        raise ValueError(f"Episode {episode} not found.")
    indices = episode_indices[:max_frames]
    recorded = data["arm"][indices].astype(np.float32)
    one_step = _predict_next(data, indices, model_path, device) if model_path else None
    rollout = _rollout_arm(data, indices, model_path, device) if model_path else None
    rollout_joints = _rollout_joints(data, indices, model_path, device) if model_path else None
    projected_rollout = (
        _rollout_arm(data, indices, model_path, device, project_kinematics=True)
        if model_path and project_kinematics
        else None
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rr.init("snake_world_model", spawn=spawn)
    rr.save(output)
    rr.log("world/axes", rr.Arrows3D(vectors=[[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]], colors=[[255, 80, 80], [80, 255, 120], [80, 160, 255]], labels=["x", "y", "z"]), static=True)
    rr.log("world/base", rr.Points3D([[0, 0, 0]], radii=[0.008], colors=[[255, 255, 255]], labels=["base"]), static=True)

    recorded_tip_trail: list[list[float]] = []
    rollout_tip_trail: list[list[float]] = []
    for frame, idx in enumerate(indices):
        rr.set_time("frame", sequence=frame)
        rec = recorded[frame]
        recorded_tip_trail.append(rec[-1].tolist())
        rr.log(
            "world/recorded/snake",
            rr.LineStrips3D([rec], radii=[0.004], colors=[[240, 193, 92]]),
        )
        rr.log(
            "world/recorded/joints",
            rr.Points3D(rec, radii=[0.008], colors=[[240, 193, 92]]),
        )
        rr.log(
            "world/recorded/tip_trail",
            rr.LineStrips3D([recorded_tip_trail], radii=[0.002], colors=[[240, 193, 92, 150]]),
        )
        rr.log(
            "world/recorded/gripper",
            rr.LineStrips3D(
                _gripper_jaw_points(rec[-1], rec[-2], float(data["state"][idx, GRIPPER_INDEX])),
                radii=[0.003],
                colors=[[240, 193, 92]],
            ),
        )
        if one_step is not None:
            pred = one_step[frame]
            rr.log(
                "world/one_step_prediction/snake",
                rr.LineStrips3D([pred], radii=[0.003], colors=[[93, 213, 215]]),
            )
            rr.log(
                "world/one_step_prediction/joints",
                rr.Points3D(pred, radii=[0.006], colors=[[93, 213, 215]]),
            )
        if rollout is not None:
            roll = rollout[frame]
            rollout_tip_trail.append(roll[-1].tolist())
            rr.log(
                "world/model_rollout/snake",
                rr.LineStrips3D([roll], radii=[0.004], colors=[[255, 106, 95]]),
            )
            rr.log(
                "world/model_rollout/joints",
                rr.Points3D(roll, radii=[0.007], colors=[[255, 106, 95]]),
            )
            rr.log(
                "world/model_rollout/tip_trail",
                rr.LineStrips3D([rollout_tip_trail], radii=[0.002], colors=[[108, 168, 255, 150]]),
            )
            gripper_value = (
                float(rollout_joints[frame, GRIPPER_INDEX])
                if rollout_joints is not None
                else float(data["state"][idx, GRIPPER_INDEX])
            )
            rr.log(
                "world/model_rollout/gripper",
                rr.LineStrips3D(
                    _gripper_jaw_points(roll[-1], roll[-2], gripper_value),
                    radii=[0.003],
                    colors=[[255, 106, 95]],
                ),
            )
        if projected_rollout is not None:
            projected = projected_rollout[frame]
            rr.log(
                "world/kinematic_projected_rollout/snake",
                rr.LineStrips3D([projected], radii=[0.004], colors=[[80, 255, 150]]),
            )
            rr.log(
                "world/kinematic_projected_rollout/joints",
                rr.Points3D(projected, radii=[0.007], colors=[[80, 255, 150]]),
            )
            gripper_value = (
                float(rollout_joints[frame, GRIPPER_INDEX])
                if rollout_joints is not None
                else float(data["state"][idx, GRIPPER_INDEX])
            )
            rr.log(
                "world/kinematic_projected_rollout/gripper",
                rr.LineStrips3D(
                    _gripper_jaw_points(projected[-1], projected[-2], gripper_value),
                    radii=[0.003],
                    colors=[[80, 255, 150]],
                ),
            )
        rr.log(
            "metrics/frame",
            rr.Scalars(float(frame)),
        )
        if rollout is not None:
            rr.log("metrics/tip_error_m", rr.Scalars(float(np.linalg.norm(rollout[frame, -1] - rec[-1]))))
            rr.log("metrics/mean_curve_error_m", rr.Scalars(float(np.linalg.norm(rollout[frame] - rec, axis=-1).mean())))
        if projected_rollout is not None:
            rr.log(
                "metrics/projected_tip_error_m",
                rr.Scalars(float(np.linalg.norm(projected_rollout[frame, -1] - rec[-1]))),
            )
            rr.log(
                "metrics/projected_mean_curve_error_m",
                rr.Scalars(float(np.linalg.norm(projected_rollout[frame] - rec, axis=-1).mean())),
            )
        if one_step is not None and frame < len(one_step):
            next_rec = data["next_arm"][idx].astype(np.float32)
            rr.log("metrics/one_step_mean_error_m", rr.Scalars(float(np.linalg.norm(one_step[frame] - next_rec, axis=-1).mean())))

    return output


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Snake World Model Viewer</title>
  <style>
    :root { color-scheme: dark; --bg: #101412; --ink: #e8f3ec; --muted: #9eb5a8; --gold: #e7b955; --cyan: #5dd5d7; --red: #f26d64; }
    body { margin: 0; font-family: Georgia, "Times New Roman", serif; background: radial-gradient(circle at 20% 0%, #233229, var(--bg) 45%); color: var(--ink); }
    main { max-width: 1100px; margin: 0 auto; padding: 28px; }
    h1 { margin: 0 0 8px; font-size: 32px; font-weight: 500; letter-spacing: -0.03em; }
    p { margin: 0 0 18px; color: var(--muted); }
    canvas { width: 100%; height: 680px; display: block; background: linear-gradient(180deg, #172019, #0c100e); border: 1px solid #314035; border-radius: 18px; box-shadow: 0 24px 80px #0008; }
    .panel { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin: 18px 0; padding: 14px 16px; background: #162018cc; border: 1px solid #314035; border-radius: 14px; }
    input[type=range] { width: min(520px, 90vw); }
    code { color: var(--ink); background: #263128; padding: 2px 6px; border-radius: 6px; }
    .legend { display: flex; gap: 14px; flex-wrap: wrap; }
    .swatch { display: inline-block; width: 11px; height: 11px; border-radius: 99px; margin-right: 6px; }
  </style>
</head>
<body>
  <main>
    <h1>Kinematic Snake World Model</h1>
    <p>Episode <code id="episode"></code>. Gold is current recorded snake, cyan is recorded next snake, red is model-predicted next snake.</p>
    <canvas id="view" width="1400" height="820"></canvas>
    <div class="panel">
      <label>Frame <input id="frame" type="range" min="0" max="0" value="0" /></label>
      <code id="readout"></code>
      <button id="play">Play</button>
      <div class="legend">
        <span><i class="swatch" style="background:var(--gold)"></i>current</span>
        <span><i class="swatch" style="background:var(--cyan)"></i>recorded next</span>
        <span><i class="swatch" style="background:var(--red)"></i>model next</span>
      </div>
    </div>
  </main>
<script>
const payload = __PAYLOAD__;
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const slider = document.getElementById("frame");
const readout = document.getElementById("readout");
const play = document.getElementById("play");
document.getElementById("episode").textContent = payload.episode;
slider.max = payload.arm.length - 1;
let playing = false;

function project(p) {
  const x = (p[0] - payload.center[0]) / payload.span;
  const y = (p[1] - payload.center[1]) / payload.span;
  const z = (p[2] - payload.center[2]) / payload.span;
  const yaw = -0.78, pitch = 0.58;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x - sy * y;
  const y1 = sy * x + cy * y;
  const z1 = z;
  const y2 = cp * y1 - sp * z1;
  const z2 = sp * y1 + cp * z1;
  const scale = 520 / (1.8 + z2);
  return [canvas.width / 2 + x1 * scale, canvas.height / 2 - y2 * scale];
}

function drawSnake(points, color, width, alpha = 1) {
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  points.map(project).forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  ctx.stroke();
  for (const p of points.map(project)) {
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(p[0], p[1], width + 2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function drawGripper(points, gripDeg, color, width, alpha = 1) {
  const tip = points[points.length - 1];
  const wrist = points[points.length - 2];
  let f = [tip[0] - wrist[0], tip[1] - wrist[1], tip[2] - wrist[2]];
  const fn = Math.hypot(f[0], f[1], f[2]) || 1;
  f = [f[0] / fn, f[1] / fn, f[2] / fn];
  let side = [f[1], -f[0], 0];
  const sn = Math.hypot(side[0], side[1], side[2]) || 1;
  side = [side[0] / sn, side[1] / sn, side[2] / sn];
  const open01 = Math.max(0, Math.min(1, gripDeg / 50));
  const half = 0.006 + 0.018 * open01;
  const len = 0.035;
  const base = [tip[0] - f[0] * 0.01, tip[1] - f[1] * 0.01, tip[2] - f[2] * 0.01];
  const strips = [
    [[base[0] + side[0] * half, base[1] + side[1] * half, base[2] + side[2] * half], [base[0] + side[0] * half + f[0] * len, base[1] + side[1] * half + f[1] * len, base[2] + side[2] * half + f[2] * len]],
    [[base[0] - side[0] * half, base[1] - side[1] * half, base[2] - side[2] * half], [base[0] - side[0] * half + f[0] * len, base[1] - side[1] * half + f[1] * len, base[2] - side[2] * half + f[2] * len]],
  ];
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  for (const strip of strips) {
    const a = project(strip[0]), b = project(strip[1]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function drawGrid() {
  ctx.strokeStyle = "#2d3a30";
  ctx.lineWidth = 1;
  for (let i = -5; i <= 5; i++) {
    const a = project([payload.center[0] + i * payload.span / 8, payload.center[1] - payload.span / 2, payload.center[2] - payload.span / 2]);
    const b = project([payload.center[0] + i * payload.span / 8, payload.center[1] + payload.span / 2, payload.center[2] - payload.span / 2]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
}

function draw() {
  const i = Number(slider.value);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid();
  drawSnake(payload.next_arm[i], "#5dd5d7", 4, payload.valid[i] ? 0.8 : 0.25);
  if (payload.pred_arm) drawSnake(payload.pred_arm[i], "#f26d64", 4, payload.valid[i] ? 0.9 : 0.25);
  drawSnake(payload.arm[i], "#e7b955", 6, 1);
  readout.textContent = `${i + 1}/${payload.arm.length} valid_next=${payload.valid[i]}`;
}

slider.addEventListener("input", draw);
play.addEventListener("click", () => {
  playing = !playing;
  play.textContent = playing ? "Pause" : "Play";
});
setInterval(() => {
  if (!playing) return;
  slider.value = (Number(slider.value) + 1) % payload.arm.length;
  draw();
}, 80);
draw();
</script>
</body>
</html>
"""


_ROLLOUT_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Snake Rollout Viewer</title>
  <style>
    :root { color-scheme: dark; --bg: #0e1318; --ink: #edf4f8; --muted: #9fb2bd; --gold: #f0c15c; --red: #ff6a5f; --blue: #6ca8ff; }
    body { margin: 0; font-family: Georgia, "Times New Roman", serif; background: radial-gradient(circle at 70% 0%, #253849, var(--bg) 48%); color: var(--ink); }
    main { max-width: 1160px; margin: 0 auto; padding: 28px; }
    h1 { margin: 0 0 8px; font-size: 34px; font-weight: 500; letter-spacing: -0.035em; }
    p { margin: 0 0 18px; color: var(--muted); }
    canvas { width: 100%; height: 720px; display: block; background: linear-gradient(180deg, #141d25, #090d10); border: 1px solid #304554; border-radius: 18px; box-shadow: 0 24px 80px #0009; }
    .panel { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin: 18px 0; padding: 14px 16px; background: #13202acc; border: 1px solid #304554; border-radius: 14px; }
    input[type=range] { width: min(520px, 90vw); }
    code { color: var(--ink); background: #22313b; padding: 2px 6px; border-radius: 6px; }
    .legend { display: flex; gap: 14px; flex-wrap: wrap; }
    .swatch { display: inline-block; width: 11px; height: 11px; border-radius: 99px; margin-right: 6px; }
  </style>
</head>
<body>
  <main>
    <h1>Recorded vs Model Rollout</h1>
    <p>Both trajectories start from the same initial folded snake. Gold is the recorded trajectory; red is the model-generated rollout using recorded actions.</p>
    <canvas id="view" width="1500" height="860"></canvas>
    <div class="panel">
      <label>Frame <input id="frame" type="range" min="0" max="0" value="0" /></label>
      <code id="readout"></code>
      <button id="play">Play</button>
      <div class="legend">
        <span><i class="swatch" style="background:var(--gold)"></i>recorded</span>
        <span><i class="swatch" style="background:var(--red)"></i>model rollout</span>
        <span><i class="swatch" style="background:var(--blue)"></i>trajectory trails</span>
      </div>
    </div>
  </main>
<script>
const payload = __PAYLOAD__;
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const slider = document.getElementById("frame");
const readout = document.getElementById("readout");
const play = document.getElementById("play");
slider.max = payload.recorded.length - 1;
let playing = false;

function project(p) {
  const x = (p[0] - payload.center[0]) / payload.span;
  const y = (p[1] - payload.center[1]) / payload.span;
  const z = (p[2] - payload.center[2]) / payload.span;
  const yaw = -0.82, pitch = 0.56;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x - sy * y;
  const y1 = sy * x + cy * y;
  const z1 = z;
  const y2 = cp * y1 - sp * z1;
  const z2 = sp * y1 + cp * z1;
  const scale = 560 / (1.9 + z2);
  return [canvas.width / 2 + x1 * scale, canvas.height / 2 - y2 * scale];
}

function drawSnake(points, color, width, alpha = 1) {
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  points.map(project).forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  ctx.stroke();
  for (const p of points.map(project)) {
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(p[0], p[1], width + 1.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function drawTrail(series, color, upto) {
  ctx.globalAlpha = 0.28;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= upto; i++) {
    const tip = project(series[i][series[i].length - 1]);
    if (i === 0) ctx.moveTo(tip[0], tip[1]); else ctx.lineTo(tip[0], tip[1]);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawGrid() {
  ctx.strokeStyle = "#263844";
  ctx.lineWidth = 1;
  for (let i = -5; i <= 5; i++) {
    const a = project([payload.center[0] + i * payload.span / 8, payload.center[1] - payload.span / 2, payload.center[2] - payload.span / 2]);
    const b = project([payload.center[0] + i * payload.span / 8, payload.center[1] + payload.span / 2, payload.center[2] - payload.span / 2]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
}

function draw() {
  const i = Number(slider.value);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid();
  drawTrail(payload.recorded, "#f0c15c", i);
  drawTrail(payload.rollout, "#6ca8ff", i);
  drawSnake(payload.recorded[i], "#f0c15c", 6, 1);
  drawSnake(payload.rollout[i], "#ff6a5f", 5, 0.95);
  drawGripper(payload.recorded[i], payload.recorded_gripper[i], "#f0c15c", 4, 1);
  drawGripper(payload.rollout[i], payload.rollout_gripper[i], "#ff6a5f", 4, 0.95);
  readout.textContent = `${i + 1}/${payload.recorded.length} tip_err=${(payload.tip_err[i] * 1000).toFixed(2)}mm mean_curve_err=${(payload.mean_curve_err[i] * 1000).toFixed(2)}mm grip_recorded=${payload.recorded_gripper[i].toFixed(1)}deg grip_model=${payload.rollout_gripper[i].toFixed(1)}deg`;
}

slider.addEventListener("input", draw);
play.addEventListener("click", () => {
  playing = !playing;
  play.textContent = playing ? "Pause" : "Play";
});
setInterval(() => {
  if (!playing) return;
  slider.value = (Number(slider.value) + 1) % payload.recorded.length;
  draw();
}, 80);
draw();
</script>
</body>
</html>
"""
