from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from .kinematics import JOINT_ORDER


@dataclass(frozen=True)
class TrainResult:
    checkpoint: Path
    final_loss: float


def _torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError("snake-train requires torch. Run inside the pixi/LeRobot environment.") from exc
    return torch, nn, DataLoader, TensorDataset


def auto_device() -> str:
    torch, _, _, _ = _torch()
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def flatten_world(arm: np.ndarray, obj: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.concatenate([arm.reshape(len(arm), -1), obj.reshape(len(obj), -1), action], axis=1).astype(np.float32)


def _episode_mask(data: np.lib.npyio.NpzFile, episodes: list[int] | None = None) -> np.ndarray:
    mask = np.asarray(data["valid_step"], dtype=bool)
    if episodes is not None:
        mask &= np.isin(data["episode_index"], np.asarray(episodes, dtype=np.int64))
    return mask


def train_world_model(
    data_path: str | Path,
    output: str | Path,
    steps: int = 50_000,
    batch_size: int = 256,
    hidden: int = 512,
    lr: float = 1e-3,
    device: str = "auto",
    log_every: int = 1000,
    save_every: int = 0,
    episodes: list[int] | None = None,
) -> TrainResult:
    torch, nn, DataLoader, TensorDataset = _torch()
    if device == "auto":
        device = auto_device()
    data = np.load(data_path, allow_pickle=True)
    mask = _episode_mask(data, episodes)
    x = flatten_world(data["arm"][mask], data["object"][mask], data["action"][mask])
    next_state = np.roll(data["state"], -1, axis=0)
    y = (next_state[mask] - data["state"][mask]).astype(np.float32)
    x_mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = (x.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    y_mean = y.mean(axis=0, keepdims=True).astype(np.float32)
    y_std = (y.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    x_train = (x - x_mean) / x_std
    y_train = (y - y_mean) / y_std

    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=True,
        drop_last=False,
        pin_memory=device == "cuda",
    )
    model = nn.Sequential(
        nn.Linear(x.shape[1], hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, y.shape[1]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    output = Path(output)

    def save_checkpoint(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "input_dim": x.shape[1],
                "output_dim": y.shape[1],
                "hidden": hidden,
                "arm_shape": tuple(data["arm"].shape[1:]),
                "object_shape": tuple(data["object"].shape[1:]),
                "target_mode": "next_state_delta",
                "device": device,
                "steps": step,
                "batch_size": batch_size,
                "train_episodes": episodes,
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
            },
            path,
        )

    final = 0.0
    started = time.perf_counter()
    it = iter(loader)
    step = 0
    for step in range(1, steps + 1):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        xb = xb.to(device, non_blocking=device == "cuda")
        yb = yb.to(device, non_blocking=device == "cuda")
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final = float(loss.detach().cpu())
        if log_every and (step == 1 or step % log_every == 0 or step == steps):
            elapsed = time.perf_counter() - started
            print(f"step={step} loss={final:.6g} device={device} elapsed_s={elapsed:.1f}", flush=True)
        if save_every and step % save_every == 0 and step != steps:
            periodic = output.with_name(f"{output.stem}.step{step:06d}{output.suffix}")
            save_checkpoint(periodic)
            print(f"saved={periodic}", flush=True)

    save_checkpoint(output)
    return TrainResult(output, final)


def evaluate_npz(data_path: str | Path, episodes: list[int] | None = None) -> dict[str, float]:
    data = np.load(data_path, allow_pickle=True)
    mask = _episode_mask(data, episodes)
    arm_err = np.linalg.norm(data["next_arm"][mask] - data["arm"][mask], axis=-1).mean()
    obj_err = np.linalg.norm(data["next_object"][mask] - data["object"][mask], axis=-1).mean()
    return {"zero_motion_arm_m": float(arm_err), "zero_motion_object_units": float(obj_err), "steps": int(mask.sum())}


def evaluate_world_model(
    data_path: str | Path,
    checkpoint: str | Path,
    device: str = "cpu",
    episodes: list[int] | None = None,
) -> dict[str, float]:
    torch, nn, _, _ = _torch()
    from .kinematics import load_default_kinematics

    data = np.load(data_path, allow_pickle=True)
    mask = _episode_mask(data, episodes)
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

    x = flatten_world(data["arm"][mask], data["object"][mask], data["action"][mask])
    target_arm = data["next_arm"][mask]
    target_obj = data["next_object"][mask]
    with torch.no_grad():
        if "x_mean" in ckpt:
            x = (x - ckpt["x_mean"]) / ckpt["x_std"]
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
        if "y_mean" in ckpt:
            pred = pred * ckpt["y_std"] + ckpt["y_mean"]
        if ckpt.get("target_mode") == "residual":
            current_world = np.concatenate(
                [data["arm"][mask].reshape(mask.sum(), -1), data["object"][mask].reshape(mask.sum(), -1)],
                axis=1,
            ).astype(np.float32)
            pred = current_world + pred
        elif ckpt.get("target_mode") in {"joint_delta", "next_state_delta"}:
            next_joints = data["state"][mask] + pred
            pred_arm = load_default_kinematics().batch_forward(next_joints)
            pred_obj = data["object"][mask]
            next_state = np.roll(data["state"], -1, axis=0)
            joint_target = data["action"][mask] if ckpt.get("target_mode") == "joint_delta" else next_state[mask]
            per_joint_mae = np.abs(next_joints - joint_target).mean(axis=0)
            metrics = {
                f"model_joint_mae_deg/{name}": float(value)
                for name, value in zip(JOINT_ORDER, per_joint_mae, strict=True)
            }
            metrics.update(
                {
                    "model_arm_m": float(np.linalg.norm(pred_arm - target_arm, axis=-1).mean()),
                    "model_object_units": float(np.linalg.norm(pred_obj - target_obj, axis=-1).mean()),
                    "model_joint_deg_mae": float(per_joint_mae.mean()),
                    "model_gripper_deg_mae": float(per_joint_mae[JOINT_ORDER.index("gripper")]),
                    "steps": int(mask.sum()),
                }
            )
            return {
                key: metrics[key]
                for key in [
                    "model_arm_m",
                    "model_object_units",
                    "model_joint_deg_mae",
                    "model_gripper_deg_mae",
                    *[f"model_joint_mae_deg/{name}" for name in JOINT_ORDER],
                    "steps",
                ]
            }

    arm_size = int(np.prod(ckpt["arm_shape"]))
    pred_arm = pred[:, :arm_size].reshape((-1, *ckpt["arm_shape"]))
    pred_obj = pred[:, arm_size:].reshape((-1, *ckpt["object_shape"]))
    return {
        "model_arm_m": float(np.linalg.norm(pred_arm - target_arm, axis=-1).mean()),
        "model_object_units": float(np.linalg.norm(pred_obj - target_obj, axis=-1).mean()),
        "model_mse": float(np.mean((pred - np.concatenate([target_arm.reshape(len(target_arm), -1), target_obj.reshape(len(target_obj), -1)], axis=1)) ** 2)),
        "steps": int(mask.sum()),
    }
