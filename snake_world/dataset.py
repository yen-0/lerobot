from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .kinematics import JOINT_ORDER, SnakeKinematics, load_default_kinematics


def resolve_dataset_root(dataset: str | Path) -> Path:
    path = Path(dataset)
    if path.exists():
        return path
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME
    except Exception as exc:
        raise RuntimeError(
            f"Dataset path {dataset!r} does not exist and LeRobot is unavailable. "
            "Run inside `pixi run` or pass a local dataset root."
        ) from exc
    root = Path(HF_LEROBOT_HOME) / str(dataset)
    if not root.exists():
        raise FileNotFoundError(f"No local dataset at {root}")
    return root


def _read_parquet_rows(root: Path) -> dict[str, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("snake-export requires pyarrow. Run inside the pixi/LeRobot environment.") from exc

    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root / 'data'}")

    chunks: dict[str, list[np.ndarray]] = {}
    for file in files:
        table = pq.read_table(file)
        for name in table.column_names:
            arr = table[name].combine_chunks().to_numpy(zero_copy_only=False)
            if arr.dtype == object:
                arr = np.stack([np.asarray(x, dtype=np.float32) for x in arr])
            chunks.setdefault(name, []).append(np.asarray(arr))
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def _load_object_tracks(path: str | Path | None, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        return np.zeros((n_frames, 1, 3), dtype=np.float32), np.zeros((n_frames, 1), dtype=bool)
    data = np.load(path)
    points = np.asarray(data["points"], dtype=np.float32)
    valid = np.asarray(data.get("valid", np.ones(points.shape[:2], dtype=bool)), dtype=bool)
    if points.shape[0] != n_frames:
        raise ValueError(f"Object track frames ({points.shape[0]}) do not match dataset frames ({n_frames})")
    if points.shape[-1] == 2:
        z = np.zeros((*points.shape[:-1], 1), dtype=np.float32)
        points = np.concatenate([points, z], axis=-1)
    return points, valid


def export_training_npz(
    dataset: str | Path,
    output: str | Path,
    object_tracks: str | Path | None = None,
    kinematics: SnakeKinematics | None = None,
    max_frames: int | None = None,
) -> Path:
    root = resolve_dataset_root(dataset)
    info = json.loads((root / "meta" / "info.json").read_text())
    state_names = [name.removesuffix(".pos") for name in info["features"]["observation.state"]["names"]]
    if tuple(state_names) != JOINT_ORDER:
        raise ValueError(f"Unexpected joint order {state_names}; expected {list(JOINT_ORDER)}")

    rows = _read_parquet_rows(root)
    states = np.asarray(rows["observation.state"], dtype=np.float32)
    actions = np.asarray(rows["action"], dtype=np.float32)
    episodes = np.asarray(rows["episode_index"], dtype=np.int64).reshape(-1)
    frame_index = np.asarray(rows["frame_index"], dtype=np.int64).reshape(-1)
    if max_frames is not None:
        states = states[:max_frames]
        actions = actions[:max_frames]
        episodes = episodes[:max_frames]
        frame_index = frame_index[:max_frames]

    kin = kinematics or load_default_kinematics()
    object_points, object_valid = _load_object_tracks(object_tracks, len(states))
    arm = kin.batch_forward(states)
    next_arm = np.roll(arm, -1, axis=0)
    next_object = np.roll(object_points, -1, axis=0)
    valid_step = np.roll(episodes, -1) == episodes
    valid_step[-1] = False

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        state=states,
        action=actions,
        arm=arm,
        next_arm=next_arm,
        object=object_points,
        next_object=next_object,
        object_valid=object_valid,
        valid_step=valid_step,
        episode_index=episodes,
        frame_index=frame_index,
        joint_order=np.asarray(JOINT_ORDER),
    )
    return output
