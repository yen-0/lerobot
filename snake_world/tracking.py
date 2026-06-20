from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def parse_points(text: str) -> np.ndarray:
    pts = []
    for item in text.split(";"):
        if not item.strip():
            continue
        x, y = item.split(",")
        pts.append([float(x), float(y)])
    if not pts:
        raise ValueError("No points supplied; use 'x,y;x,y'.")
    return np.asarray(pts, dtype=np.float32)


@dataclass
class DINOFeatureTracker:
    model_name: str = "facebook/dinov2-small"
    search_radius_px: int = 80

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:
            raise RuntimeError(
                "DINO tracking requires torch and transformers. Install the pixi environment "
                "and add transformers if it is not already present."
            ) from exc
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).eval()

    def _features(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(**inputs).last_hidden_state[:, 1:, :]
        feats = out[0].cpu().numpy()
        size = self.processor.size
        h = size["height"] if isinstance(size, dict) and "height" in size else 224
        w = size["width"] if isinstance(size, dict) and "width" in size else 224
        grid = int(round(feats.shape[0] ** 0.5))
        feats = feats.reshape(grid, grid, -1)
        feats /= np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-9
        scale = np.array([frame_bgr.shape[1] / w, frame_bgr.shape[0] / h], dtype=np.float32)
        return feats, scale

    def _sample(self, feats: np.ndarray, scale: np.ndarray, points: np.ndarray) -> np.ndarray:
        grid = feats.shape[0]
        xy = points / scale
        ij = np.clip(np.round(xy / 224.0 * (grid - 1)).astype(int), 0, grid - 1)
        return feats[ij[:, 1], ij[:, 0]]

    def track_video(self, video: str | Path, initial_points: np.ndarray, output: str | Path, max_frames: int | None = None) -> Path:
        cap = cv2.VideoCapture(str(video))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read first frame from {video}")
        feats, scale = self._features(frame)
        refs = self._sample(feats, scale, initial_points)
        points = [initial_points.astype(np.float32)]
        valid = [np.ones(len(initial_points), dtype=bool)]
        current = initial_points.astype(np.float32)

        frame_count = 1
        while max_frames is None or frame_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            feats, scale = self._features(frame)
            grid = feats.shape[0]
            next_points = []
            next_valid = []
            for point, ref in zip(current, refs, strict=True):
                center = point / scale / 224.0 * (grid - 1)
                radius = max(1, int(self.search_radius_px / float(scale.mean()) / 224.0 * (grid - 1)))
                y0, y1 = max(0, int(center[1]) - radius), min(grid, int(center[1]) + radius + 1)
                x0, x1 = max(0, int(center[0]) - radius), min(grid, int(center[0]) + radius + 1)
                crop = feats[y0:y1, x0:x1]
                sims = crop @ ref
                iy, ix = np.unravel_index(np.argmax(sims), sims.shape)
                score = float(sims[iy, ix])
                grid_xy = np.array([x0 + ix, y0 + iy], dtype=np.float32)
                next_points.append(grid_xy / max(grid - 1, 1) * 224.0 * scale)
                next_valid.append(score > 0.55)
            current = np.asarray(next_points, dtype=np.float32)
            points.append(current)
            valid.append(np.asarray(next_valid, dtype=bool))
            frame_count += 1
        cap.release()

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, points=np.stack(points), valid=np.stack(valid))
        return output
