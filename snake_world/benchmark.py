from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .model import evaluate_npz, evaluate_world_model


def _parse_policy_test(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    match = re.search(r"mean \|action - recorded\| = ([0-9.]+) deg", output)
    if match:
        metrics["policy_action_mae_deg"] = float(match.group(1))
    hz = re.search(r"\(~([0-9.]+) Hz\)", output)
    if hz:
        metrics["policy_inference_hz"] = float(hz.group(1))
    return metrics


def evaluate_lerobot_policy(policy_path: str | Path, repo_id: str, device: str, steps: int, episode: int) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/so101.py",
        "policy-test",
        "--policy",
        str(policy_path),
        "--repo-id",
        repo_id,
        "--device",
        device,
        "--steps",
        str(steps),
        "--episode",
        str(episode),
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=600)
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}", "command": cmd}
    output = f"{proc.stdout}\n{proc.stderr}".strip()
    if proc.returncode != 0:
        reason = output.splitlines()[-1] if output else f"exit_code={proc.returncode}"
        if "No module named 'lerobot'" in output:
            reason = "missing lerobot in current Python; run via pixi"
        return {"status": "failed", "reason": reason, "command": cmd}
    return {"status": "ok", "command": cmd, **_parse_policy_test(output)}


def build_benchmark_report(
    data: str | Path,
    snake_model: str | Path | None,
    device: str,
    episodes: list[int] | None,
    act_policy: str | Path | None,
    diffusion_policy: str | Path | None,
    repo_id: str,
    policy_steps: int,
    policy_episode: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "data": str(data),
        "episodes": episodes or "all",
        "snake": {"zero_motion": evaluate_npz(data, episodes=episodes)},
        "baselines": {},
    }
    if snake_model is not None:
        report["snake"]["model"] = {
            "checkpoint": str(snake_model),
            **evaluate_world_model(data, snake_model, device=device, episodes=episodes),
        }
    if act_policy is not None:
        report["baselines"]["act"] = evaluate_lerobot_policy(act_policy, repo_id, device, policy_steps, policy_episode)
    if diffusion_policy is not None:
        report["baselines"]["diffusion"] = evaluate_lerobot_policy(
            diffusion_policy, repo_id, device, policy_steps, policy_episode
        )
    return report


def write_report(report: dict[str, Any], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output
