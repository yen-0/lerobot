from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from .dataset import export_training_npz
from .benchmark import build_benchmark_report, write_report
from .kinematics import DEFAULT_URDF, SnakeKinematics, load_default_kinematics
from .model import auto_device, evaluate_npz, evaluate_world_model, train_world_model
from .planner import load_target_snake, solve_snake_target
from .robot import execute_one_step, preview_move
from .tracking import DINOFeatureTracker, parse_points
from .visualize import make_rerun_recording, make_rollout_viewer, make_snake_viewer

app = typer.Typer(add_completion=False, help="Articulated-snake world-model tools for SO-101 datasets.")


def _parse_episodes(text: str | None) -> list[int] | None:
    if text is None or text.strip() == "":
        return None
    return sorted({int(item.strip()) for item in text.split(",") if item.strip()})


@app.command("export")
def export(
    dataset: str = typer.Option(..., "--dataset", help="Local LeRobot dataset root or repo id."),
    output: Path = typer.Option(Path("outputs/snake/train.npz"), "--output"),
    object_tracks: Path | None = typer.Option(None, "--object-tracks", help=".npz from `snake-track-object`."),
    urdf: Path = typer.Option(DEFAULT_URDF, "--urdf"),
    max_frames: int | None = typer.Option(None, "--max-frames"),
) -> None:
    kin = SnakeKinematics.from_urdf(urdf)
    out = export_training_npz(dataset, output, object_tracks=object_tracks, kinematics=kin, max_frames=max_frames)
    typer.secho(f"Wrote {out}", fg="green")


@app.command("track-object")
def track_object(
    video: Path = typer.Option(..., "--video", help="Video file to track."),
    points: str = typer.Option(..., "--points", help="Initial image points as 'x,y;x,y'."),
    output: Path = typer.Option(Path("outputs/snake/object_tracks.npz"), "--output"),
    model: str = typer.Option("facebook/dinov2-small", "--model"),
    max_frames: int | None = typer.Option(None, "--max-frames"),
) -> None:
    tracker = DINOFeatureTracker(model_name=model)
    out = tracker.track_video(video, parse_points(points), output, max_frames=max_frames)
    typer.secho(f"Wrote {out}", fg="green")


@app.command("train")
def train(
    data: Path = typer.Option(..., "--data", help=".npz from `snake-export`."),
    output: Path = typer.Option(Path("outputs/snake/world_model.pt"), "--output"),
    steps: int = typer.Option(50_000, "--steps"),
    batch_size: int = typer.Option(512, "--batch-size"),
    hidden: int = typer.Option(512, "--hidden"),
    lr: float = typer.Option(1e-3, "--lr"),
    device: str = typer.Option("auto", "--device", help="auto, cuda, mps, or cpu."),
    log_every: int = typer.Option(1000, "--log-every"),
    save_every: int = typer.Option(0, "--save-every", help="Write periodic checkpoints every N steps."),
    episodes: str | None = typer.Option(None, "--episodes", help="Comma-separated episode ids to train on; omit for all."),
) -> None:
    resolved = auto_device() if device == "auto" else device
    typer.secho(f"Training on {resolved}: steps={steps} batch_size={batch_size} hidden={hidden}", fg="blue")
    result = train_world_model(
        data,
        output,
        steps=steps,
        batch_size=batch_size,
        hidden=hidden,
        lr=lr,
        device=resolved,
        log_every=log_every,
        save_every=save_every,
        episodes=_parse_episodes(episodes),
    )
    typer.secho(f"Wrote {result.checkpoint} | final_loss={result.final_loss:.6g}", fg="green")


@app.command("eval")
def evaluate(
    data: Path = typer.Option(..., "--data"),
    model: Path | None = typer.Option(None, "--model", help="Optional checkpoint from `snake-train`."),
    device: str = typer.Option("auto", "--device"),
    episodes: str | None = typer.Option(None, "--episodes", help="Comma-separated episode ids to evaluate; omit for all."),
) -> None:
    episode_ids = _parse_episodes(episodes)
    metrics = evaluate_npz(data, episodes=episode_ids)
    if model is not None:
        metrics.update(
            evaluate_world_model(data, model, device=auto_device() if device == "auto" else device, episodes=episode_ids)
        )
    for key, value in metrics.items():
        typer.echo(f"{key}: {value}")


@app.command("benchmark")
def benchmark(
    data: Path = typer.Option(..., "--data", help=".npz from `snake-export`."),
    snake_model: Path | None = typer.Option(None, "--snake-model", help="Optional snake checkpoint."),
    output: Path = typer.Option(Path("outputs/snake/benchmark_report.json"), "--output"),
    episodes: str | None = typer.Option(None, "--episodes", help="Comma-separated episode ids for snake metrics."),
    device: str = typer.Option("auto", "--device"),
    act_policy: Path | None = typer.Option(None, "--act-policy", help="ACT checkpoint/pretrained_model path."),
    diffusion_policy: Path | None = typer.Option(None, "--diffusion-policy", help="Diffusion checkpoint/pretrained_model path."),
    repo_id: str = typer.Option("record-test2", "--repo-id", help="LeRobot repo id for policy-test."),
    policy_steps: int = typer.Option(100, "--policy-steps"),
    policy_episode: int = typer.Option(0, "--policy-episode"),
) -> None:
    resolved = auto_device() if device == "auto" else device
    report = build_benchmark_report(
        data=data,
        snake_model=snake_model,
        device=resolved,
        episodes=_parse_episodes(episodes),
        act_policy=act_policy,
        diffusion_policy=diffusion_policy,
        repo_id=repo_id,
        policy_steps=policy_steps,
        policy_episode=policy_episode,
    )
    out = write_report(report, output)
    typer.echo(out)
    typer.echo(report)


@app.command("plan-offline")
def plan_offline(
    current_joints: str = typer.Option(..., "--current-joints", help="Comma-separated degrees in SO-101 joint order."),
    target_snake: Path = typer.Option(..., "--target-snake", help=".npy/.npz snake target."),
    max_delta: float = typer.Option(5.0, "--max-delta"),
) -> None:
    joints = np.asarray([float(x) for x in current_joints.split(",")], dtype=np.float32)
    step = solve_snake_target(joints, load_target_snake(target_snake), max_delta_deg=max_delta)
    typer.echo(",".join(f"{x:.3f}" for x in step.target_joints_deg))
    typer.echo(f"ik_error={step.ik_error:.6f} max_delta={step.max_delta_deg:.3f}")


@app.command("viz")
def visualize(
    data: Path = typer.Option(..., "--data", help=".npz from `snake-export`."),
    output: Path = typer.Option(Path("outputs/snake/viewer.html"), "--output"),
    model: Path | None = typer.Option(None, "--model", help="Optional checkpoint from `snake-train`."),
    episode: int = typer.Option(0, "--episode"),
    max_frames: int = typer.Option(300, "--max-frames"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    out = make_snake_viewer(
        data,
        output,
        model_path=model,
        episode=episode,
        max_frames=max_frames,
        device=auto_device() if device == "auto" else device,
    )
    typer.secho(f"Wrote {out}", fg="green")


@app.command("viz-rollout")
def visualize_rollout(
    data: Path = typer.Option(..., "--data", help=".npz from `snake-export`."),
    model: Path = typer.Option(..., "--model", help="Checkpoint from `snake-train`."),
    output: Path = typer.Option(Path("outputs/snake/rollout_viewer.html"), "--output"),
    episode: int = typer.Option(0, "--episode"),
    max_frames: int = typer.Option(300, "--max-frames"),
    device: str = typer.Option("auto", "--device"),
    project_kinematics: bool = typer.Option(True, "--project-kinematics/--raw-rollout", help="Project each rollout step back onto the SO-101 kinematic chain."),
) -> None:
    out = make_rollout_viewer(
        data,
        output,
        model,
        episode=episode,
        max_frames=max_frames,
        device=auto_device() if device == "auto" else device,
        project_kinematics=project_kinematics,
    )
    typer.secho(f"Wrote {out}", fg="green")


@app.command("viz-rerun")
def visualize_rerun(
    data: Path = typer.Option(..., "--data", help=".npz from `snake-export`."),
    output: Path = typer.Option(Path("outputs/snake/snake_world.rrd"), "--output"),
    model: Path | None = typer.Option(None, "--model", help="Optional checkpoint from `snake-train`."),
    episode: int = typer.Option(0, "--episode"),
    max_frames: int = typer.Option(300, "--max-frames"),
    device: str = typer.Option("auto", "--device"),
    spawn: bool = typer.Option(False, "--spawn", help="Open the Rerun viewer after writing the recording."),
    project_kinematics: bool = typer.Option(True, "--project-kinematics/--raw-rollout-only", help="Also log a green rollout projected through SO-101 IK/FK."),
) -> None:
    out = make_rerun_recording(
        data,
        output,
        model_path=model,
        episode=episode,
        max_frames=max_frames,
        device=auto_device() if device == "auto" else device,
        spawn=spawn,
        project_kinematics=project_kinematics,
    )
    typer.secho(f"Wrote {out}", fg="green")


@app.command("plan-robot")
def plan_robot(
    target_snake: Path = typer.Option(..., "--target-snake"),
    max_delta: float = typer.Option(3.0, "--max-delta"),
    current_raw: str | None = typer.Option(None, "--current-raw", help="Dry-run raw counts as name=value,... or six comma-separated values."),
    homing_offsets: str | None = typer.Option(None, "--homing-offsets", help="Dry-run homing offsets as name=value,... or six comma-separated values."),
    yes: bool = typer.Option(False, "--yes", help="Skip the interactive confirmation prompt before moving."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not connect to the robot; requires --current-raw."),
) -> None:
    target = load_target_snake(target_snake)
    if dry_run:
        if current_raw is None:
            raise typer.BadParameter("--dry-run requires --current-raw.")
        preview = preview_move(target, _parse_motor_map(current_raw), _parse_motor_map(homing_offsets), max_delta)
    else:
        preview = execute_one_step(target, max_delta_deg=max_delta, confirm=None if yes else _confirm_preview)

    _print_preview(preview)


def _parse_motor_map(text: str | None) -> dict[str, int]:
    from .robot import JOINT_ORDER

    if not text:
        return {name: 0 for name in JOINT_ORDER}
    if "=" not in text:
        vals = [int(float(x.strip())) for x in text.split(",")]
        if len(vals) != len(JOINT_ORDER):
            raise typer.BadParameter(f"Expected {len(JOINT_ORDER)} comma-separated values.")
        return dict(zip(JOINT_ORDER, vals, strict=True))
    out = {name: 0 for name in JOINT_ORDER}
    for item in text.split(","):
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in out:
            raise typer.BadParameter(f"Unknown motor {name!r}; expected {', '.join(JOINT_ORDER)}")
        out[name] = int(float(value.strip()))
    return out


def _print_preview(preview) -> None:
    typer.secho("Guarded one-step target:", fg="green")
    typer.echo("current_deg: " + ",".join(f"{x:.3f}" for x in preview.current_joints_deg))
    typer.echo("target_deg:  " + ",".join(f"{x:.3f}" for x in preview.target_joints_deg))
    typer.echo("delta_deg:   " + ",".join(f"{x:.3f}" for x in preview.delta_deg))
    typer.echo("target_raw:  " + ",".join(f"{k}={v}" for k, v in preview.target_raw.items()))
    typer.echo(f"ik_error={preview.ik_error:.6f} max_delta={preview.max_delta_deg:.3f}")


def _confirm_preview(preview) -> bool:
    typer.secho("Computed guarded one-step target:", fg="yellow")
    _print_preview(preview)
    if not typer.confirm("Write this Goal_Position target to the registered follower?", default=False):
        raise typer.Exit(1)
    return True


if __name__ == "__main__":
    app()
