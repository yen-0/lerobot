# Kinematic Snake World Model

This repo now includes a separate research pipeline for representing the SO-101
arm as a 3D articulated line string and training a lightweight world model over
arm geometry plus object keypoints.

## Commands

Export a LeRobot dataset into compact snake tensors:

```bash
pixi run snake-export --dataset data/record-test2 --output outputs/snake/train.npz
```

Track object keypoints from a video using DINOv2 features:

```bash
pixi run snake-track-object \
  --video data/record-test2/videos/observation.images.wrist/chunk-000/file-000.mp4 \
  --points "320,240;360,240" \
  --output outputs/snake/object_tracks.npz
```

Export with object tracks:

```bash
pixi run snake-export \
  --dataset data/record-test2 \
  --object-tracks outputs/snake/object_tracks.npz \
  --output outputs/snake/train.npz
```

Train and evaluate a physically valid next-state world model:

```bash
pixi run snake-train --data outputs/snake/train.npz --output outputs/snake/world_model.pt
pixi run snake-eval --data outputs/snake/train.npz --model outputs/snake/world_model.pt --device auto
```

`snake-train` predicts observed `state[t+1] - state[t]`, using `action[t]` as an
input. The predicted next state is decoded through the SO-101 URDF FK, so model
rollouts stay on the articulated-chain manifold. `--device auto` selects CUDA
when available. For a longer run with periodic checkpoints:

```bash
pixi run snake-train \
  --data outputs/snake/train.npz \
  --output outputs/snake/world_model_long.pt \
  --steps 200000 \
  --device auto \
  --save-every 25000
```

Evaluate a held-out episode split:

```bash
pixi run snake-train \
  --data outputs/snake/train.npz \
  --output outputs/snake/world_model_split.pt \
  --episodes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

pixi run snake-eval \
  --data outputs/snake/train.npz \
  --model outputs/snake/world_model_split.pt \
  --episodes 16,17,18,19
```

Generate a consolidated offline benchmark report. ACT/diffusion policy metrics
require running inside the LeRobot/Pixi environment:

```bash
pixi run snake-benchmark \
  --data outputs/snake/train.npz \
  --snake-model outputs/snake/world_model_split.pt \
  --act-policy data/record-test2/train/act_record-test2-act/checkpoints/040000 \
  --diffusion-policy data/record-test2/train/diffusion_record-test2/checkpoints/015000 \
  --episodes 16,17,18,19 \
  --output outputs/snake/benchmark_report.json
```

Create an HTML viewer comparing recorded and predicted snake motion:

```bash
pixi run snake-viz \
  --data outputs/snake/train.npz \
  --model outputs/snake/world_model.pt \
  --output outputs/snake/viewer.html
```

Create an HTML viewer comparing the recorded trajectory against a model rollout
from the same initial folded snake:

```bash
pixi run snake-viz-rollout \
  --data outputs/snake/train.npz \
  --model outputs/snake/world_model.pt \
  --output outputs/snake/rollout_viewer.html
```

Create a Rerun recording with recorded, one-step predicted, and rollout
trajectories:

```bash
pixi run snake-viz-rerun \
  --data outputs/snake/train.npz \
  --model outputs/snake/world_model.pt \
  --output outputs/snake/snake_world.rrd
```

Open it with Rerun, or add `--spawn` on a desktop machine to open the viewer
immediately.

The Rerun recording includes a raw learned rollout and, by default, a green
`kinematic_projected_rollout` that projects every predicted step through SO-101
IK/FK. If the raw rollout stretches into impossible link lengths, use the green
track to inspect the physically valid interpretation.

Solve a desired snake target back to bounded joint targets without touching the
robot:

```bash
pixi run snake-plan-offline \
  --current-joints "0,0,0,0,0,0" \
  --target-snake outputs/snake/target_snake.npy \
  --max-delta 3
```

Move the registered follower by exactly one guarded step:

```bash
pixi run snake-plan-robot \
  --target-snake outputs/snake/target_snake.npy \
  --max-delta 3
```

Use `--yes` to skip the confirmation prompt after you have verified the target
and have physical power cut-off within reach.

## Notes

- The default kinematic source is `config/so101/so101_new_calib.urdf`.
- LeRobot state/action values are interpreted as degrees in SO-101 joint order.
- Object keypoints are currently tracked in image coordinates with `z=0`; this is
  enough to train arm-plus-object losses but not metric 3D object dynamics.
- `snake-plan-robot` reads the registered follower from `.so101_arms.json`, reads
  current raw motor positions, computes bounded IK deltas, writes one raw
  `Goal_Position` target per motor, and disconnects without disabling torque.
