# SO-101 Link Geometry

This repo's local SO-101 workflow uses LeRobot's `so101_leader` and
`so101_follower` device types for motor-level control. It does not vendor robot
link geometry. The upstream open-source geometry lives in TheRobotStudio's
SO-ARM100 repository:

- Repository: https://github.com/TheRobotStudio/SO-ARM100
- SO-101 simulation files: https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
- New-calibration URDF: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf

The SO-101 simulation folder includes URDF, MuJoCo XML, joint properties, and
STL assets generated from the Onshape CAD model via `onshape-to-robot`.

## Joint Chain From `so101_new_calib.urdf`

All origins are URDF joint origins in meters and radians.

| Parent link | Child link | Joint | Type | Origin xyz | Origin rpy | Axis | Limits rad |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `base_link` | `shoulder_link` | `shoulder_pan` | revolute | `0.0388353 -8.97657e-09 0.0624` | `3.14159 4.18253e-17 -3.14159` | `0 0 1` | `[-1.91986, 1.91986]` |
| `shoulder_link` | `upper_arm_link` | `shoulder_lift` | revolute | `-0.0303992 -0.0182778 -0.0542` | `-1.5708 -1.5708 0` | `0 0 1` | `[-1.74533, 1.74533]` |
| `upper_arm_link` | `lower_arm_link` | `elbow_flex` | revolute | `-0.11257 -0.028 1.73763e-16` | `-3.63608e-16 8.74301e-16 1.5708` | `0 0 1` | `[-1.69, 1.69]` |
| `lower_arm_link` | `wrist_link` | `wrist_flex` | revolute | `-0.1349 0.0052 3.62355e-17` | `4.02456e-15 8.67362e-16 -1.5708` | `0 0 1` | `[-1.65806, 1.65806]` |
| `wrist_link` | `gripper_link` | `wrist_roll` | revolute | `5.55112e-17 -0.0611 0.0181` | `1.5708 0.0486795 3.14159` | `0 0 1` | `[-2.74385, 2.84121]` |
| `gripper_link` | `moving_jaw_so101_v1_link` | `gripper` | revolute | `0.0202 0.0188 -0.0234` | `1.5708 -5.24284e-08 -1.41553e-15` | `0 0 1` | `[-0.174533, 1.74533]` |

## Available Upstream Files

The upstream `Simulation/SO101` directory includes:

- `so101_new_calib.urdf`
- `so101_old_calib.urdf`
- `so101_new_calib.xml`
- `so101_old_calib.xml`
- `scene.xml`
- `joints_properties.xml`
- `assets/*.stl`

## Notes

- The "new calibration" model sets each joint's virtual zero at the middle of
  its joint range. The "old calibration" model sets virtual zero to the fully
  extended horizontal configuration.
- LeRobot represents the gripper as a linear command where `0` is fully closed
  and `100` is fully open. The upstream SO-101 simulation README says this
  mapping is not yet reflected in the current URDF and MuJoCo files.
- For teleoperation, recording, replay, and normal imitation-learning policy
  training, LeRobot only needs joint state/action and camera observations. This
  geometry is mainly needed for FK, IK, collision checking, simulation, or
  geometry-aware state representations.
