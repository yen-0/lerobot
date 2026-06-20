# SO-101 Robot Model Configuration

These files are copied from the official upstream SO-101 simulation directory:

https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101

Files:

- `so101_new_calib.urdf` - URDF with virtual joint zero at the middle of each joint range.
- `so101_old_calib.urdf` - URDF with virtual joint zero in the fully extended horizontal pose.
- `so101_new_calib.xml` - MuJoCo model for the new calibration convention.
- `so101_old_calib.xml` - MuJoCo model for the old calibration convention.
- `scene.xml` - MuJoCo scene wrapper.
- `joints_properties.xml` - Joint and actuator properties.

The URDF and MuJoCo files reference mesh assets under `assets/*.stl` in the
upstream repository. Those binary mesh files are not vendored here yet. The
configuration files are still useful for forward kinematics, inverse kinematics,
joint limits, and geometry-aware state representations.
