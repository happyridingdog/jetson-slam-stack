# Jetson SLAM Stack

Source snapshot of the LiDAR-inertial mapping, odometry, pose-graph mapping,
global localization, point-cloud navigation, and modified GPU registration code from the Jetson robot
archive dated 2026-10-06. This is a ROS 2 source workspace, not a system image.

## Packages

| Package | Purpose |
| --- | --- |
| `robot_slam` | MID360 LiDAR-inertial mapping and map export |
| `robots_dog_msgs` | `MapState` service used by `robot_slam` |
| `super_odometry` | LiDAR-inertial odometry, including custom CUDA kNN and residual evaluation |
| `super_odometry_msgs` | Interfaces required by `super_odometry` |
| `fast_gicp` | Registration implementation with the retained CUDA modifications |
| `lightweight_fusion_bringup` | Scan Context/GICP pose graph, global localization, Nav2 plugins, navigation supervisor, and launch files |
| `rviz_cloud_editor` | Map-cloud editing node and RViz plugin used by navigation |

The original source files, GPU paths, and package names are retained. The
release omits chassis and lift control, historical platforms, vendor driver source,
unneeded vendored third-party code, maps, models, logs, and compiled artifacts.
SuperOdom's local-map code still uses its bundled `nanoflann.h`; it remains
because removing it would break the retained odometry implementation.

## External dependencies

Install ROS 2 Humble, Nav2, the Livox ROS 2 driver (`livox_ros_driver2`), PCL,
Eigen3, OpenCV, Sophus, Ceres, GTSAM, TBB, and the ROS packages declared in
each `package.xml`. GPU builds also require a CUDA toolkit with NVCC and a
compatible NVIDIA GPU. These dependencies are not copied into this repository.

The original `super_odometry/CMakeLists.txt` contains Jetson-specific CUDA and
third-party library paths. Adjust those paths for another machine before
building. Its default CUDA architecture is `sm_87`. Sensor IP addresses and
calibration values belong to the target device and should be checked before
launching. Map output paths in the mapping configs point to the original
Jetson user's `~/.jszr/map` directory.

In a ROS 2 Humble environment with the external dependencies installed:

```bash
colcon build --packages-up-to lightweight_fusion_bringup
source install/setup.bash
ros2 launch lightweight_fusion_bringup superodom_pose_graph_mapping.launch.py
```

The alternative Fast-LIO frontend is available through
`fastlio_pose_graph_mapping.launch.py`. `mid360_pose_graph.launch.py` starts
only the graph mapper and expects its input topics to be supplied separately.

For navigation against a saved map:

```bash
ros2 launch lightweight_fusion_bringup global_localization_nav.launch.py \
  map_path:=/path/to/pose_graph_map_latest.pcd
```

The navigation launch publishes bounded `geometry_msgs/Twist` commands on
`/cmd_vel`; connect a platform-specific motor interface separately. No chassis
or lift driver is included. Chassis readiness monitoring is disabled by default;
set `require_chassis_ready:=true` only when an external driver publishes the
expected `/chassis/status`. Patrol requests that require lift control are
rejected. The launch expects the external Nav2 and Livox packages and uses
the original MID360 sensor-to-base offset; check that offset and the navigation
parameters for the target robot before commanding motion.

## Provenance and licensing

This repository is a selected source snapshot, not a tested clean-room
reimplementation. It retains the original `super_odometry` GPL-3.0-only and
`fast_gicp` BSD license files. Other packages retain their source metadata;
no repository-wide license is asserted here. Copyright notices in embedded
`robot_slam` filter headers are preserved.
