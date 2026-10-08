"""Fast-LIO navigation frontend with the existing SC/GICP backend.

SuperOdom's launch remains available for mapping.  This profile uses the
Livox PointCloud2 transfer format and publishes the same frontend contract:
/laser_odometry, /body_points and map -> base_link.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    fusion_share = get_package_share_directory("lightweight_fusion_bringup")
    livox_share = get_package_share_directory("livox_ros_driver2")
    livox_config = os.path.join(livox_share, "config", "MID360_config.json")
    fastlio_config = os.path.join(fusion_share, "config", "fastlio_mid360_nav.yaml")
    graph_config = os.path.join(fusion_share, "config", "mid360_pose_graph.yaml")
    rviz_config = os.path.join(fusion_share, "rviz", "superodom_pose_graph.rviz")

    use_rviz = LaunchConfiguration("use_rviz")
    start_driver = LaunchConfiguration("start_driver")

    driver = Node(
        package="livox_ros_driver2",
        executable="livox_ros_driver2_safe_node",
        name="livox_lidar_publisher_fastlio",
        output="screen",
        sigterm_timeout="30",
        condition=IfCondition(start_driver),
        parameters=[
            {
                # Fast-LIO consumes the PointCloud2 format with per-point time.
                "xfer_format": 0,
                "multi_topic": 0,
                "data_src": 0,
                "publish_freq": 10.0,
                "output_data_type": 0,
                "frame_id": "livox_frame",
                "lvx_file_path": "/tmp/mid360_unused.lvx",
                "user_config_path": livox_config,
                "cmdline_input_bd_code": "livox0000000001",
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_rviz", default_value="false", description="Start RViz2"
            ),
            DeclareLaunchArgument(
                "start_driver", default_value="true", description="Start MID360 driver"
            ),
            driver,
            Node(
                package="robot_slam",
                executable="mapping",
                # The YAML root key is laser_mapping; keep the node name in
                # sync so calibration arrays are actually loaded.
                name="laser_mapping",
                output="screen",
                parameters=[fastlio_config],
                remappings=[('/tf', '/fastlio/raw_tf')],
            ),
            Node(
                package="lightweight_fusion_bringup",
                executable="sc_gicp_pose_graph_mapper",
                name="sc_gicp_pose_graph_mapper",
                output="screen",
                parameters=[graph_config],
                respawn=True,
                respawn_delay=3.0,
            ),
            Node(
                package="lightweight_fusion_bringup",
                executable="laser_odometry_tf_bridge.py",
                name="laser_odometry_tf_bridge",
                output="screen",
                parameters=[
                    {
                        "input_topic": "/laser_odometry",
                        "parent_frame": "map",
                        "child_frame": "sensor",
                        "use_input_stamp": True,
                        "imu_topic": "/livox/imu",
                        "prediction_output_topic": "/nav_predicted_odom",
                        "max_prediction_s": 0.2,
                        "prediction_rate_hz": 50.0,
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2_fastlio_mapping",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(use_rviz),
            ),
        ]
    )
