"""Compact MID360 LiDAR+IMU mapping: SuperOdom -> cloud bridge -> SC/GICP graph."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    super_share = get_package_share_directory("super_odometry")
    livox_share = get_package_share_directory("livox_ros_driver2")
    fusion_share = get_package_share_directory("lightweight_fusion_bringup")

    super_config = os.path.join(
        super_share, "config", "jetson_mid360_lidar_imu.yaml"
    )
    calibration = os.path.join(
        super_share, "config", "livox", "livox_mid360_calibration.yaml"
    )
    livox_config = os.path.join(livox_share, "config", "MID360_config.json")
    default_graph_config = os.path.join(fusion_share, "config", "mid360_pose_graph.yaml")
    graph_config = LaunchConfiguration("graph_config")
    rviz_config = os.path.join(fusion_share, "rviz", "superodom_pose_graph.rviz")
    use_rviz = LaunchConfiguration("use_rviz")

    driver = Node(
        package="livox_ros_driver2",
        executable="livox_ros_driver2_safe_node",
        name="livox_lidar_publisher",
        output="screen",
        parameters=[
            {
                # CustomMsg carries per-point time required by SuperOdom.
                "xfer_format": 1,
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

    super_nodes = [
        Node(
            package="super_odometry",
            executable="feature_extraction_node",
            name="superodom_feature_extraction",
            output="screen",
            remappings=[("/tf", "/superodom_tf"), ("/tf_static", "/superodom_tf_static")],
            parameters=[super_config, {"calibration_file": calibration}],
        ),
        Node(
            package="super_odometry",
            executable="laser_mapping_node",
            name="superodom_laser_mapping",
            output="screen",
            # Keep SuperOdom's intermediate prediction TF out of the global
            # tree.  The accepted /laser_odometry stream is bridged below.
            remappings=[("/tf", "/superodom_tf"), ("/tf_static", "/superodom_tf_static")],
            parameters=[
                super_config,
                {
                    "calibration_file": calibration,
                    "map_dir": os.path.join(os.path.expanduser("~"), ".jszr", "map"),
                },
            ],
        ),
        Node(
            package="super_odometry",
            executable="imu_preintegration_node",
            name="superodom_imu_preintegration",
            output="screen",
            remappings=[("/tf", "/superodom_tf"), ("/tf_static", "/superodom_tf_static")],
            parameters=[super_config, {"calibration_file": calibration}],
        ),
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "graph_config", default_value=default_graph_config,
                description="Pose graph configuration (explicit same-frame recovery profile when needed)",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Start RViz2 when an X11 display is available",
            ),
            driver,
            *super_nodes,
            Node(
                package="lightweight_fusion_bringup",
                executable="superodom_feature_cloud_bridge.py",
                name="superodom_feature_cloud_bridge",
                output="screen",
                parameters=[
                    {
                        "input_topic": "/feature_info",
                        "output_topic": "/body_points",
                        "use_latest_tf": False,
                        # Wait for the laser odometry TF at the cloud's own
                        # timestamp.  Publishing immediately forced the
                        # backend to use stale transforms and created ghosts.
                        "publish_delay_s": 0.80,
                        "max_pending_clouds": 24,
                    }
                ],
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
                        # SuperOdom's optimized pose can arrive 0.4--0.7 s
                        # behind acquisition. Mapping needs historical TF at
                        # that exact stamp; the navigation 0.35 s gate must
                        # not discard it. This does not extend prediction.
                        "max_odometry_age_s": 2.0,
                        "prediction_rate_hz": 50.0,
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2_superodom_mapping",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(use_rviz),
            ),
        ]
    )
