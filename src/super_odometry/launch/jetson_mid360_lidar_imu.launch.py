import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    super_share = get_package_share_directory("super_odometry")
    livox_share = get_package_share_directory("livox_ros_driver2")

    super_config = os.path.join(
        super_share, "config", "jetson_mid360_lidar_imu.yaml"
    )
    calibration = os.path.join(
        super_share, "config", "livox", "livox_mid360_calibration.yaml"
    )
    livox_config = os.path.join(livox_share, "config", "MID360_config.json")

    return LaunchDescription(
        [
            Node(
                package="livox_ros_driver2",
                executable="livox_ros_driver2_safe_node",
                name="livox_lidar_publisher",
                output="screen",
                parameters=[
                    {
                        "xfer_format": 1,  # livox_ros_driver2/CustomMsg
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
            ),
            Node(
                package="super_odometry",
                executable="feature_extraction_node",
                output="screen",
                parameters=[
                    super_config,
                    {"calibration_file": calibration},
                ],
            ),
            Node(
                package="super_odometry",
                executable="laser_mapping_node",
                output="screen",
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
                output="screen",
                parameters=[
                    super_config,
                    {"calibration_file": calibration},
                ],
            ),
        ]
    )
