import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_parameter_config = os.path.join(get_package_share_directory(
        'robot_slam'), 'config', 'config.yaml')
    default_rviz_config_path = os.path.join(
        get_package_share_directory('robot_slam'), 'rviz', 'mapping.rviz')
    config_file = LaunchConfiguration('config_file')
    print(default_parameter_config)
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_parameter_config,
            description='Path to robot_slam parameter file'
        ),
        Node(
            package='robot_slam',
            executable='mapping',
            parameters=[
                config_file
            ]
        ),
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     arguments=['-d', default_rviz_config_path]
        # )
    ])
