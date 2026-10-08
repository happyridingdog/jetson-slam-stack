import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("robot_slam")
    config_file = os.path.join(package_share, "config", "config_highspeed.yaml")
    rviz_file = os.path.join(package_share, "rviz", "slam_highspeed_minimal.rviz")
    livox_share = get_package_share_directory("livox_ros_driver2")
    livox_launch = os.path.join(livox_share, "launch_ROS2", "msg_MID360_launch.py")

    return LaunchDescription([
        # Keep OpenMP/PCL from occupying every CPU core on the deployment
        # machine.  Two cores remain available for control and other nodes.
        SetEnvironmentVariable("OMP_NUM_THREADS", "4"),
        SetEnvironmentVariable("OMP_DYNAMIC", "TRUE"),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            description="Start RViz2 for the LIO outputs",
        ),
        DeclareLaunchArgument(
            "start_driver",
            default_value="true",
            description="Start the Livox MID360 driver",
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(livox_launch),
            condition=IfCondition(LaunchConfiguration("start_driver")),
        ),
        Node(
            package="robot_slam",
            executable="mapping",
            name="laser_mapping",
            output="screen",
            parameters=[config_file],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="robot_slam_rviz",
            output="screen",
            arguments=["-d", rviz_file],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        ),
    ])
