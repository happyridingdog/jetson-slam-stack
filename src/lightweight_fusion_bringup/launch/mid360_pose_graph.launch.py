import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("lightweight_fusion_bringup")
    config = os.path.join(share, "config", "mid360_pose_graph.yaml")
    return LaunchDescription([
        Node(
            package="lightweight_fusion_bringup",
            executable="sc_gicp_pose_graph_mapper",
            name="sc_gicp_pose_graph_mapper",
            output="screen",
            parameters=[config],
        )
    ])
