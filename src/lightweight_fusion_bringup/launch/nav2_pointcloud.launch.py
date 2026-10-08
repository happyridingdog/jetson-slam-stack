import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    nav2_share = get_package_share_directory('nav2_bringup')
    fusion_share = get_package_share_directory('lightweight_fusion_bringup')
    params = RewrittenYaml(
        source_file=os.path.join(fusion_share, 'config', 'nav2_pointcloud.yaml'),
        root_key='',
        param_rewrites={
            'default_nav_to_pose_bt_xml': os.path.join(fusion_share, 'behavior_trees', 'navigate_to_pose_no_motion_recovery.xml'),
            'default_nav_through_poses_bt_xml': os.path.join(fusion_share, 'behavior_trees', 'navigate_through_poses_no_motion_recovery.xml'),
        }, convert_types=True)
    return LaunchDescription([
        DeclareLaunchArgument('map_path', default_value='', description='Source PCD for manual map edit records'),
        DeclareLaunchArgument('require_chassis_ready', default_value='false',
                              description='Require an external /chassis/status readiness report'),
        DeclareLaunchArgument('require_saved_map', default_value='true',
                              description='Require saved map input before accepting UI navigation'),
        DeclareLaunchArgument('height_filter_enabled', default_value='true',
                              description='Use one map-frame ground model for both costmaps'),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_mid360',
            output='screen',
            # The SLAM bridge publishes map -> sensor.  Make the vehicle base
            # a child of the lidar frame so the tree remains single-parent:
            # sensor -> base_link is the inverse of the 210 mm forward offset.
            arguments=['-0.21', '0.0', '0.0', '0.0', '0.0', '0.0', 'sensor', 'base_link'],
        ),
        Node(
            package='lightweight_fusion_bringup',
            executable='nav_cloud_retimestamp.py',
            name='nav_cloud_retimestamp',
            output='screen',
            parameters=[os.path.join(fusion_share, 'config', 'nav_cloud_height.yaml'), {
                'input_topic': '/body_points',
                'output_topic': '/nav_body_points',
                # Full live scans also span DDS fragments. Keep latest-only
                # buffering and real stamps, but allow missing-fragment repair.
                'reliable_cloud': True,
                'max_cloud_age_s': 0.50,
                'target_frame': '',
                'exclude_robot_footprint': True,
                'height_filter_enabled': ParameterValue(LaunchConfiguration('height_filter_enabled'), value_type=bool),
            }],
        ),
        Node(
            package='lightweight_fusion_bringup',
            executable='nav_cloud_retimestamp.py',
            name='nav_map_cloud_retimestamp',
            output='screen',
            # Use the system NumPy paired with system SciPy for ground fitting.
            # User-site NumPy 2 is ABI-incompatible with this SciPy installation.
            additional_env={'PYTHONNOUSERSITE': '1'},
            parameters=[os.path.join(fusion_share, 'config', 'nav_cloud_height.yaml'), {
                'input_topic': '/map_points_3d',
                'output_topic': '/nav_map_points_unedited',
                'target_frame': 'map',
                'exclude_robot_footprint': False,
                'height_filter_enabled': ParameterValue(LaunchConfiguration('height_filter_enabled'), value_type=bool),
                'estimate_ground_plane': True,
                # Saved maps span many DDS fragments; request retransmission.
                'reliable_cloud': True,
                'max_cloud_age_s': 5.0,
                # The accumulated cloud is static between mapper updates;
                # thin it for the costmap without changing the SLAM map.
                'point_stride': 1,
            }],
        ),
        Node(package='rviz_cloud_editor', executable='cloud_editor_node.py',
             name='map_cloud_editor', output='screen',
             additional_env={'OPENBLAS_NUM_THREADS': '1'},
             parameters=[{'map_path': LaunchConfiguration('map_path')}]),
        GroupAction([
            # Keep BOTH remaps inside the group. Humble SetRemap appends to
            # a shared list; creating that list in the parent leaks later
            # group additions into RViz and bypasses the goal supervisor.
            SetRemap(src='controller_smoothed_feedback', dst='cmd_vel'),
            # A single RViz goal consumer queues requests until Nav2 is ready.
            SetRemap(src='goal_pose', dst='navigation/internal_goal_pose'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(nav2_share, 'launch', 'navigation_launch.py')),
            launch_arguments={'params_file': params, 'use_sim_time': 'false', 'autostart': 'true', 'use_composition': 'False', 'use_respawn': 'False', 'log_level': 'info'}.items()),
        ]),
        Node(package='lightweight_fusion_bringup', executable='navigation_supervisor.py',
             name='navigation_supervisor', output='screen',
             remappings=[('/navigation/rviz_goal_input','/goal_pose')],
             parameters=[{'require_chassis_ready': ParameterValue(
                 LaunchConfiguration('require_chassis_ready'), value_type=bool),
                 'require_saved_map': ParameterValue(LaunchConfiguration('require_saved_map'), value_type=bool)}]),
    ])
