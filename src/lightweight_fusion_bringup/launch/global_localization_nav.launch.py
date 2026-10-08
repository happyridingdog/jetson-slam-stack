"""Run Fast-LIO as local odometry, global Scan Context localization and Nav2.

This is the map-use mode. It deliberately does not launch the pose-graph mapper,
so no scans are accumulated into a new map.
"""
import os
import atexit
import fcntl
import socket
import struct
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def navigation_transport_profile(share):
    """Exclude tunnel adapters without hard-coding DHCP addresses or disabling LAN."""
    override = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE")
    if override:
        return override
    addresses = {"127.0.0.1"}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _, name in socket.if_nameindex():
            try:
                # ARPHRD_ETHER and ARPHRD_LOOPBACK; exclude TUN/proxy and CAN.
                if int(Path('/sys/class/net', name, 'type').read_text()) not in (1, 772):
                    continue
                request = struct.pack('256s', name.encode()[:15])
                flags = struct.unpack_from('H', fcntl.ioctl(sock, 0x8913, request), 16)[0]
                if not flags & 1:
                    continue
                address = socket.inet_ntoa(fcntl.ioctl(sock, 0x8915, request)[20:24])
                addresses.add(address)
            except (OSError, ValueError):
                continue
    namespace = 'http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles'
    ET.register_namespace('', namespace)
    tag = lambda name: '{'+namespace+'}'+name
    tree = ET.parse(os.path.join(share, 'config', 'fastdds_navigation.xml'))
    for transport in tree.iter(tag('transport_descriptor')):
        if transport.findtext(tag('type')) == 'UDPv4':
            whitelist = ET.SubElement(transport, tag('interfaceWhiteList'))
            for address in sorted(addresses):
                ET.SubElement(whitelist, tag('address')).text = address
    with tempfile.NamedTemporaryFile(prefix='navigation_fastdds_', suffix='.xml', delete=False) as file:
        tree.write(file, encoding='utf-8', xml_declaration=True)
        profile = file.name
    atexit.register(lambda: Path(profile).unlink(missing_ok=True))
    return profile


def generate_launch_description():
    share = get_package_share_directory("lightweight_fusion_bringup")
    livox_share = get_package_share_directory("livox_ros_driver2")
    nav_launch = os.path.join(share, "launch", "nav2_pointcloud.launch.py")

    map_path = LaunchConfiguration("map_path")
    use_rviz = LaunchConfiguration("use_rviz")
    start_driver = LaunchConfiguration("start_driver")
    driver_config = os.path.join(livox_share, "config", "MID360_config.json")
    fastlio_config = os.path.join(share, "config", "fastlio_mid360_localization.yaml")
    localizer_config = os.path.join(share, "config", "global_localization.yaml")
    rviz_config = os.path.join(share, "rviz", "navigation.rviz")

    return LaunchDescription([
        # The default 512 KiB SHM segment / 512-entry port can overflow when
        # maps and 200 Hz IMU share DDS, delaying reliable IMU behind repairs.
        # Keep reliability; reserve bounded transport capacity for local data.
        SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", navigation_transport_profile(share)),
        # SSH/XRDP shells may not export DISPLAY.  The local desktop is :1;
        # preserve an explicitly supplied display when one is available.
        SetEnvironmentVariable("DISPLAY", os.environ.get("DISPLAY", ":1")),
        SetEnvironmentVariable(
            "XAUTHORITY", os.environ.get("XAUTHORITY", "/run/user/1000/gdm/Xauthority")),
        DeclareLaunchArgument(
            "map_path", default_value="/home/jetson/.jszr/map/pose_graph_map_latest.pcd"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("start_driver", default_value="true"),
        DeclareLaunchArgument("require_chassis_ready", default_value="false"),
        Node(
            package="livox_ros_driver2", executable="livox_ros_driver2_safe_node",
            name="livox_lidar_publisher_localization", output="screen",
            condition=IfCondition(start_driver),
            parameters=[{
                "xfer_format": 0, "multi_topic": 0, "data_src": 0,
                "publish_freq": 10.0, "output_data_type": 0, "frame_id": "livox_frame",
                "lvx_file_path": "/tmp/mid360_unused.lvx", "user_config_path": driver_config,
                "cmdline_input_bd_code": "livox0000000001",
            }],
        ),
        # Frontend is only the continuous local odometry source. No mapper node
        # is launched in this mode.
        Node(package="robot_slam", executable="mapping", name="laser_mapping",
             output="screen", parameters=[fastlio_config],
             # Yield idle OpenMP workers so IMU/DDS callbacks retain CPU time.
             additional_env={"OMP_WAIT_POLICY": "PASSIVE"},
             # The prediction bridge owns odom -> sensor on /tf. Keep raw
             # Fast-LIO transforms available without publishing a second owner.
             remappings=[('/tf', '/fastlio/raw_tf')]),
        Node(
            package="lightweight_fusion_bringup", executable="laser_odometry_tf_bridge.py",
            name="laser_odometry_tf_bridge", output="screen",
            parameters=[{
                "input_topic": "/laser_odometry", "parent_frame": "odom",
                "child_frame": "sensor", "use_input_stamp": True,
                # Match sensor -> base_link in nav2_pointcloud.launch.py.
                "odometry_child_frame": "base_link", "sensor_offset_x_m": 0.21,
                "imu_topic": "/livox/imu", "prediction_output_topic": "/nav_predicted_odom",
                # Keep only a short radar-IMU prediction window for control
                # smoothness.  The bridge clamps every gap to this cap.
                "max_prediction_s": 0.15, "max_yaw_prediction_s": 0.02,
                "prediction_rate_hz": 50.0,
                # Navigation-only pose stabilization; raw Fast-LIO output is
                # still available on /laser_odometry unchanged.
                "pose_filter_alpha": 0.50, "yaw_filter_alpha": 0.50,
                "pose_deadband_m": 0.002, "yaw_deadband_rad": 0.002,
                "imu_gyro_deadband_rad_s": 0.01,
                "imu_accel_deadband_mps2": 0.10,
                # Limit navigation heading extrapolation to 20 ms. Keep
                # measured yaw-rate feedback for turning and braking.
                "integrate_imu_yaw": True,
            }],
        ),
        Node(
            package="lightweight_fusion_bringup", executable="scan_context_global_localizer",
            name="scan_context_global_localizer", output="screen",
            parameters=[localizer_config, {"map_path": map_path}],
            respawn=True, respawn_delay=3.0,
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav_launch),
            launch_arguments={"require_chassis_ready": LaunchConfiguration("require_chassis_ready"), "map_path": map_path}.items(),
        ),
        # Reuse the existing navigation RViz layout and goal tool.
        # XRDP uses Mesa llvmpipe. Bound its raster workers so repaint bursts
        # cannot occupy every CPU while odometry is processing a scan.
        Node(package="rviz2", executable="rviz2", name="rviz2_global_localization",
             additional_env={"LP_NUM_THREADS": "2"},
             output="screen", arguments=["-d", rviz_config], condition=IfCondition(use_rviz)),
    ])
