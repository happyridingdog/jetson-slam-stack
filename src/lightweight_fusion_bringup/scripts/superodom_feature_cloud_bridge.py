#!/usr/bin/env python3
"""Forward SuperOdom's undistorted LiDAR cloud to the pose-graph backend."""

import rclpy
from collections import deque
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from super_odometry_msgs.msg import LaserFeature


class SuperOdomFeatureCloudBridge(Node):
    def __init__(self):
        super().__init__("superodom_feature_cloud_bridge")
        # SuperOdom uses an empty ProjectName by default, therefore its
        # LaserFeature topic is /feature_info for the compact deployment.
        self.declare_parameter("input_topic", "/feature_info")
        self.declare_parameter("output_topic", "/body_points")
        # Keep the scan timestamp.  The pose-graph backend must look up the
        # transform at the scan time; using the latest transform introduces a
        # time offset whenever processing lags behind the sensor.
        self.declare_parameter("use_latest_tf", False)
        self.declare_parameter("publish_delay_s", 0.15)
        self.declare_parameter("max_pending_clouds", 8)
        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.use_latest_tf = self.get_parameter("use_latest_tf").value
        self.publish_delay_s = max(0.0, float(self.get_parameter("publish_delay_s").value))
        self.max_pending_clouds = max(1, int(self.get_parameter("max_pending_clouds").value))
        self.pending = deque()
        self.publisher = self.create_publisher(PointCloud2, output_topic, qos_profile_sensor_data)
        self.subscription = self.create_subscription(
            LaserFeature, input_topic, self.on_feature, 10
        )
        self.timer = self.create_timer(0.02, self.flush_pending)
        self.get_logger().info(f"bridging {input_topic}.cloud_nodistortion -> {output_topic}")

    def on_feature(self, msg):
        cloud = msg.cloud_nodistortion
        if cloud.width == 0 or cloud.height == 0 or not cloud.data:
            return
        if self.use_latest_tf:
            self.publisher.publish(cloud)
            return

        stamp_s = float(cloud.header.stamp.sec) + float(cloud.header.stamp.nanosec) * 1e-9
        self.pending.append((stamp_s, cloud))
        while len(self.pending) > self.max_pending_clouds:
            self.pending.popleft()

    def flush_pending(self):
        now_s = self.get_clock().now().nanoseconds * 1e-9
        while self.pending and now_s - self.pending[0][0] >= self.publish_delay_s:
            _, cloud = self.pending.popleft()
            # Keep the original stamp. The pose-graph node performs the TF
            # lookup and drops the frame if that timestamp is unavailable.
            self.publisher.publish(cloud)


def main():
    rclpy.init()
    node = SuperOdomFeatureCloudBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        # launch may already have shut down the shared context while stopping
        # the process; avoid turning normal Ctrl-C teardown into an error.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
