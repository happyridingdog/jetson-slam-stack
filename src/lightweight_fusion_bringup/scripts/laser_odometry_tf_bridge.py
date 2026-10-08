#!/usr/bin/env python3
"""Publish navigation TF with a strictly bounded radar-IMU prediction.

The SLAM odometry stream remains the anchor. Between accepted odometry
messages, MID360 IMU samples are integrated for at most ``max_prediction_s``.
A gap larger than that is deliberately clamped to the same maximum.
"""
import collections
import copy
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


class LaserOdometryTfBridge(Node):
    def __init__(self):
        super().__init__('laser_odometry_tf_bridge')
        self.declare_parameter('input_topic', '/laser_odometry')
        self.declare_parameter('parent_frame', 'map')
        self.declare_parameter('child_frame', 'sensor')
        # TF remains at the lidar origin; navigation odometry may be shifted
        # to the base origin. Frames have parallel axes in the installed robot.
        self.declare_parameter('odometry_child_frame', '')
        self.declare_parameter('sensor_offset_x_m', 0.0)
        self.declare_parameter('sensor_offset_y_m', 0.0)
        self.declare_parameter('use_input_stamp', True)
        self.declare_parameter('max_stamp_regression_s', 0.02)
        self.declare_parameter('imu_topic', '/livox/imu')
        self.declare_parameter('prediction_output_topic', '/nav_predicted_odom')
        self.declare_parameter('max_prediction_s', 0.2)
        self.declare_parameter('max_odometry_age_s', 0.35)
        self.declare_parameter('max_yaw_prediction_s', 0.08)
        self.declare_parameter('prediction_rate_hz', 50.0)
        # These filters apply only to the navigation TF/output.  The raw
        # /laser_odometry frontend stream remains untouched.
        self.declare_parameter('pose_filter_alpha', 0.50)
        self.declare_parameter('yaw_filter_alpha', 0.50)
        self.declare_parameter('pose_deadband_m', 0.002)
        self.declare_parameter('yaw_deadband_rad', 0.002)
        self.declare_parameter('imu_gyro_deadband_rad_s', 0.01)
        self.declare_parameter('imu_accel_deadband_mps2', 0.10)
        self.declare_parameter('integrate_imu_yaw', False)
        # Fast-LIO publishes a valid pose but leaves Odometry.twist at zero.
        # Navigation needs a measured current speed for its acceleration
        # limits, so estimate it from successive filtered poses here.
        self.declare_parameter('estimate_twist_from_pose', True)
        self.declare_parameter('twist_filter_alpha', 0.45)
        self.declare_parameter('twist_deadband_mps', 0.01)
        self.declare_parameter('twist_deadband_rad_s', 0.01)
        self.declare_parameter('max_estimated_linear_speed_mps', 1.5)
        self.declare_parameter('max_estimated_angular_speed_rad_s', 2.0)

        self.parent_frame = str(self.get_parameter('parent_frame').value)
        self.child_frame = str(self.get_parameter('child_frame').value)
        self.odometry_child_frame = str(self.get_parameter('odometry_child_frame').value) or self.child_frame
        self.sensor_offset_x = float(self.get_parameter('sensor_offset_x_m').value)
        self.sensor_offset_y = float(self.get_parameter('sensor_offset_y_m').value)
        if not all(math.isfinite(v) for v in (self.sensor_offset_x, self.sensor_offset_y)):
            raise ValueError('Sensor offsets must be finite')
        self.use_input_stamp = bool(self.get_parameter('use_input_stamp').value)
        self.max_regression_ns = int(max(0.0, float(
            self.get_parameter('max_stamp_regression_s').value)) * 1e9)
        self.max_prediction_s = max(0.0, float(self.get_parameter('max_prediction_s').value))
        self.max_odometry_age_s = max(.05, float(self.get_parameter('max_odometry_age_s').value))
        self.max_yaw_prediction_s = min(self.max_prediction_s, max(0.0, float(
            self.get_parameter('max_yaw_prediction_s').value)))
        self.prediction_rate_hz = max(1.0, float(self.get_parameter('prediction_rate_hz').value))
        self.pose_filter_alpha = min(1.0, max(0.05, float(
            self.get_parameter('pose_filter_alpha').value)))
        self.yaw_filter_alpha = min(1.0, max(0.05, float(
            self.get_parameter('yaw_filter_alpha').value)))
        self.pose_deadband_m = max(0.0, float(self.get_parameter('pose_deadband_m').value))
        self.yaw_deadband_rad = max(0.0, float(self.get_parameter('yaw_deadband_rad').value))
        self.imu_gyro_deadband = max(0.0, float(
            self.get_parameter('imu_gyro_deadband_rad_s').value))
        self.imu_accel_deadband = max(0.0, float(
            self.get_parameter('imu_accel_deadband_mps2').value))
        self.integrate_imu_yaw = bool(self.get_parameter('integrate_imu_yaw').value)
        self.estimate_twist_from_pose = bool(
            self.get_parameter('estimate_twist_from_pose').value)
        self.twist_filter_alpha = min(1.0, max(0.05, float(
            self.get_parameter('twist_filter_alpha').value)))
        self.twist_deadband_mps = max(0.0, float(
            self.get_parameter('twist_deadband_mps').value))
        self.twist_deadband_rad_s = max(0.0, float(
            self.get_parameter('twist_deadband_rad_s').value))
        self.max_estimated_linear_speed = max(0.05, float(
            self.get_parameter('max_estimated_linear_speed_mps').value))
        self.max_estimated_angular_speed = max(0.05, float(
            self.get_parameter('max_estimated_angular_speed_rad_s').value))
        self.last_published_odom_ns = 0
        self.last_stamp_ns = 0
        self.last_odom = None
        self.filtered_pose = None
        self.last_velocity_pose = None
        self.last_velocity_stamp_ns = 0
        self.nav_velocity = (0.0, 0.0, 0.0)
        self.imu_samples = collections.deque(maxlen=512)
        self.tf_broadcaster = TransformBroadcaster(self)

        topic = str(self.get_parameter('input_topic').value)
        self.subscription = self.create_subscription(Odometry, topic, self.odom_cb, 20)
        imu_topic = str(self.get_parameter('imu_topic').value)
        imu_qos = QoSProfile(depth=256)
        imu_qos.reliability = ReliabilityPolicy.RELIABLE
        imu_qos.history = HistoryPolicy.KEEP_LAST
        self.imu_subscription = self.create_subscription(Imu, imu_topic, self.imu_cb, imu_qos)
        output_topic = str(self.get_parameter('prediction_output_topic').value)
        self.predicted_pub = self.create_publisher(Odometry, output_topic, 10)
        self.prediction_timer = self.create_timer(
            1.0 / self.prediction_rate_hz, self.predict_cb)
        self.get_logger().info(
            'publishing %s -> %s from %s; IMU=%s; prediction cap=%.3fs' %
            (self.parent_frame, self.child_frame, topic, imu_topic, self.max_prediction_s))

    @staticmethod
    def _stamp_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _yaw(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def _quat(yaw):
        return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)

    @staticmethod
    def _wrap_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _filter_pose(self, msg):
        """Suppress sub-millimetre scan noise without changing SLAM output."""
        raw_yaw = self._yaw(msg.pose.pose.orientation)
        raw = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), raw_yaw)
        if self.filtered_pose is None:
            self.filtered_pose = raw
            return raw
        fx, fy, fyaw = self.filtered_pose
        dx = raw[0] - fx
        dy = raw[1] - fy
        dyaw = self._wrap_angle(raw[2] - fyaw)
        if math.hypot(dx, dy) <= self.pose_deadband_m:
            x, y = fx, fy
        else:
            x = fx + self.pose_filter_alpha * dx
            y = fy + self.pose_filter_alpha * dy
        yaw = fyaw if abs(dyaw) <= self.yaw_deadband_rad else \
            self._wrap_angle(fyaw + self.yaw_filter_alpha * dyaw)
        self.filtered_pose = (x, y, yaw)
        return self.filtered_pose

    def imu_cb(self, msg: Imu):
        stamp_ns = self._stamp_ns(msg.header.stamp)
        if stamp_ns <= 0 or (self.imu_samples and stamp_ns <= self.imu_samples[-1][0]):
            return
        if not all(math.isfinite(v) for v in
                   (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.angular_velocity.z)):
            return
        def deadband(value, threshold):
            return 0.0 if abs(value) < threshold else value
        self.imu_samples.append((stamp_ns,
                                 deadband(float(msg.linear_acceleration.x), self.imu_accel_deadband),
                                 deadband(float(msg.linear_acceleration.y), self.imu_accel_deadband),
                                 deadband(float(msg.angular_velocity.z), self.imu_gyro_deadband)))

    def _publish_tf_and_odom(self, pose, velocity, stamp):
        if self.last_odom is None:
            return
        x, y, yaw = pose
        qx, qy, qz, qw = self._quat(yaw)
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.child_frame
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = self.last_odom.pose.pose.position.z
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)

        # TF retains measurement-time corrections for point-cloud lookup.
        # Control odometry must never jump backwards after a newer prediction.
        stamp_ns = self._stamp_ns(stamp)
        if stamp_ns <= self.last_published_odom_ns:
            return
        self.last_published_odom_ns = stamp_ns

        predicted = Odometry()
        predicted.header = transform.header
        predicted.child_frame_id = self.odometry_child_frame
        c, s = math.cos(yaw), math.sin(yaw)
        predicted.pose.pose.position.x = x - c*self.sensor_offset_x + s*self.sensor_offset_y
        predicted.pose.pose.position.y = y - s*self.sensor_offset_x - c*self.sensor_offset_y
        predicted.pose.pose.position.z = transform.transform.translation.z
        predicted.pose.pose.orientation = transform.transform.rotation
        # v_sensor = v_base + omega cross r_base_to_sensor.
        predicted.twist.twist.linear.x = velocity[0] + velocity[2]*self.sensor_offset_y
        predicted.twist.twist.linear.y = velocity[1] - velocity[2]*self.sensor_offset_x
        predicted.twist.twist.angular.z = velocity[2]
        self.predicted_pub.publish(predicted)

    def _estimate_velocity(self, pose, stamp_ns, fallback):
        """Estimate body velocity without changing the raw frontend odom.

        The pose stream is the authoritative Fast-LIO result.  Estimating the
        twist here gives Nav2 a meaningful current-speed signal, while the
        filter/deadbands prevent scan jitter from becoming steering commands.
        """
        if not self.estimate_twist_from_pose or stamp_ns <= 0:
            return fallback
        if self.last_velocity_pose is None or self.last_velocity_stamp_ns <= 0:
            self.last_velocity_pose = pose
            self.last_velocity_stamp_ns = stamp_ns
            return (0.0, 0.0, 0.0)
        previous_pose = self.last_velocity_pose
        previous_stamp_ns = self.last_velocity_stamp_ns
        dt = (stamp_ns - self.last_velocity_stamp_ns) * 1e-9
        # Never move the differentiation anchor backwards on duplicate data.
        if dt < 0.005:
            return self.nav_velocity
        self.last_velocity_pose = pose
        self.last_velocity_stamp_ns = stamp_ns
        if dt > 2.0:
            self.nav_velocity = (0.0, 0.0, 0.0)
            return self.nav_velocity
        px, py, pyaw = previous_pose
        x, y, yaw = pose
        vx_map = (x - px) / dt
        vy_map = (y - py) / dt
        wz = self._wrap_angle(yaw - pyaw) / dt
        # Reject scan-registration spikes before they reach Nav2.
        speed = math.hypot(vx_map, vy_map)
        if speed > self.max_estimated_linear_speed:
            scale = self.max_estimated_linear_speed / speed
            vx_map *= scale
            vy_map *= scale
        wz = max(-self.max_estimated_angular_speed,
                 min(self.max_estimated_angular_speed, wz))
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        measured = (cy * vx_map + sy * vy_map,
                    -sy * vx_map + cy * vy_map, wz)
        if abs(measured[0]) < self.twist_deadband_mps:
            measured = (0.0, measured[1], measured[2])
        if abs(measured[1]) < self.twist_deadband_mps:
            measured = (measured[0], 0.0, measured[2])
        if abs(measured[2]) < self.twist_deadband_rad_s:
            measured = (measured[0], measured[1], 0.0)
        a = self.twist_filter_alpha
        self.nav_velocity = tuple(
            old + a * (new - old) for old, new in zip(self.nav_velocity, measured))
        return self.nav_velocity

    def predict_cb(self):
        if self.last_odom is None:
            return
        now = self.get_clock().now().to_msg()
        now_ns = self._stamp_ns(now)
        anchor_ns = self._stamp_ns(self.last_odom.header.stamp)
        if anchor_ns <= 0:
            return
        age = (now_ns-anchor_ns)*1e-9
        # A bounded pose with an unbounded fresh timestamp masks a dead LIO.
        # Stop publishing beyond the prediction horizon so consumers can stop.
        if age < 0.0 or age > self.max_prediction_s:
            return
        anchor = self.last_odom.pose.pose
        yaw = self._yaw(anchor.orientation)
        anchor_w = float(self.last_odom.twist.twist.angular.z)
        vx = float(self.last_odom.twist.twist.linear.x) + anchor_w*self.sensor_offset_y
        vy = float(self.last_odom.twist.twist.linear.y) - anchor_w*self.sensor_offset_x
        x = anchor.position.x - math.cos(yaw)*self.sensor_offset_x + math.sin(yaw)*self.sensor_offset_y
        y = anchor.position.y - math.sin(yaw)*self.sensor_offset_x - math.cos(yaw)*self.sensor_offset_y
        # Short translation prediction uses LIO body velocity. Raw accelerometer
        # gravity/bias is not a reliable ground-velocity correction. IMU supplies
        # fresh yaw rate, with the existing separate angular prediction bound.
        angular_velocity = anchor_w
        steps = max(1, math.ceil(age/.01))
        step = age/steps
        for i in range(steps):
            offset = i*step
            t_ns = anchor_ns+int((offset+step)*1e9)
            angular_velocity = anchor_w
            if self.integrate_imu_yaw:
                for stamp_ns, _, _, wz in reversed(self.imu_samples):
                    if stamp_ns <= t_ns:
                        if t_ns-stamp_ns <= int(self.max_prediction_s*1e9):
                            angular_velocity = wz
                        break
            yaw_dt = min(step, max(0.0, self.max_yaw_prediction_s-offset))
            mid_yaw = yaw+angular_velocity*yaw_dt*.5
            x += (math.cos(mid_yaw)*vx-math.sin(mid_yaw)*vy)*step
            y += (math.sin(mid_yaw)*vx+math.cos(mid_yaw)*vy)*step
            yaw += angular_velocity*yaw_dt
        sensor_x = x+math.cos(yaw)*self.sensor_offset_x-math.sin(yaw)*self.sensor_offset_y
        sensor_y = y+math.sin(yaw)*self.sensor_offset_x+math.cos(yaw)*self.sensor_offset_y
        self._publish_tf_and_odom((sensor_x, sensor_y, yaw),
            (vx-angular_velocity*self.sensor_offset_y,
             vy+angular_velocity*self.sensor_offset_x, angular_velocity), now)

    def odom_cb(self, msg: Odometry):
        stamp_ns = self._stamp_ns(msg.header.stamp)
        now_ns = self.get_clock().now().nanoseconds
        age = (now_ns-stamp_ns)*1e-9
        if stamp_ns <= 0 or stamp_ns <= self.last_stamp_ns or age < -0.05 or age > self.max_odometry_age_s:
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        if not all(math.isfinite(v) for v in (p.x,p.y,p.z,q.x,q.y,q.z,q.w)) or \
                q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w < 1e-9:
            return
        self.last_stamp_ns = stamp_ns
        filtered_pose = self._filter_pose(msg)
        filtered = copy.deepcopy(msg)
        filtered.pose.pose.position.x = filtered_pose[0]
        filtered.pose.pose.position.y = filtered_pose[1]
        filtered.pose.pose.orientation.x, filtered.pose.pose.orientation.y, \
            filtered.pose.pose.orientation.z, filtered.pose.pose.orientation.w = self._quat(filtered_pose[2])
        fallback_velocity = (filtered.twist.twist.linear.x,
                             filtered.twist.twist.linear.y,
                             filtered.twist.twist.angular.z)
        estimated_velocity = self._estimate_velocity(filtered_pose, stamp_ns,
                                                     fallback_velocity)
        filtered.twist.twist.linear.x = estimated_velocity[0]
        filtered.twist.twist.linear.y = estimated_velocity[1]
        filtered.twist.twist.angular.z = estimated_velocity[2]
        self.last_odom = filtered
        stamp = msg.header.stamp if self.use_input_stamp else self.get_clock().now().to_msg()
        self._publish_tf_and_odom(
            filtered_pose,
            estimated_velocity, stamp)
        self.predict_cb()


def main(args=None):
    rclpy.init(args=args)
    node = LaserOdometryTfBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
