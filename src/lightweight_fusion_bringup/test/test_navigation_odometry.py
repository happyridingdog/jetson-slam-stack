#!/usr/bin/env python3
"""Test the actual bridge publisher/predictor without ROS traffic or hardware."""
import collections
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry

spec = importlib.util.spec_from_file_location('bridge', Path(__file__).resolve().parents[1] / 'scripts/laser_odometry_tf_bridge.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Bridge = module.LaserOdometryTfBridge

class NavigationOdometry(unittest.TestCase):
    def bridge(self):
        obj = Bridge.__new__(Bridge)
        obj.parent_frame, obj.child_frame, obj.odometry_child_frame = 'odom', 'sensor', 'base_link'
        obj.sensor_offset_x, obj.sensor_offset_y = .21, 0.
        obj.last_published_odom_ns = 0
        obj.last_odom = Odometry()
        obj.last_odom.header.stamp = Time(sec=10)
        obj.last_odom.pose.pose.orientation.w = 1.
        obj.tf_broadcaster, obj.predicted_pub = Mock(), Mock()
        obj.max_prediction_s, obj.max_yaw_prediction_s = .15, .08
        obj.integrate_imu_yaw = True
        obj.imu_samples = collections.deque()
        obj.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=10,nanosec=50000000)))
        return obj

    def test_measurement_correction_does_not_rewind_control_odometry(self):
        b = self.bridge()
        b._publish_tf_and_odom((1.,0.,0.), (0.,0.,0.), Time(sec=10,nanosec=90000000))
        b._publish_tf_and_odom((.9,0.,0.), (0.,0.,0.), Time(sec=10,nanosec=20000000))
        self.assertEqual(b.predicted_pub.publish.call_count, 1)
        self.assertEqual(b.tf_broadcaster.sendTransform.call_count, 2)
        b._publish_tf_and_odom((1.1,0.,0.), (0.,0.,0.), Time(sec=10,nanosec=100000000))
        self.assertEqual(b.predicted_pub.publish.call_count, 2)

    def test_pure_base_rotation_has_no_false_translation(self):
        b = self.bridge()
        b._publish_tf_and_odom((0.,.21,math.pi/2), (0.,.105,.5), Time(sec=10))
        msg = b.predicted_pub.publish.call_args.args[0]
        tf = b.tf_broadcaster.sendTransform.call_args.args[0]
        self.assertEqual(msg.child_frame_id, 'base_link')
        self.assertAlmostEqual(msg.pose.pose.position.x, 0.)
        self.assertAlmostEqual(msg.pose.pose.position.y, 0.)
        self.assertAlmostEqual(msg.twist.twist.linear.y, 0.)
        self.assertEqual(msg.twist.twist.angular.z, .5)
        self.assertEqual(tf.child_frame_id, 'sensor')
        self.assertAlmostEqual(tf.transform.translation.y, .21)

    def test_prediction_retains_angular_velocity_without_imu(self):
        b = self.bridge()
        b.last_odom.twist.twist.angular.z = .4
        b.last_odom.twist.twist.linear.y = .084
        b.predict_cb()
        msg = b.predicted_pub.publish.call_args.args[0]
        self.assertAlmostEqual(msg.twist.twist.angular.z, .4)
        self.assertAlmostEqual(msg.twist.twist.linear.y, 0.)

    def test_fresh_imu_angular_velocity_is_published(self):
        b = self.bridge()
        b.last_odom.twist.twist.angular.z = .4
        b.imu_samples.append((10040000000,0.,0.,.6))
        b.predict_cb()
        self.assertAlmostEqual(b.predicted_pub.publish.call_args.args[0].twist.twist.angular.z, .6)

    def test_default_sensor_output_is_unchanged(self):
        b = self.bridge()
        b.sensor_offset_x = b.sensor_offset_y = 0.
        b.odometry_child_frame = 'sensor'
        b._publish_tf_and_odom((1.,2.,.3), (.4,.1,.2), Time(sec=10))
        msg = b.predicted_pub.publish.call_args.args[0]
        self.assertEqual(msg.child_frame_id, 'sensor')
        self.assertEqual(msg.pose.pose.position.x, 1.)
        self.assertEqual(msg.twist.twist.linear.y, .1)

    def test_prediction_stops_when_lio_is_stale(self):
        b = self.bridge()
        b.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=11)))
        b.predict_cb()
        b.predicted_pub.publish.assert_not_called()
        b.tf_broadcaster.sendTransform.assert_not_called()

    def test_navigation_yaw_prediction_caps_at_twenty_ms_without_clipping_rate(self):
        b = self.bridge()
        b.max_yaw_prediction_s = .02
        b.last_odom.pose.pose.position.x = .21
        b.last_odom.twist.twist.angular.z = .4
        b.last_odom.twist.twist.linear.y = .084
        b.imu_samples.append((10000000000, 0., 0., .4))
        for age_ns in [10000000, 20000000, 80000000, 140000000]:
            b.get_clock = lambda age=age_ns: SimpleNamespace(now=lambda: SimpleNamespace(
                to_msg=lambda: Time(sec=10, nanosec=age)))
            b.predict_cb()
            msg = b.predicted_pub.publish.call_args.args[0]
            self.assertAlmostEqual(b._yaw(msg.pose.pose.orientation), .4*min(age_ns*1e-9, .02))
            self.assertAlmostEqual(msg.twist.twist.angular.z, .4)
            self.assertAlmostEqual(msg.pose.pose.position.x, 0.)
            self.assertAlmostEqual(msg.pose.pose.position.y, 0.)

    def test_translation_predicts_without_an_imu_packet(self):
        b = self.bridge()
        b.last_odom.pose.pose.position.x = .21
        b.last_odom.twist.twist.linear.x = .4
        b.predict_cb()
        msg = b.predicted_pub.publish.call_args.args[0]
        self.assertAlmostEqual(msg.pose.pose.position.x, .02, places=6)

if __name__ == '__main__':
    unittest.main()
