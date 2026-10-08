#!/usr/bin/env python3
"""Queue the latest RViz goal until navigation is ready and expose its state.

This node never arms the chassis or emits velocity. Nav2 remains the executor;
brief sensor gaps retain the paused goal; sustained loss cancels it without replay.
"""
import copy
import json
import math
import time
import uuid

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from action_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import NavigateToPose
from lifecycle_msgs.srv import GetState
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from rcl_interfaces.msg import Log
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class NavigationSupervisor(Node):
    def __init__(self):
        super().__init__('navigation_supervisor')
        self.declare_parameter('require_chassis_ready', True)
        self.declare_parameter('require_saved_map', True)
        self.declare_parameter('goal_wait_timeout_s', 60.0)
        self.require_chassis = bool(self.get_parameter('require_chassis_ready').value)
        self.require_saved_map = bool(self.get_parameter('require_saved_map').value)
        self.have_saved_map = False
        self.wait_timeout = float(self.get_parameter('goal_wait_timeout_s').value)
        self.sensor_gap_started = None
        # Match the controller's input_pause_timeout. Freshness limits stay
        # strict; only the zero-velocity wait before canceling is extended.
        self.sensor_gap_grace = 3.0
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.lifecycle_clients = {name:self.create_client(GetState, '/'+name+'/get_state')
            for name in ('bt_navigator','velocity_smoother')}
        self.lifecycle_states = {}
        self.lifecycle_pending = {}
        self.lifecycle_poll_time = 0.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/navigation/status', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/navigation/status_markers', latched)
        self.odom_stamp = self.cloud_stamp = 0
        self.odom = None
        self.command = Twist()
        self.command_time = 0.
        self.chassis = None
        self.chassis_time = 0.
        self.goal = self.pending = self.goal_handle = None
        self.serial = 0
        self.sending = False
        self.action_inflight = set()
        self.cancel_settle_until = 0.
        self.queued_at = 0.
        self.plan_time = 0.
        self.problem = ''
        self.problem_time = 0.
        self.external_active = False
        self.terminal = None
        self.last_state = None
        self.distance = None
        self.goal_source = ''
        self.goal_id = ''
        self.session_id = uuid.uuid4().hex
        self.patrol_heartbeat = 0.
        self.subs = [
            self.create_subscription(PoseStamped, '/navigation/rviz_goal_input', self.on_goal, 10),
            self.create_subscription(String, '/navigation/patrol_goal', self.on_patrol_goal, 10),
            self.create_subscription(String, '/navigation/patrol_heartbeat', self.on_patrol_heartbeat, 10),
            self.create_subscription(String, '/navigation/patrol_cancel', self.on_patrol_cancel, 10),
            self.create_subscription(Odometry, '/nav_predicted_odom', self.on_odom, 10),
            self.create_subscription(PointCloud2, '/nav_body_points', self.on_cloud, qos_profile_sensor_data),
            self.create_subscription(PointCloud2, '/nav_map_points', self.on_map, 1),
            self.create_subscription(Twist, '/cmd_vel', self.on_command, 10),
            self.create_subscription(Path, '/plan', self.on_plan, 10),
            self.create_subscription(String, '/chassis/status', self.on_chassis, latched),
            self.create_subscription(Log, '/rosout', self.on_log, 50),
            self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self.on_action_status, latched),
        ]
        self.cancel_service = self.create_service(Trigger, '/navigation/cancel', self.on_cancel)
        self.timer = self.create_timer(.2, self.tick)

    @staticmethod
    def stamp_ns(stamp):
        return stamp.sec*1000000000+stamp.nanosec

    def on_odom(self, msg):
        self.odom, self.odom_stamp = msg, self.stamp_ns(msg.header.stamp)

    def on_cloud(self, msg):
        self.cloud_stamp = self.stamp_ns(msg.header.stamp)

    def on_map(self, msg):
        if msg.width*msg.height > 0 and msg.header.frame_id == 'map':
            self.have_saved_map = True

    def on_command(self, msg):
        self.command, self.command_time = msg, time.monotonic()

    def on_plan(self, msg):
        if msg.poses:
            self.plan_time = time.monotonic()

    def on_chassis(self, msg):
        try:
            self.chassis = json.loads(msg.data)
            self.chassis_time = time.monotonic()
        except (TypeError, ValueError):
            pass

    def on_log(self, msg):
        if msg.level >= 30 and msg.name in ('controller_server', 'planner_server', 'bt_navigator'):
            # Preemption aborts an old action handle; it is not a path failure.
            if any(text in msg.msg for text in ('Aborting handle', 'goal checker was specified')):
                return
            # Nav2 emits generic action/BT wrappers after the useful cause.
            # Keep that cause across the bounded recovery attempts.
            generic = any(text in msg.msg for text in (
                'Goal failed', 'Controller patience exceeded',
                'failed to generate a valid path', 'Failed to get a valid path'))
            if generic and self.problem and time.monotonic()-self.problem_time < 15.:
                return
            detail = msg.msg
            if 'start or goal pose are an obstacle' in detail:
                detail = '起点或目标点落在障碍区域，请检查地图中的车体位置和目标位置'
            elif 'outside bounds' in detail or 'outside map bounds' in detail or 'outside the map' in detail:
                detail = '起点或目标点超出当前规划地图范围：'+detail
            elif 'Failed to make progress' in detail:
                detail = '已下发运动指令，但位置和朝向持续无进展，请检查底盘状态及实际起动速度'
            elif 'no safe moving speed' in detail:
                detail = '车体轮廓或制动轨迹被障碍阻挡，等待重新规划'
            self.problem, self.problem_time = detail, time.monotonic()

    def on_action_status(self, msg):
        self.external_active = any(s.status in (1,2,3) for s in msg.status_list)

    def on_patrol_goal(self, msg):
        # Arbitrate here, in the same executor as /goal_pose. A delayed patrol
        # publication must never replace a pending or executing manual goal.
        if self.goal_source == 'manual' and (self.pending is not None or
                self.sending or self.goal_handle is not None):
            return
        try:
            request = json.loads(msg.data)
            if request.get('lift_enabled'):
                self.get_logger().warning('Ignoring patrol request requiring unavailable lift control')
                return
            if request['session_id'] != self.session_id or request['expected_serial'] != self.serial:
                return  # The world changed since the route client chose a point.
            token = int(request['request_id'])
            if token <= 0 or token // 1000000000 > 2147483647:
                return
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp.sec, pose.header.stamp.nanosec = divmod(token, 1000000000)
            pose.pose.position.x = float(request['x'])
            pose.pose.position.y = float(request['y'])
            yaw = float(request['yaw'])
            pose.pose.orientation.z = math.sin(yaw / 2)
            pose.pose.orientation.w = math.cos(yaw / 2)
        except (TypeError, ValueError, KeyError, OverflowError):
            self.get_logger().warning('忽略无效巡航请求')
            return
        if str(token) == self.goal_id:
            return
        self.on_goal(pose, source='patrol')
        self.patrol_heartbeat = time.monotonic()

    def on_patrol_heartbeat(self, msg):
        if self.goal_source == 'patrol' and msg.data == self.goal_id:
            self.patrol_heartbeat = time.monotonic()

    def on_patrol_cancel(self, msg):
        if self.goal_source == 'patrol' and msg.data == self.goal_id:
            self.on_cancel(None, Trigger.Response())

    def on_goal(self, msg, source='manual'):
        p, q = msg.pose.position, msg.pose.orientation
        values = (p.x,p.y,p.z,q.x,q.y,q.z,q.w)
        norm = sum(v*v for v in (q.x,q.y,q.z,q.w))
        self.stop_active()
        self.serial += 1
        self.goal_source = source
        self.goal_id = str(self.stamp_ns(msg.header.stamp)) if source == 'patrol' else str(self.serial)
        self.sending = False
        if not msg.header.frame_id or not all(math.isfinite(v) for v in values) or norm < 1e-9:
            self.pending = None
            self.terminal = ('INVALID_GOAL', '目标坐标或朝向无效，请重新选择')
            return
        self.goal = self.pending = copy.deepcopy(msg)
        n = math.sqrt(norm)
        for attr in ('x','y','z','w'):
            setattr(self.pending.pose.orientation, attr, getattr(q,attr)/n)
        self.queued_at = time.monotonic()
        self.problem = ''
        self.distance = None
        self.terminal = None
        self.get_logger().info(f'收到导航目标 ({p.x:.2f}, {p.y:.2f})，检查定位、传感器和底盘状态')

    def stop_active(self):
        self.sensor_gap_started = None
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None

    def on_cancel(self, request, response):
        self.stop_active()
        self.serial += 1
        self.pending = None
        self.sending = False
        self.terminal = ('CANCELED', '导航已取消')
        response.success = True
        response.message = 'Navigation canceled, pending goal cleared'
        return response

    def readiness(self):
        if (any(s != self.serial for s in getattr(self, 'action_inflight', set())) or
                time.monotonic() < getattr(self, 'cancel_settle_until', 0.)):
            return 'WAIT_CANCEL', '等待旧导航任务完全取消，再执行最新目标'
        if self.require_chassis:
            if self.chassis is None or time.monotonic()-self.chassis_time > 2.:
                return 'WAIT_CHASSIS', '等待底盘状态'
            if self.chassis.get('monitor_enabled', True):
                if not self.chassis.get('heartbeat_fresh'):
                    return 'CHASSIS_OFFLINE', '底盘心跳丢失，请检查通信'
                if not self.chassis.get('armed'):
                    return 'CHASSIS_DISARMED', '底盘未解锁；节点不会自动解锁'
                if not self.chassis.get('guided'):
                    return 'CHASSIS_MODE', '底盘需要切换到 GUIDED 模式'
        if not self.client.server_is_ready() or any(
            name not in self.lifecycle_states or self.lifecycle_states[name][0] != 3 or
            time.monotonic()-self.lifecycle_states[name][1] > 2.
            for name in self.lifecycle_clients):
            return 'WAIT_NAVIGATION', '等待导航服务器启动'
        now = self.get_clock().now().nanoseconds
        def fresh(stamp, limit):
            return stamp > 0 and -.05 <= (now-stamp)*1e-9 <= limit
        if not fresh(self.odom_stamp, .35):
            return 'WAIT_ODOMETRY', '等待有效里程计；暂停导航'
        if not fresh(self.cloud_stamp, .60):
            return 'WAIT_SCAN', '等待新鲜雷达扫描；暂停导航'
        if self.require_saved_map and not self.have_saved_map:
            return 'WAIT_MAP', '等待保存地图进入导航，尚未接收到地图障碍点云'
        try:
            tf = self.buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            stamp = self.stamp_ns(tf.header.stamp)
            if stamp and not fresh(stamp, .35):
                return 'WAIT_LOCALIZATION', '全局定位或坐标变换已过期'
        except Exception:
            return 'WAIT_LOCALIZATION', '等待全局定位；必要时使用初始位姿工具定位'
        return None

    def guard_active_goal(self, reason, now):
        # The controller already commands zero at the original freshness limits.
        # Keep only the existing action alive for a bounded transient sensor gap.
        # Chassis / lifecycle gates are checked before sensors in readiness().
        if self.goal_handle is None or reason is None:
            self.sensor_gap_started = None
            return
        if reason[0] in ('WAIT_ODOMETRY', 'WAIT_SCAN', 'WAIT_LOCALIZATION'):
            if self.sensor_gap_started is None:
                self.sensor_gap_started = now
            if now-self.sensor_gap_started < self.sensor_gap_grace:
                return
        self.terminal = ('STOPPED', reason[1]+'；恢复后请重新下发目标')
        self.stop_active()

    def submit(self):
        if getattr(self, 'action_inflight', set()) or \
                time.monotonic() < getattr(self, 'cancel_settle_until', 0.):
            return
        serial = self.serial
        if not hasattr(self, 'action_inflight'):
            self.action_inflight = set()
        self.action_inflight.add(serial)
        goal = NavigateToPose.Goal()
        goal.pose = copy.deepcopy(self.pending)
        # Goal is a position in its declared frame, not a historical sensor sample.
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        self.pending = None
        self.sending = True
        self.plan_time = 0.
        def feedback(msg):
            if serial == self.serial:
                self.distance = float(msg.feedback.distance_remaining)
        future = self.client.send_goal_async(goal, feedback_callback=feedback)
        def accepted(done):
            handle = done.result()
            if serial != self.serial:
                if handle.accepted:
                    def stale_done(result):
                        self.action_inflight.discard(serial)
                        self.cancel_settle_until = time.monotonic() + .3
                    handle.get_result_async().add_done_callback(stale_done)
                    handle.cancel_goal_async()
                else:
                    self.action_inflight.discard(serial)
                return
            self.sending = False
            if not handle.accepted:
                self.action_inflight.discard(serial)
                self.terminal = ('REJECTED', '导航服务器拒绝目标；请检查生命周期状态')
                return
            self.goal_handle = handle
            def completed(result):
                self.action_inflight.discard(serial)
                if serial != self.serial:
                    self.cancel_settle_until = time.monotonic() + .3
                    return
                self.goal_handle = None
                status = result.result().status
                if self.terminal and self.terminal[0] == 'STOPPED':
                    return
                self.terminal = {4: ('SUCCEEDED','已到达目标'),
                    5: ('CANCELED','导航已取消')}.get(status,
                    ('FAILED', self.problem or '规划或控制失败，请检查目标位置及障碍'))
            handle.get_result_async().add_done_callback(completed)
        future.add_done_callback(accepted)

    def tick(self):
        if self.goal_source == 'patrol' and (self.pending is not None or
                self.sending or self.goal_handle is not None) and time.monotonic()-self.patrol_heartbeat > 2.:
            self.on_cancel(None, Trigger.Response())
            self.terminal = ('STOPPED', '巡航程序心跳丢失，任务已取消；请重新启动巡航')
        if time.monotonic()-self.lifecycle_poll_time > .5:
            self.lifecycle_poll_time = time.monotonic()
            for key, (future, started) in list(self.lifecycle_pending.items()):
                if time.monotonic()-started > 2.:
                    self.lifecycle_pending.pop(key, None)
                    future.cancel()
            for name, client in self.lifecycle_clients.items():
                if name in self.lifecycle_pending or not client.service_is_ready():
                    continue
                future = client.call_async(GetState.Request())
                self.lifecycle_pending[name] = (future, time.monotonic())
                def received(done, key=name):
                    current = self.lifecycle_pending.get(key)
                    if current is None or current[0] is not done:
                        return
                    self.lifecycle_pending.pop(key, None)
                    try:
                        if done.result() is not None:
                            self.lifecycle_states[key] = (done.result().current_state.id, time.monotonic())
                    except Exception:
                        self.lifecycle_states.pop(key, None)
                future.add_done_callback(received)
        reason = self.readiness()
        self.guard_active_goal(reason, time.monotonic())
        if self.pending is not None:
            if reason and reason[0] == 'WAIT_CANCEL':
                self.queued_at = time.monotonic()
            elif time.monotonic()-self.queued_at > self.wait_timeout:
                self.pending = None
                self.terminal = ('WAIT_TIMEOUT','等待导航条件超时，请重新下发目标')
            elif reason is None and not self.sending:
                self.submit()
        if self.terminal is not None:
            state, detail = self.terminal
        elif reason:
            state, detail = reason
            if self.goal_handle is not None and self.sensor_gap_started is not None:
                state = "PAUSED_SENSOR"
                detail += "；车辆已暂停，数据恢复后自动继续，持续中断将取消任务"
        elif self.sending or self.goal_handle is not None or self.external_active:
            v = self.command if time.monotonic()-self.command_time < .5 else Twist()
            if math.hypot(v.linear.x,v.linear.y) > .02:
                state, detail = 'MOVING', '正在沿路径行驶'
            elif abs(v.angular.z) > .02:
                state, detail = 'TURNING', '正在原地转向'
            elif self.problem and time.monotonic()-self.problem_time < 3.:
                state, detail = 'WAIT_REPLAN', self.problem
            elif self.plan_time:
                state, detail = 'WAIT_MOTION', '已有路径，等待运动或重新规划'
            else:
                state, detail = 'PLANNING', '正在规划路径'
        else:
            state, detail = 'READY', '导航就绪，等待目标'
        if state != self.last_state:
            self.get_logger().info(f'{state}: {detail}')
            self.last_state = state
        data = dict(state=state, detail=detail, queued=self.pending is not None,
                    active=self.goal_handle is not None or self.sending, distance_remaining_m=self.distance,
                    patrol_protocol=1, session_id=self.session_id, goal_source=self.goal_source,
                    goal_id=self.goal_id, goal_serial=self.serial)
        self.status_pub.publish(String(data=json.dumps(data,ensure_ascii=False)))
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns, marker.id, marker.type, marker.action = 'navigation_status', 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        marker.pose.orientation.w = 1.
        try:
            tf = self.buffer.lookup_transform('map','base_link',rclpy.time.Time())
            marker.pose.position.x = tf.transform.translation.x
            marker.pose.position.y = tf.transform.translation.y
        except Exception:
            pass
        marker.pose.position.z = 1.3
        marker.scale.z = .25
        marker.color.a = 1.
        marker.color.r, marker.color.g, marker.color.b = (0.3,1.,.4) if state in ('READY','MOVING','TURNING','SUCCEEDED') else (1.,.65,.15)
        marker.text = state + ('\nGoal queued' if self.pending is not None else '')
        if self.distance is not None:
            marker.text += f'\nRemaining: {self.distance:.2f} m'
        self.marker_pub.publish(MarkerArray(markers=[marker]))


def main(args=None):
    rclpy.init(args=args)
    node = NavigationSupervisor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.stop_active()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
