#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav2_core/controller.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav2_core/exceptions.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "collision_projection.hpp"
#include "path_tracking.hpp"
#include "polyline_shortcut.hpp"
#include "nav2_costmap_2d/footprint.hpp"

namespace lightweight_fusion_bringup
{

class ExclusiveVelocityController : public nav2_core::Controller
{
public:
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override
  {
    node_ = parent;
    name_ = std::move(name);
    tf_ = std::move(tf);
    costmap_ros_ = std::move(costmap_ros);
    auto node = node_.lock();
    if (!node || !tf_ || !costmap_ros_ || !costmap_ros_->getCostmap()) {
      throw nav2_core::PlannerException("Controller requires node, TF and local costmap");
    }

    auto param = [&](const std::string & key, double & value) {
        node->declare_parameter(name_ + "." + key, value);
        node->get_parameter(name_ + "." + key, value);
      };
    node->declare_parameter(name_+".heading_correction",tracking_.heading_correction);
    node->get_parameter(name_+".heading_correction",tracking_.heading_correction);
    node->declare_parameter(name_+".stopped_alignment",tracking_.stopped_alignment);
    node->get_parameter(name_+".stopped_alignment",tracking_.stopped_alignment);
    param("alignment_position_tolerance",tracking_.alignment_position_tolerance);
    param("alignment_heading_tolerance",tracking_.alignment_heading_tolerance);
    param("realign_cross_threshold",tracking_.realign_cross_threshold);
    param("realign_heading_threshold",tracking_.realign_heading_threshold);
    param("alignment_lateral_speed",tracking_.alignment_lateral_speed);
    param("tracking_lookahead",tracking_.tracking_lookahead);
    param("max_tracking_yaw",tracking_.max_tracking_yaw);
    param("max_linear_speed", tracking_.max_linear);
    param("max_lateral_speed", tracking_.max_lateral);
    param("max_angular_speed", tracking_.max_angular);
    param("min_angular_speed", tracking_.min_angular);
    param("preferred_min_angular_speed", tracking_.preferred_min_angular);
    param("turn_stop_latency", tracking_.turn_stop_latency);
    param("min_linear_speed", tracking_.min_linear);
    param("heading_exit_threshold", tracking_.heading_exit);
    param("lookahead_distance", tracking_.lookahead);
    param("xy_goal_tolerance", tracking_.xy_tolerance);
    param("yaw_goal_tolerance", tracking_.yaw_tolerance);
    param("approach_gain", tracking_.approach_gain);
    param("turn_gain", tracking_.turn_gain);
    param("segment_capture", tracking_.segment_capture);
    param("max_segment_drift", tracking_.max_segment_drift);
    param("straight_clearance_margin", straight_clearance_margin_);
    param("straight_preview_distance", straight_preview_distance_);
    param("goal_adjustment_distance", tracking_.goal_adjustment_distance);
    param("goal_adjustment_speed", tracking_.goal_adjustment_speed);
    param("odometry_timeout", odometry_timeout_);
    param("cloud_timeout", cloud_timeout_);
    param("input_pause_timeout", input_pause_timeout_);
    if (!std::isfinite(straight_clearance_margin_) || straight_clearance_margin_<0 ||
      straight_clearance_margin_>0.20 || !std::isfinite(straight_preview_distance_) ||
      straight_preview_distance_<0.5 || straight_preview_distance_>3.0 || odometry_timeout_ <= 0 || cloud_timeout_ <= 0 ||
      !std::isfinite(input_pause_timeout_) || input_pause_timeout_ <= 0) {
      throw nav2_core::PlannerException("Input freshness timeouts must be positive");
    }
    odometry_sub_ = node->create_subscription<nav_msgs::msg::Odometry>(
      "/nav_predicted_odom", 10, [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(input_mutex_);
        last_odometry_ns_ = rclcpp::Time(msg->header.stamp).nanoseconds();
      });
    cloud_sub_ = node->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/nav_body_points", rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(input_mutex_);
        last_cloud_ns_ = rclcpp::Time(msg->header.stamp).nanoseconds();
      });
    if (!tracking_.valid()) {throw nav2_core::PlannerException("Invalid path tracking parameters");}
    node->declare_parameter(name_ + ".collision_reaction_time", 0.65);
    node->declare_parameter(name_ + ".collision_horizon", 1.0);
    node->declare_parameter(name_ + ".collision_acceleration", std::vector<double>{0.65, 0.20, 0.80});
    node->declare_parameter(name_ + ".collision_deceleration", std::vector<double>{0.75, 0.20, 0.90});

    node->get_parameter(name_ + ".collision_reaction_time", projection_.reaction_time);
    node->get_parameter(name_ + ".collision_horizon", projection_.horizon);
    const auto accel = node->get_parameter(name_ + ".collision_acceleration").as_double_array();
    const auto decel = node->get_parameter(name_ + ".collision_deceleration").as_double_array();
    if (accel.size() != 3 || decel.size() != 3) {
      throw nav2_core::PlannerException("Collision acceleration/deceleration must have 3 axes");
    }
    std::copy(accel.begin(), accel.end(), projection_.acceleration.begin());
    std::copy(decel.begin(), decel.end(), projection_.deceleration.begin());
    if (!projection_.valid()) {
      throw nav2_core::PlannerException("Invalid collision projection limits");
    }
    tracking_.turn_deceleration=projection_.deceleration[2];
    tracking_.lateral_deceleration=projection_.deceleration[1];
    smoothed_sub_ = node->create_subscription<geometry_msgs::msg::Twist>(
      // Use an alias: navigation_launch remaps this node's cmd_vel to the
      // raw controller output. The bringup maps this distinct name to /cmd_vel.
      "controller_smoothed_feedback", 1, [this](geometry_msgs::msg::Twist::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(smoothed_mutex_);
        smoothed_velocity_ = {msg->linear.x, msg->linear.y, msg->angular.z};
        have_smoothed_velocity_ = true;
      });
    speed_limit_ = tracking_.max_linear;
    logger_ = node->get_logger();
    tracked_path_pub_ = node->create_publisher<nav_msgs::msg::Path>(
      "/navigation/tracked_path", rclcpp::QoS(1).transient_local());
  }

  void cleanup() override
  {
    input_gap_started_ns_ = 0;
    plan_.poses.clear();
    tracking_.setPlan({});
    smoothed_sub_.reset();
    odometry_sub_.reset();
    cloud_sub_.reset();
    tracked_path_pub_.reset();
    std::lock_guard<std::mutex> lock(smoothed_mutex_);
    have_smoothed_velocity_ = false;
  }

  void activate() override {tracked_path_pub_->on_activate();}
  void deactivate() override {tracked_path_pub_->on_deactivate();}

  void setPlan(const nav_msgs::msg::Path & path) override
  {
    if (plan_.poses!=path.poses || plan_.header.frame_id!=path.header.frame_id) {
      straight_margin_required_=false;
    }
    std::vector<TrackingPoint> points;
    points.reserve(path.poses.size());
    for (const auto & p : path.poses) {
      points.push_back({p.pose.position.x, p.pose.position.y, tf2::getYaw(p.pose.orientation)});
    }
    try {tracking_.setPlan(points);}
    catch (const std::invalid_argument & e) {throw nav2_core::PlannerException(e.what());}
    plan_ = path;
    if (tracked_path_pub_ && tracked_path_pub_->is_activated()) {
      auto display=plan_;
      display.poses.clear();
      if (!plan_.poses.empty()) {display.poses.push_back(plan_.poses.front());}
      for (size_t i : tracking_.legEnds()) {
        if (i>0) {display.poses.push_back(plan_.poses[i]);}
      }
      tracked_path_pub_->publish(display);
    }
  }

  geometry_msgs::msg::TwistStamped pauseForInput(int64_t now_ns, const std::string & reason)
  {
    if (input_gap_started_ns_ == 0) {input_gap_started_ns_ = now_ns;}
    if ((now_ns-input_gap_started_ns_)*1e-9 >= input_pause_timeout_) {
      throw nav2_core::PlannerException(reason + "; input recovery timed out");
    }
    // Keep the action and straight-leg state, but command zero immediately.
    // A data pause is not a new path: do not force a mid-leg realignment when
    // fresh data returns. The drift and collision guards still run on resume.
    last_command_ns_ = now_ns;
    RCLCPP_WARN_THROTTLE(logger_, *node_.lock()->get_clock(), 1000,
      "%s; paused at zero velocity awaiting fresh data", reason.c_str());
    geometry_msgs::msg::TwistStamped stop;
    stop.header.stamp = node_.lock()->now();
    stop.header.frame_id = costmap_ros_->getBaseFrameID();
    return stop;
  }

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker *) override
  {
    const auto now_ns = node_.lock()->now().nanoseconds();
    {
      std::lock_guard<std::mutex> lock(input_mutex_);
      auto fresh = [now_ns](int64_t stamp, double timeout) {
          const double age = (now_ns-stamp)*1e-9;
          return stamp > 0 && age >= -0.05 && age <= timeout;
        };
      if (!fresh(last_odometry_ns_, odometry_timeout_)) {
        return pauseForInput(now_ns, "Navigation odometry missing or stale");
      }
      if (!fresh(last_cloud_ns_, cloud_timeout_)) {
        return pauseForInput(now_ns, "Live obstacle scan missing or stale");
      }
    }
    auto tracking_pose = pose;
    if (!plan_.poses.empty() && plan_.header.frame_id != pose.header.frame_id) {
      try {
        const auto transform = tf_->lookupTransform(
          plan_.header.frame_id, pose.header.frame_id, tf2::TimePointZero);
        const auto stamp = rclcpp::Time(transform.header.stamp).nanoseconds();
        if (stamp > 0 && (now_ns-stamp)*1e-9 > odometry_timeout_) {
          return pauseForInput(now_ns, "Global localization transform is stale");
        }
        tf2::doTransform(pose, tracking_pose, transform);
      } catch (const tf2::TransformException & exc) {
        return pauseForInput(now_ns, std::string("Path tracking TF unavailable: ") + exc.what());
      }
    }
    input_gap_started_ns_ = 0;
    TrackingVelocity residual{velocity.linear.x, velocity.linear.y, velocity.angular.z};
    {
      std::lock_guard<std::mutex> lock(smoothed_mutex_);
      if (have_smoothed_velocity_) {
        auto larger = [](double a, double b) {return std::abs(a) > std::abs(b) ? a : b;};
        residual.x = larger(residual.x, smoothed_velocity_[0]);
        residual.y = larger(residual.y, smoothed_velocity_[1]);
        residual.yaw = larger(residual.yaw, smoothed_velocity_[2]);
      }
    }
    const double dt = last_command_ns_ > 0 ? (now_ns-last_command_ns_)*1e-9 : 1.0/15.0;
    last_command_ns_ = now_ns;
    auto command = computeCandidate(tracking_pose, residual, dt,
      {velocity.linear.x, velocity.linear.y, velocity.angular.z});
    {
      // A zero raw command does not instantly stop the smoother. Wait for its
      // previous axis group to reach zero before requesting the other group.
      std::lock_guard<std::mutex> lock(smoothed_mutex_);
      if (!tracking_.trackingForward() && !tracking_.aligningReference() && have_smoothed_velocity_ &&
        ((std::abs(command.twist.angular.z)>0 &&
        std::hypot(smoothed_velocity_[0],smoothed_velocity_[1])>1e-6) ||
        (std::hypot(command.twist.linear.x,command.twist.linear.y)>0 &&
        std::abs(smoothed_velocity_[2])>1e-6))) {
        command.twist = geometry_msgs::msg::Twist{};
      }
    }
    geometry_msgs::msg::PoseStamped costmap_pose;
    try {
      if (pose.header.frame_id == costmap_ros_->getGlobalFrameID()) {costmap_pose = pose;}
      else {tf_->transform(pose, costmap_pose, costmap_ros_->getGlobalFrameID(),
          tf2::durationFromSec(costmap_ros_->getTransformTolerance()));}
    } catch (const tf2::TransformException & exc) {
      throw nav2_core::PlannerException(std::string("Collision check TF unavailable: ") + exc.what());
    }
    const Velocity requested{command.twist.linear.x, command.twist.linear.y, command.twist.angular.z};
    const Velocity measured{velocity.linear.x, velocity.linear.y, velocity.angular.z};
    // The planner/entry check reserve tracking margin. During feedback
    // correction check the physical footprint and full braking trajectory;
    // treating that extra margin as body can forbid the correction itself.
    const auto footprint = costmap_ros_->getRobotFootprint();
    const auto & p = costmap_pose.pose.position;
    const double yaw = tf2::getYaw(costmap_pose.pose.orientation);
    auto free = [&](const Velocity & initial, const Velocity & desired) {
        return projection_.isFree(*costmap_ros_->getCostmap(), footprint,
          p.x, p.y, yaw, initial, desired, tracking_.remainingTurn(),tracking_.remainingAlignmentLateral());
      };
    bool safe = false;
    std::vector<Velocity> modes{requested};
    if (tracking_.aligningReference() && requested[0]==0 && requested[1]!=0 && requested[2]!=0) {
      // A simultaneous correction may not fit a narrow passage. Try the
      // checked lateral centering first, then a checked pure rotation.
      modes.push_back({0,requested[1],0});modes.push_back({0,0,requested[2]});
    }
    if (tracking_.aligningReference() && (requested[1]!=0 || requested[2]!=0)) {
      // Let an already-started lateral/turn braking tail finish when adding
      // the other axis is not yet safe. This zero candidate is also checked.
      modes.push_back({0,0,0});
    }
    for (const auto & mode:modes) {
      std::vector<double> scales{1.0,0.75,0.50,0.25};
      const double requested_linear=std::hypot(mode[0],mode[1]);
      if (tracking_.min_linear>0 && requested_linear>tracking_.min_linear) {
        scales.push_back(tracking_.min_linear/requested_linear);
      }
      if (requested_linear==0 && tracking_.min_angular>0 && std::abs(mode[2])>tracking_.min_angular) {
        scales.push_back(tracking_.min_angular/std::abs(mode[2]));
      }
      for (double scale:scales) {
        const Velocity desired{mode[0]*scale,mode[1]*scale,mode[2]*scale};
        const double linear=std::hypot(desired[0],desired[1]);
        if (scale<1.0 && ((linear>0 && linear+1e-9<tracking_.min_linear) ||
          ((linear==0 || tracking_.aligningReference()) && std::abs(desired[2])>0 &&
            std::abs(desired[2])+1e-9<tracking_.min_angular))) {continue;}
        // Finite alignment pulses are projected from actual measured and
        // smoothed velocities, with their full hold and braking phases.
        // They need not instantaneously attain cruise speed at the start.
        safe=free(measured,desired) && (tracking_.aligningReference() || free(desired,desired));
        {
          std::lock_guard<std::mutex> lock(smoothed_mutex_);
          if (have_smoothed_velocity_) {safe=safe && free(smoothed_velocity_,desired);}
        }
        if (safe) {
          command.twist.linear.x=desired[0];command.twist.linear.y=desired[1];command.twist.angular.z=desired[2];
          if (scale<1.0) {
            RCLCPP_INFO_THROTTLE(logger_,*node_.lock()->get_clock(),2000,
              "Reducing speed to fit the checked stopping corridor (scale %.2f)",scale);
          }
          break;
        }
      }
      if (safe) {break;}
    }
    if (!safe) {
      throw nav2_core::PlannerException("Local footprint trajectory blocked; no safe moving speed");
    }
    return command;
  }

  geometry_msgs::msg::TwistStamped computeCandidate(
    const geometry_msgs::msg::PoseStamped & pose, const TrackingVelocity & residual, double dt,
    const TrackingVelocity & measured)
  {
    geometry_msgs::msg::TwistStamped command;
    command.header.stamp = node_.lock()->now();
    command.header.frame_id = costmap_ros_->getBaseFrameID();
    TrackingVelocity v;
    try {
      auto actual_straight_free=[&](const TrackingPoint & from,const TrackingPoint & end) {
          geometry_msgs::msg::PoseStamped start=pose;
          if (start.header.frame_id!=costmap_ros_->getGlobalFrameID()) {
            tf_->transform(pose,start,costmap_ros_->getGlobalFrameID(),
              tf2::durationFromSec(costmap_ros_->getTransformTolerance()));
          }
          const double heading=tf2::getYaw(start.pose.orientation);
          // Stay within the local rolling map; runtime braking checks remain
          // active throughout the rest of a long leg.
          // A distant ray amplifies harmless heading error into repeated
          // alignment/replans. Preview locally, but never below the current
          // reaction + braking distance and a 0.30 m anticipation reserve.
          // The separate swept-footprint projection still checks all motion.
          const double speed=std::hypot(residual.x,residual.y);
          const double stopping=speed*projection_.reaction_time+
            speed*speed/(2*projection_.deceleration[0])+0.30;
          const double preview=tracking_.stopped_alignment ?
            std::min(3.0,std::max(straight_preview_distance_,stopping)) : 3.0;
          const double distance=std::min(preview,std::hypot(end.x-from.x,end.y-from.y));
          auto finish=start.pose.position;
          finish.x+=distance*std::cos(heading);finish.y+=distance*std::sin(heading);
          auto map=costmap_ros_->getCostmap();
          std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(*map->getMutex());
          const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(20);
          auto footprint=costmap_ros_->getRobotFootprint();
          auto padded=footprint;
          nav2_costmap_2d::padFootprint(padded,straight_clearance_margin_);
          const bool margin_fits=footprintPoseFree(*map,padded,
            start.pose.position.x,start.pose.position.y,heading);
          if (tracking_.aligningStraightLeg() && std::hypot(residual.x,residual.y)<=0.02) {
            // A completed early stop may have used some of the extra margin.
            // Reacquire it along a newly checked heading from the stopped
            // pose, just as when starting an escape from a tight passage.
            straight_margin_required_=margin_fits;
          }
          // Preserve a buffer during motion. A vehicle already close to a
          // wall must still be able to take its checked straight exit.
          if (straight_margin_required_ || margin_fits) {
            // Once acquired, never drop the buffer merely because drift
            // consumed it while driving. Reset only after stopping or for
            // a new checked plan.
            straight_margin_required_=true;
            footprint=std::move(padded);
          }
          const bool actual_free=straightCorridorFree(*map,footprint,start.pose.position,finish,deadline);
          if (std::chrono::steady_clock::now()>=deadline) {
            throw std::runtime_error("Straight-leg clearance check exceeded its time budget");
          }
          return actual_free;
        };
      auto lateral_free=[&](const TrackingPoint &,const TrackingPoint & end) {
          auto start=pose,finish=pose;
          finish.pose.position.x=end.x;finish.pose.position.y=end.y;
          if (pose.header.frame_id!=costmap_ros_->getGlobalFrameID()) {
            const auto transform=tf_->lookupTransform(costmap_ros_->getGlobalFrameID(),
              pose.header.frame_id,tf2::TimePointZero,
              tf2::durationFromSec(costmap_ros_->getTransformTolerance()));
            tf2::doTransform(pose,start,transform);
            const auto original_finish=finish;
            tf2::doTransform(original_finish,finish,transform);
          }
          auto map=costmap_ros_->getCostmap();
          std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(*map->getMutex());
          auto footprint=costmap_ros_->getRobotFootprint();
          nav2_costmap_2d::padFootprint(footprint,straight_clearance_margin_);
          return translationCorridorFree(*map,footprint,start.pose.position,finish.pose.position,
            tf2::getYaw(start.pose.orientation),
            std::chrono::steady_clock::now()+std::chrono::milliseconds(20));
        };
      const auto previous_realignments=tracking_.clearanceRealignments();
      v = tracking_.command({pose.pose.position.x, pose.pose.position.y,
          tf2::getYaw(pose.pose.orientation)}, residual, speed_limit_, dt, measured, actual_straight_free, lateral_free);
      if (tracking_.trackingForward() && std::abs(tracking_.crossTrackError())>0.025) {
        RCLCPP_INFO_THROTTLE(logger_,*node_.lock()->get_clock(),1000,
          "Tracking planned edge: cross error %.3f m, vx %.3f, wz %.3f",
          tracking_.crossTrackError(),v.x,v.yaw);
      }
      if (tracking_.clearanceRealignments()!=previous_realignments) {
        RCLCPP_INFO(logger_,"Straight-leg clearance ahead lost; stopping before realignment at (%.2f, %.2f)",
          pose.pose.position.x,pose.pose.position.y);
      }
    } catch (const std::runtime_error & exc) {
      throw nav2_core::PlannerException(exc.what());
    }
    command.twist.linear.x = v.x;
    command.twist.linear.y = v.y;
    command.twist.angular.z = v.yaw;
    return command;
  }

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override
  {
    // Nav2 defines zero as NO_SPEED_LIMIT, not a request to stop the robot.
    speed_limit_ = speed_limit == 0.0 ? tracking_.max_linear :
      (percentage ? tracking_.max_linear * speed_limit / 100.0 : speed_limit);
    speed_limit_ = std::isfinite(speed_limit_) ?
      std::clamp(speed_limit_, 0.0, tracking_.max_linear) : 0.0;
  }

private:
  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  rclcpp::Logger logger_{rclcpp::get_logger("exclusive_velocity_controller")};
  std::string name_;
  nav_msgs::msg::Path plan_;
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>::SharedPtr tracked_path_pub_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  CollisionProjection projection_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr smoothed_sub_;
  std::mutex smoothed_mutex_;
  Velocity smoothed_velocity_{};
  bool have_smoothed_velocity_{false};
  PathTracking tracking_;
  double straight_clearance_margin_{0.04};
  double straight_preview_distance_{3.0};
  bool straight_margin_required_{false};
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  std::mutex input_mutex_;
  int64_t last_odometry_ns_{0}, last_cloud_ns_{0}, last_command_ns_{0};
  int64_t input_gap_started_ns_{0};
  double odometry_timeout_{0.35}, cloud_timeout_{0.60};
  double input_pause_timeout_{3.0};
  double speed_limit_{0.45};
};

}  // namespace lightweight_fusion_bringup

PLUGINLIB_EXPORT_CLASS(
  lightweight_fusion_bringup::ExclusiveVelocityController, nav2_core::Controller)
