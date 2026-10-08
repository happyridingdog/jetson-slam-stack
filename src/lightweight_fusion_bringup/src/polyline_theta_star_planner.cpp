#include "nav2_theta_star_planner/theta_star_planner.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "nav2_core/exceptions.hpp"
#include "path_resampling.hpp"
#include "planner_clearance.hpp"
#include "nearby_side_goal.hpp"

namespace lightweight_fusion_bringup
{
class PolylineThetaStarPlanner : public nav2_theta_star_planner::ThetaStarPlanner
{
public:
  void configure(const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override
  {
    ThetaStarPlanner::configure(parent,name,tf,costmap_ros);
    costmap_ros_=costmap_ros;
    auto node=parent.lock();
    nav2_util::declare_parameter_if_not_declared(node,name+".preferred_obstacle_distance",
      rclcpp::ParameterValue(preferred_distance_));
    nav2_util::declare_parameter_if_not_declared(node,name+".tracking_clearance_margin",
      rclcpp::ParameterValue(tracking_margin_));
    node->get_parameter(name+".preferred_obstacle_distance",preferred_distance_);
    node->get_parameter(name+".tracking_clearance_margin",tracking_margin_);
    if (!std::isfinite(preferred_distance_) || preferred_distance_<0.4 || preferred_distance_>2.0 ||
      !std::isfinite(tracking_margin_) || tracking_margin_<0 || tracking_margin_>0.3) {
      throw nav2_core::PlannerException("Invalid planner clearance parameters");
    }
  }

  nav_msgs::msg::Path createPlan(const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal) override
  {
    auto map=costmap_ros_->getCostmap();
    const auto footprint=costmap_ros_->getRobotFootprint();
    std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(*map->getMutex());
    nav_msgs::msg::Path direct;
    direct.header.frame_id=global_frame_;direct.header.stamp=clock_->now();
    direct.poses={start,goal};
    for (auto & pose:direct.poses) {
      unsigned int mx,my;
      if (pose.header.frame_id!=global_frame_ ||
        !std::isfinite(pose.pose.position.x) || !std::isfinite(pose.pose.position.y) ||
        !map->worldToMap(pose.pose.position.x,pose.pose.position.y,mx,my)) {
        throw nav2_core::PlannerException("Start or goal coordinates outside map bounds/frame");
      }
      if (!footprintPoseFree(*map,footprint,pose.pose.position.x,pose.pose.position.y,
        planarYaw(pose.pose.orientation))) {
        throw nav2_core::PlannerException("Either of the start or goal pose are an obstacle! (filled footprint)");
      }
      pose.header=direct.header;
    }
    double inscribed=0,circumscribed=0;
    nav2_costmap_2d::calculateMinAndMaxDistances(footprint,inscribed,circumscribed);
    if (preferred_distance_<=inscribed+tracking_margin_) {
      throw nav2_core::PlannerException("Preferred clearance must exceed body radius plus tracking margin");
    }
    const double held_yaw=planarYaw(start.pose.orientation);
    if (nearbySideGoal(goal.pose.position.x-start.pose.position.x,
      goal.pose.position.y-start.pose.position.y,held_yaw)) {
      const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(50);
      auto padded=footprint;
      nav2_costmap_2d::padFootprint(padded,tracking_margin_);
      // Check the rectangular body at its held heading, including the separate
      // final rotation. A blocked side step falls back to ordinary routing.
      if (translationCorridorFree(*map,padded,start.pose.position,goal.pose.position,held_yaw,deadline) &&
        rotationCorridorFree(*map,padded,goal.pose.position,held_yaw,
          planarYaw(goal.pose.orientation),deadline)) {
        // Intermediate orientations encode the actual body attitude for both
        // the controller and Nav2 IsPathValid. Preserve the goal arrow at end.
        auto side=resamplePath(direct,std::min(0.05,map->getResolution()*0.5),12);
        RCLCPP_INFO(logger_,"Nearby side goal: translate at held heading, then align goal yaw");
        return side;
      }
    }
    auto search_map=clearanceSearchMap(*map,preferred_distance_,inscribed+tracking_margin_);
    ShortcutPreference preference;
    preference.costs=&search_map;
    preference.distance_weight=planner_->w_euc_cost_;
    preference.traversal_weight=planner_->w_traversal_cost_;
    preference.pose_allowed=trackingMarginCheck(*map,footprint,start.pose,goal.pose,tracking_margin_);
    auto simplify=[&](const nav_msgs::msg::Path & candidate,int milliseconds) {
        return shortcutPolyline(candidate,*map,footprint,
          std::chrono::steady_clock::now()+std::chrono::milliseconds(milliseconds),&preference);
      };
    // A feasible direct line is a candidate, not an unconditional shortcut
    // around cost-aware search. Compare it with routes having more clearance.
    auto corners=simplify(direct,50);
    double best_cost=polylineTraversalCost(corners,preference);
    auto consider=[&](nav_msgs::msg::Path candidate) {
        const double cost=polylineTraversalCost(candidate,preference);
        if (cost+0.01<best_cost) {best_cost=cost;corners=std::move(candidate);}
      };
    // The theoretical free-space lower bound proves a clear straight line
    // already optimal. Near obstacles we always run the cost-aware search.
    const double length=std::hypot(goal.pose.position.x-start.pose.position.x,
      goal.pose.position.y-start.pose.position.y);
    const double free_cost=length*(preference.distance_weight+
      preference.traversal_weight*std::pow(26.0/252.0,2));
    if (corners.poses.empty() || best_cost>free_cost+0.01) {
      struct RestoreMap {
        theta_star::ThetaStar & planner;
        nav2_costmap_2d::Costmap2D * original;
        ~RestoreMap() {planner.costmap_=original;}
      } restore{*planner_,planner_->costmap_};
      planner_->costmap_=&search_map;
      try {
        auto path=ThetaStarPlanner::createPlan(start,goal);
        if (!path.poses.empty()) {
          path.poses.front()=direct.poses.front();
          if (path.poses.size()==1) {path.poses.push_back(direct.poses.back());}
          else {path.poses.back()=direct.poses.back();}
          auto checked=simplify(path,200);
          RCLCPP_INFO(logger_,"Clearance candidates: direct %.2f, search %.2f, checked %.2f (%zu corners)",
            best_cost,polylineTraversalCost(path,preference),
            polylineTraversalCost(checked,preference),checked.poses.size());
          consider(std::move(checked));
        }
      } catch (const nav2_core::PlannerException & error) {
        RCLCPP_INFO(logger_,"Clearance grid search unavailable: %s",error.what());
        // Continuous exact poses may fit a centred corridor which grid-cell
        // snapping cannot represent. Retain only an already checked candidate.
        // If none exists yet, try the checked doorway-exit fallback below.
      }
    }
    if (corners.poses.empty()) {
      // Preserve the checked doorway-exit fallback: continue along the entry
      // heading until a turn fits. Compare feasible alternatives by clearance
      // cost instead of accepting the first (usually tightest) turn location.
      const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(100);
      const double step=std::max(0.05,map->getResolution()*0.5);
      auto padded=footprint;
      nav2_costmap_2d::padFootprint(padded,tracking_margin_);
      const bool start_has_margin=footprintPoseFree(*map,padded,start.pose.position.x,
        start.pose.position.y,planarYaw(start.pose.orientation));
      for (double distance=step;distance<=length+2*circumscribed &&
        std::chrono::steady_clock::now()<deadline;distance+=step) {
        for (bool from_start:{true,false}) {
          auto waypoint=from_start ? direct.poses.front() : direct.poses.back();
          const double heading=planarYaw(waypoint.pose.orientation);
          const double direction=from_start ? 1.0 : -1.0;
          waypoint.pose.position.x+=direction*distance*std::cos(heading);
          waypoint.pose.position.y+=direction*distance*std::sin(heading);
          auto candidate=direct;
          candidate.poses.insert(candidate.poses.begin()+1,waypoint);
          consider(shortcutPolyline(candidate,*map,footprint,deadline,&preference));
          // If already off-centre inside a tight passage, requiring the new
          // margin immediately can imprison a physically clear vehicle.
          // Permit ONLY a straight exit along its existing heading, ending
          // at a turn pose with the full margin. All subsequent legs and
          // turn sweeps still use the normal clearance checks.
          if (from_start && !start_has_margin && footprintPoseFree(*map,padded,
            waypoint.pose.position.x,waypoint.pose.position.y,heading) &&
            straightCorridorFree(*map,footprint,start.pose.position,waypoint.pose.position,deadline)) {
            auto tail=direct;tail.poses.front()=waypoint;
            tail=shortcutPolyline(tail,*map,footprint,deadline,&preference);
            if (!tail.poses.empty()) {
              tail.poses.insert(tail.poses.begin(),direct.poses.front());
              consider(std::move(tail));
            }
          }
        }
      }
    }
    if (corners.poses.empty()) {
      throw nav2_core::PlannerException(
        "No straight-leg route with tracking clearance and collision-free turns");
    }
    // Keep straight-line samples for Nav2 IsPathValid so a newly appearing
    // obstacle between corners still invalidates the route. The controller
    // publishes only the corner vertices in RViz and never fits a curve.
    return orientPolyline(resamplePath(corners,std::min(0.05,map->getResolution()*0.5),12));
  }
private:
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  double preferred_distance_{0.9},tracking_margin_{0.06};
};
}  // namespace lightweight_fusion_bringup

PLUGINLIB_EXPORT_CLASS(lightweight_fusion_bringup::PolylineThetaStarPlanner, nav2_core::GlobalPlanner)
