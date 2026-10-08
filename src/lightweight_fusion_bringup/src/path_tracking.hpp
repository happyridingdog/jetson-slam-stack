#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include "nearby_side_goal.hpp"

namespace lightweight_fusion_bringup
{
struct TrackingPoint {double x, y, yaw{0.0};};
struct TrackingVelocity {double x{0.0}, y{0.0}, yaw{0.0};};

// Geometry and motion state are independent of ROS so closed-loop tests can
// exercise the same controller used by the plugin, including actuator lag.
class PathTracking
{
public:
  bool heading_correction{false}, stopped_alignment{false};
  double alignment_position_tolerance{0.015}, alignment_heading_tolerance{0.010};
  double realign_cross_threshold{0.04}, realign_heading_threshold{0.035};
  double alignment_lateral_speed{0.10}, lateral_deceleration{0.20};
  double tracking_lookahead{0.60}, max_tracking_yaw{0.20};
  double max_linear{0.45}, max_lateral{0.10}, max_angular{0.50};
  double min_linear{0.10};
  double min_angular{0.18}, preferred_min_angular{0.27}, heading_exit{0.025};
  double turn_stop_latency{0.25}, turn_deceleration{0.90};
  double lookahead{0.65}, xy_tolerance{0.20}, yaw_tolerance{0.087266};
  double approach_gain{0.8}, turn_gain{1.4};
  double segment_capture{0.12}, max_segment_drift{0.25};
  double goal_adjustment_distance{0.45}, goal_adjustment_speed{0.10};

  static double angle(double a) {return std::atan2(std::sin(a), std::cos(a));}

  bool valid() const
  {
    for (double v : {max_linear, max_lateral, max_angular, min_linear, min_angular, preferred_min_angular,
      turn_stop_latency, turn_deceleration,
      heading_exit, lookahead, xy_tolerance, yaw_tolerance, approach_gain, turn_gain,
      segment_capture, max_segment_drift,
      goal_adjustment_distance, goal_adjustment_speed, tracking_lookahead, max_tracking_yaw,
      alignment_position_tolerance, alignment_heading_tolerance, realign_cross_threshold,
      realign_heading_threshold, alignment_lateral_speed, lateral_deceleration})
    {
      if (!std::isfinite(v)) {return false;}
    }
    return !(heading_correction && stopped_alignment) &&
           alignment_position_tolerance>0 && alignment_heading_tolerance>0 &&
           realign_cross_threshold>alignment_position_tolerance &&
           realign_heading_threshold>alignment_heading_tolerance &&
           alignment_lateral_speed>=min_linear && alignment_lateral_speed<=max_lateral &&
           lateral_deceleration>0 && tracking_lookahead>=0.25 && max_tracking_yaw>0 && max_tracking_yaw<=max_angular &&
           max_linear > 0 && max_lateral > 0 && max_angular > 0 &&
           min_linear >= 0 && min_linear <= max_linear &&
           min_angular >= 0 && min_angular <= preferred_min_angular &&
           preferred_min_angular <= max_angular &&
           turn_stop_latency >= 0 && turn_deceleration > 0 &&
           heading_exit > 0 && heading_exit < 0.2 &&
           lookahead > 0 && xy_tolerance > 0 && yaw_tolerance > 0 &&
           approach_gain > 0 && turn_gain > 0 &&
           segment_capture > 0 && max_segment_drift > segment_capture &&
           goal_adjustment_distance > xy_tolerance && goal_adjustment_speed >= min_linear &&
           goal_adjustment_speed <= max_linear && max_lateral >= min_linear;
  }

  void setPlan(const std::vector<TrackingPoint> & points)
  {
    for (const auto & p : points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.yaw)) {
        throw std::invalid_argument("Path contains non-finite coordinates");
      }
    }
    // Re-sending the same reference must not restart a straight leg.
    if (points.size() == path_.size() && std::equal(points.begin(), points.end(), path_.begin(),
      [](const auto & a, const auto & b) {
        return a.x == b.x && a.y == b.y && a.yaw == b.yaw;
      })) {return;}
    const bool same_goal = !points.empty() && !path_.empty() &&
      std::hypot(points.back().x - path_.back().x, points.back().y - path_.back().y) < 0.005 &&
      std::abs(angle(points.back().yaw - path_.back().yaw)) < 0.02;
    if (!same_goal) {
      at_goal_ = false; rotation_started_ = false; turn_direction_ = 0;
      goal_adjusting_ = false;
    }
    path_ = points;
    arc_.assign(path_.size(), 0.0);
    for (size_t i = 1; i < path_.size(); ++i) {
      arc_[i] = arc_[i-1] + std::hypot(path_[i].x-path_[i-1].x, path_[i].y-path_[i-1].y);
    }
    corners_.clear();
    for (size_t i = 1; i+1 < path_.size(); ++i) {
      const double before = arc_[i]-arc_[i-1], after = arc_[i+1]-arc_[i];
      if (before < 1e-6 || after < 1e-6) {continue;}
      const double cosine = ((path_[i].x-path_[i-1].x)*(path_[i+1].x-path_[i].x) +
        (path_[i].y-path_[i-1].y)*(path_[i+1].y-path_[i].y)) / (before*after);
      if (cosine < 0.5) {corners_.push_back(i);}  // Turns sharper than 60 degrees.
    }
    progress_ = -1.0;
    leg_initialized_ = false; needs_alignment_ = true; rotation_started_ = false;
    alignment_active_=false;strafe_braking_=false;alignment_settle_=0;
    // Theta* already plans collision-aware straight edges. Remove only
    // collinear interpolation samples; retain every real bend and reversal.
    // No curve fitting or shortcutting across the planned obstacle corridor.
    leg_ends_.clear();
    size_t anchor=0, previous=0;
    for (size_t i=1; i<path_.size(); ++i) {
      const double dx=path_[i].x-path_[previous].x, dy=path_[i].y-path_[previous].y;
      const double next_length=std::hypot(dx,dy);
      if (next_length<1e-9) {continue;}
      if (previous>anchor) {
        const double ax=path_[previous].x-path_[anchor].x, ay=path_[previous].y-path_[anchor].y;
        if (ax*dx+ay*dy<=0 || std::abs(ax*dy-ay*dx)>1e-4*std::hypot(ax,ay)*next_length) {
          leg_ends_.push_back(previous);anchor=previous;
        }
      }
      previous=i;
    }
    if (!path_.empty()) {leg_ends_.push_back(path_.size()-1);}
    lateral_plan_=path_.size()>1 && leg_ends_.size()==1 &&
      nearbySideGoal(path_.back().x-path_.front().x,path_.back().y-path_.front().y,path_.front().yaw);
    if (lateral_plan_) {
      for (size_t i=0;i+1<path_.size();++i) {
        if (std::abs(angle(path_[i].yaw-path_.front().yaw))>1e-4) {lateral_plan_=false;}
      }
    }
  }

  const std::vector<size_t> & legEnds() const {return leg_ends_;}
  double remainingTurn() const {return remaining_turn_;}
  size_t clearanceRealignments() const {return clearance_realignments_;}
  bool aligningReference() const {return alignment_motion_;}
  double remainingAlignmentLateral() const {return remaining_alignment_lateral_;}
  bool trackingForward() const {return tracking_forward_;}
  double crossTrackError() const {return cross_track_error_;}
  bool aligningStraightLeg() const {return needs_alignment_;}

  TrackingPoint target(double x, double y)
  {
    if (path_.empty()) {return {x, y};}
    double best = std::numeric_limits<double>::infinity();
    double projected_arc = std::max(0.0, progress_);
    // After initial acquisition, a spatially nearby later leg of a loop must
    // not steal the target. Advance within a bounded arc-length neighborhood.
    const double end = progress_ < 0 ? arc_.back() : progress_ + std::max(2.0, 3*lookahead);
    for (size_t i = 1; i < path_.size(); ++i) {
      if (arc_[i] < progress_ || arc_[i-1] > end) {continue;}
      const auto & a = path_[i-1];
      const auto & b = path_[i];
      const double length = arc_[i] - arc_[i-1];
      if (length < 1e-9) {continue;}
      const double low = std::max(0.0, (progress_ - arc_[i-1]) / length);
      const double high = std::min(1.0, (end - arc_[i-1]) / length);
      const double t = std::clamp(((x-a.x)*(b.x-a.x)+(y-a.y)*(b.y-a.y)) /
        (length*length), low, high);
      const double distance = std::hypot(a.x+t*(b.x-a.x)-x, a.y+t*(b.y-a.y)-y);
      if (distance < best) {best = distance; projected_arc = arc_[i-1] + t*length;}
    }
    progress_ = projected_arc;
    double wanted = std::min(arc_.back(), progress_ + lookahead);
    corner_distance_ = std::numeric_limits<double>::infinity();
    for (size_t i : corners_) {
      if (arc_[i] <= progress_ + 1e-6 || arc_[i] > wanted) {continue;}
      const double distance = std::hypot(path_[i].x-x, path_[i].y-y);
      if (distance <= 0.10) {
        // Once captured, advance beyond this corner even while stopping to
        // turn. Otherwise localization jitter can select the incoming leg again.
        progress_ = arc_[i];
        wanted = std::min(arc_.back(), progress_ + lookahead);
      } else {
        // Reach a sharp corner before aiming down its outgoing leg. Looking
        // across it can cut
        // the corner. Approach speed decreases continuously with distance.
        wanted = arc_[i];
        corner_distance_ = distance;
      }
      break;
    }
    for (size_t i = 1; i < path_.size(); ++i) {
      const double length = arc_[i]-arc_[i-1];
      if (arc_[i] < wanted || length < 1e-9) {continue;}
      const double t = (wanted-arc_[i-1]) / length;
      return {path_[i-1].x+t*(path_[i].x-path_[i-1].x),
        path_[i-1].y+t*(path_[i].y-path_[i-1].y)};
    }
    return path_.back();
  }

  TrackingVelocity command(const TrackingPoint & pose, const TrackingVelocity & residual,
    double speed_limit, double dt = 1.0/15.0, const TrackingVelocity & measured = {},
    const std::function<bool(const TrackingPoint &, const TrackingPoint &)> & straight_allowed = {},
    const std::function<bool(const TrackingPoint &, const TrackingPoint &)> & lateral_allowed = {})
  {
    tracking_forward_=false;alignment_motion_=false;
    remaining_alignment_lateral_=std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(remaining_turn_) || dt>0.5) {turn_braking_=false;}
    remaining_turn_=std::numeric_limits<double>::quiet_NaN();
    measured_turn_rate_=measured.yaw;
    residual_turn_rate_=residual.yaw;
    if (!std::isfinite(dt) || dt <= 0 || path_.empty() || speed_limit <= 0) {return {};}
    if (dt > 0.5) {needs_alignment_ = true; rotation_started_ = false;alignment_active_=false;}
    target(pose.x, pose.y);
    if (!leg_initialized_) {
      leg_ = 0;
      while (leg_+1 < leg_ends_.size() && arc_[leg_ends_[leg_]] < progress_-1e-6) {++leg_;}
      leg_initialized_ = true;
    }
    const bool translating = std::hypot(residual.x, residual.y) > 0.02;
    const bool turning = std::abs(residual.yaw) > 0.02;
    const auto & goal = path_.back();
    const double distance_goal = std::hypot(goal.x-pose.x, goal.y-pose.y);
    // Advance only at a leg's endpoint plane, allowing bounded sideways drift.
    // The next bearing is acquired after braking, from the actual stopped pose.
    if (!needs_alignment_ && leg_+1 < leg_ends_.size()) {
      const auto & end = path_[leg_ends_[leg_]];
      const double dx=end.x-pose.x, dy=end.y-pose.y;
      const double along=std::cos(drive_heading_)*dx+std::sin(drive_heading_)*dy;
      if (along <= segment_capture &&
        (!(heading_correction || stopped_alignment) || std::hypot(dx,dy)<=segment_capture)) {
        if (std::abs(-std::sin(drive_heading_)*dx+std::cos(drive_heading_)*dy) > max_segment_drift) {
          throw std::runtime_error("Straight leg missed its waypoint; replan from current pose");
        }
        ++leg_; needs_alignment_=true; rotation_started_=false;alignment_active_=false;
      }
    }
    const bool last_leg = leg_+1 == leg_ends_.size();
    if (at_goal_ && distance_goal > xy_tolerance+0.15) {at_goal_=false;rotation_started_=false;}
    at_goal_ = at_goal_ || (last_leg && distance_goal <= xy_tolerance);
    if (at_goal_) {
      if (!rotation_started_ && translating) {return {};}
      rotation_started_ = true;
      const double error=angle(goal.yaw-pose.yaw);
      return {0,0,std::abs(error)>yaw_tolerance ? turn(error,yaw_tolerance) : 0.0};
    }
    // Only a planner-checked direct path with held body orientations enables
    // this exception. A detour or an ordinary forward path cannot trigger it.
    if (lateral_plan_ && lateral_allowed) {
      if (turning) {return {};}
      if (std::abs(angle(pose.yaw-path_.front().yaw))>0.15) {
        throw std::runtime_error("Heading drift during side step; stop and replan");
      }
      if (!lateral_allowed(pose,goal)) {
        throw std::runtime_error("Side-step corridor blocked; stop and replan");
      }
      rotation_started_=false;
      const double c=std::cos(pose.yaw),s=std::sin(pose.yaw);
      const double dx=goal.x-pose.x,dy=goal.y-pose.y;
      const double ux=(c*dx+s*dy)/distance_goal,uy=(-s*dx+c*dy)/distance_goal;
      // Cruise laterally, then approach at the existing motor-effective fine
      // adjustment speed. Fore/aft correction remains limited independently.
      double speed=std::min({speed_limit,max_lateral,
        std::max(goal_adjustment_speed,approach_gain*std::max(0.0,distance_goal-xy_tolerance))});
      if (std::abs(ux*speed)>goal_adjustment_speed) {speed=goal_adjustment_speed/std::abs(ux);}
      if (std::abs(uy*speed)>max_lateral) {speed=max_lateral/std::abs(uy);}
      return {ux*speed,uy*speed,0};
    }
    const double adjustment_limit=goal_adjustment_distance+(goal_adjusting_ ? 0.10 : 0.0);
    const bool adjust=last_leg && distance_goal<=adjustment_limit &&
      arc_.back()-progress_<=adjustment_limit;
    if (adjust) {
      // Even if reached while rotating, first stop yaw; planar fine adjustment
      // and final heading alignment are strictly separate phases.
      if (turning) {return {};}
      goal_adjusting_=true; rotation_started_=false;
      const double c=std::cos(pose.yaw), s=std::sin(pose.yaw);
      const double dx=goal.x-pose.x, dy=goal.y-pose.y;
      const double ux=(c*dx+s*dy)/distance_goal, uy=(-s*dx+c*dy)/distance_goal;
      double speed=std::min({speed_limit,goal_adjustment_speed,
        std::max(min_linear,approach_gain*distance_goal)});
      if (std::abs(uy*speed)>max_lateral) {speed=max_lateral/std::abs(uy);}
      return {ux*speed,uy*speed,0};
    }
    if (goal_adjusting_) {needs_alignment_=true;rotation_started_=false;goal_adjusting_=false;}
    const auto & end=path_[leg_ends_[leg_]];
    const auto & reference_start=path_[leg_==0 ? 0 : leg_ends_[leg_-1]];
    const double reference_heading=std::atan2(end.y-reference_start.y,end.x-reference_start.x);
    const double rc=std::cos(reference_heading),rs=std::sin(reference_heading);
    const double reference_along=rc*(end.x-pose.x)+rs*(end.y-pose.y);
    const double reference_cross=-rs*(pose.x-reference_start.x)+rc*(pose.y-reference_start.y);
    if (stopped_alignment) {
      return stoppedReferenceCommand(pose,residual,speed_limit,dt,measured,end,
        reference_start,reference_heading,reference_along,reference_cross,straight_allowed);
    }
    // Aim only along THIS edge. Never look through a bend, and never redefine
    // the reference line through the vehicle's drifted stopping position.
    const double ahead=std::max(0.02,std::min(reference_along,tracking_lookahead));
    TrackingPoint aim{pose.x+rc*ahead+rs*reference_cross,
      pose.y+rs*ahead-rc*reference_cross,reference_heading};
    if (!heading_correction) {aim=end;}
    if (needs_alignment_) {
      if (!rotation_started_ && translating) {return {};}
      const double error=angle(std::atan2(aim.y-pose.y,aim.x-pose.x)-pose.yaw);
      if (std::abs(error)>heading_exit) {rotation_started_=true;return {0,0,turn(error,heading_exit)};}
      if (turning) {return {};}
      // Nominal heading tolerance is not enough in a tight passage: check
      // the motion along the ACTUAL body heading before committing to a leg.
      // Refine entry alignment before starting forward tracking.
      if (straight_allowed && !straight_allowed(pose,aim)) {
        if (std::abs(error)>0.006) {rotation_started_=true;return {0,0,turn(error,0.006)};}
        throw std::runtime_error("Actual straight-leg corridor blocked; stop and replan");
      }
      needs_alignment_=false;rotation_started_=false;
      drive_heading_=heading_correction ? reference_heading : std::atan2(end.y-pose.y,end.x-pose.x);
      drive_origin_=heading_correction ? reference_start : pose;
    }
    const double c=std::cos(drive_heading_), s=std::sin(drive_heading_);
    const double cross=-s*(pose.x-drive_origin_.x)+c*(pose.y-drive_origin_.y);
    cross_track_error_=cross;
    // A large departure needs a new checked route. Within this bound, the
    // optional heading feedback below corrects against the original edge.
    if (std::abs(cross)>max_segment_drift || std::abs(angle(pose.yaw-drive_heading_))>0.35) {
      throw std::runtime_error("Excessive drift on straight leg; stop and replan (cross="+
        std::to_string(cross)+" m, heading="+std::to_string(angle(pose.yaw-drive_heading_))+" rad)");
    }
    // Recheck during motion as well as at entry. A long open-loop leg can
    // lose its clearance while still below the absolute drift limit. Stop
    // and realign while there is space left to turn, not only once the normal
    // braking collision check has brought the body right up to a wall.
    if (!heading_correction && straight_allowed && !straight_allowed(pose,end)) {
      // Keep this action/reference for a small recoverable heading error.
      // The alignment branch waits for translation to stop, checks rotation
      // through the normal collision projection, and checks the forward ray
      // again before resuming. If the nominal ray is also blocked it throws
      // there and requests a new path instead of retrying this indefinitely.
      needs_alignment_=true;rotation_started_=false;++clearance_realignments_;
      return {};
    }
    const double along=c*(end.x-pose.x)+s*(end.y-pose.y);
    if (along < -segment_capture) {
      throw std::runtime_error("Straight leg endpoint passed without capture; replan");
    }
    const double speed=std::min({speed_limit,max_linear,
      std::max(min_linear,approach_gain*std::max(0.0,along))});
    if (heading_correction) {
      // Small coupled v/w corrections; vy remains exactly zero. Collision
      // projection checks the resulting curved motion and complete braking.
      const double heading_error=angle(pose.yaw-reference_heading);
      const double body_y=-std::sin(heading_error)*ahead-std::cos(heading_error)*cross;
      const double curvature=2*body_y/(ahead*ahead+cross*cross);
      const double cruise=std::min(speed,std::max(min_linear,
        max_tracking_yaw/std::max(1e-6,std::abs(curvature))));
      double yaw=std::clamp(cruise*curvature,-max_tracking_yaw,max_tracking_yaw);
      if (std::abs(cross)<0.005 && std::abs(heading_error)<0.005) {yaw=0;}
      tracking_forward_=true;
      return {cruise,0,yaw};
    }
    return {speed,0,0};
  }

private:
  TrackingVelocity stoppedReferenceCommand(const TrackingPoint & pose,
    const TrackingVelocity & residual,double speed_limit,double dt,const TrackingVelocity & measured,
    const TrackingPoint & end,const TrackingPoint & start,double heading,double along,double cross,
    const std::function<bool(const TrackingPoint &,const TrackingPoint &)> & straight_allowed)
  {
    cross_track_error_=cross;
    const double error=angle(heading-pose.yaw);
    if (std::abs(cross)>max_segment_drift || along < -segment_capture) {
      throw std::runtime_error("Outside planned straight edge; stop and replan");
    }
    if (!needs_alignment_) {
      if (std::abs(cross)>realign_cross_threshold || std::abs(error)>realign_heading_threshold ||
        (straight_allowed && !straight_allowed(pose,end))) {
        needs_alignment_=true;alignment_active_=false;rotation_started_=false;
        ++clearance_realignments_;
        return {};
      }
      // Cruise is deliberately only body-forward. Reacquisition is a separate
      // stopped phase; it cannot turn or strafe while advancing along an edge.
      return {std::min({speed_limit,max_linear,
        std::max(min_linear,approach_gain*std::max(0.0,along))}),0,0};
    }
    if (!alignment_active_) {
      if (std::hypot(residual.x,residual.y)>0.02 || std::abs(residual.yaw)>0.02) {return {};}
      alignment_active_=true;strafe_braking_=false;alignment_settle_=0;
    }
    alignment_motion_=true;
    double position_tolerance=alignment_position_tolerance;
    double heading_tolerance=alignment_heading_tolerance;
    if (std::abs(cross)<=position_tolerance && std::abs(error)<=heading_tolerance &&
      straight_allowed && !straight_allowed(pose,end)) {
      // Inside the normal band, prefer a short centering pulse to changing
      // heading or discarding this path. Stop refining as soon as the actual
      // corridor clears; these smaller bands only choose a useful correction,
      // and are never mandatory acquisition tolerances in free space.
      if (std::abs(cross)>0.25*position_tolerance) {
        position_tolerance*=0.25;
      } else if (std::abs(error)>0.25*heading_tolerance) {
        heading_tolerance*=0.25;
      } else {
        throw std::runtime_error("Aligned straight corridor blocked; stop and replan");
      }
    }
    const bool position_ok=std::abs(cross)<=position_tolerance;
    const bool heading_ok=std::abs(error)<=heading_tolerance;
    if (position_ok && heading_ok) {
      if (std::hypot(residual.x,residual.y)>0.02 || std::abs(residual.yaw)>0.02) {
        alignment_settle_=0;return {};
      }
      alignment_settle_+=dt;
      if (alignment_settle_<0.20) {return {};}
      needs_alignment_=false;alignment_active_=false;alignment_motion_=false;
      rotation_started_=false;drive_heading_=heading;drive_origin_=start;
      return {};  // Next cycle starts straight motion after the settling check.
    }
    alignment_settle_=0;
    double vy=0;
    // Close to the desired heading, body-lateral motion returns to the
    // ORIGINAL planned line without introducing forward cruise. For a large
    // corner turn, first rotate until lateral correction has useful geometry.
    if (!position_ok && std::abs(error)<0.35) {
      const double cosine=std::cos(error);
      const double remaining=-cross/cosine;
      remaining_alignment_lateral_=remaining;
      if (strafe_braking_ && std::abs(residual.y)<=0.005) {strafe_braking_=false;}
      const double stop_distance=std::abs(measured.y)*
        (turn_stop_latency+std::abs(residual.y)/(2*lateral_deceleration));
      if (measured.y*remaining>0 && std::abs(remaining)<=stop_distance+0.5*position_tolerance) {
        strafe_braking_=true;
      }
      if (!strafe_braking_) {
        vy=std::copysign(std::min({speed_limit,alignment_lateral_speed,max_lateral}),remaining);
      }
    }
    const double wz=heading_ok ? 0.0 : turn(error,heading_tolerance);
    return {0,vy,wz};
  }

  double turn(double error,double tolerance)
  {
    const int sign = error >= 0.0 ? 1 : -1;
    // Near the antipode, +pi and -pi are the same physical heading. Keep the
    // chosen turn side across noise/replans until safely away from that seam.
    // Outside this 15-degree neighborhood, normal shortest-angle feedback
    // still permits correction after overshoot or a meaningful target change.
    if (std::abs(error) > M_PI-0.261799 && turn_direction_ != 0) {
      if (sign != turn_direction_) {error = turn_direction_*(2*M_PI-std::abs(error));}
    } else {
      turn_direction_ = sign;
    }
    remaining_turn_=error;
    // A useful motor command needs an earlier stop, not an arbitrarily tiny
    // tail command. Use physical yaw feedback and the smoother's remaining
    // ramp time. Once braking, wait for the vehicle to settle before deciding
    // whether another correction is needed; never reverse through its coast.
    const bool moving=std::abs(measured_turn_rate_)>0.02 || std::abs(residual_turn_rate_)>0.02;
    if (turn_braking_) {
      if (moving) {return 0;}
      turn_braking_=false;
    }
    const double stopping_angle=std::abs(measured_turn_rate_)*
      (turn_stop_latency+std::abs(residual_turn_rate_)/(2*turn_deceleration));
    if ((measured_turn_rate_*error<0 && std::abs(measured_turn_rate_)>0.02) ||
      (measured_turn_rate_*error>0 && std::abs(error)<=stopping_angle+0.5*tolerance)) {
      turn_braking_=true;return 0;
    }
    // Keep ordinary end-of-turn motion above the motor's weak-response band.
    // The controller may still try min_angular when a tighter stopping
    // corridor requires a slower, collision-checked rotation.
    return std::copysign(std::clamp(turn_gain*std::abs(error), preferred_min_angular, max_angular), error);
  }
  std::vector<TrackingPoint> path_;
  std::vector<double> arc_;
  std::vector<size_t> corners_;
  double progress_{-1.0}, corner_distance_{std::numeric_limits<double>::infinity()};
  bool tracking_forward_{false}, alignment_motion_{false}, alignment_active_{false}, strafe_braking_{false};
  double alignment_settle_{0};
  double remaining_alignment_lateral_{std::numeric_limits<double>::quiet_NaN()};
  double cross_track_error_{0};
  bool at_goal_{false}, rotation_started_{false}, lateral_plan_{false};
  int turn_direction_{0};
  bool goal_adjusting_{false}, leg_initialized_{false}, needs_alignment_{true};
  std::vector<size_t> leg_ends_;
  size_t leg_{0};
  double drive_heading_{0.0};
  double remaining_turn_{std::numeric_limits<double>::quiet_NaN()};
  double measured_turn_rate_{0},residual_turn_rate_{0};
  bool turn_braking_{false};
  size_t clearance_realignments_{0};
  TrackingPoint drive_origin_{0,0};
};
}  // namespace lightweight_fusion_bringup
