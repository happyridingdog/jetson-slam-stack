#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <vector>

#include "geometry_msgs/msg/point.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/cost_values.hpp"

namespace lightweight_fusion_bringup
{
using Velocity = std::array<double, 3>;

struct CollisionProjection
{
  // Match the velocity smoother's axis limits. The hold covers command and
  // costmap latency; the final phase checks the complete braking trajectory.
  double reaction_time{0.65};
  double horizon{1.0};
  Velocity acceleration{0.65, 0.20, 0.80};
  Velocity deceleration{0.75, 0.20, 0.90};

  bool valid() const
  {
    if (!std::isfinite(reaction_time) || reaction_time < 0.0 ||
      !std::isfinite(horizon) || horizon <= 0.0) {return false;}
    for (size_t i = 0; i < 3; ++i) {
      if (!std::isfinite(acceleration[i]) || acceleration[i] <= 0.0 ||
        !std::isfinite(deceleration[i]) || deceleration[i] <= 0.0) {return false;}
    }
    return true;
  }

  bool isFree(
    nav2_costmap_2d::Costmap2D & map,
    const std::vector<geometry_msgs::msg::Point> & footprint,
    double x, double y, double yaw, Velocity current, const Velocity & target,
    double remaining_turn=std::numeric_limits<double>::quiet_NaN(),
    double remaining_lateral=std::numeric_limits<double>::quiet_NaN()) const
  {
    if (!valid() || footprint.size() < 3 || !std::isfinite(x) ||
      !std::isfinite(y) || !std::isfinite(yaw)) {return false;}
    double radius = 0.0;
    for (const auto & p : footprint) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y)) {return false;}
      radius = std::max(radius, std::hypot(p.x, p.y));
    }
    if (radius <= 0.0) {return false;}
    Velocity bound;
    for (size_t i = 0; i < 3; ++i) {
      if (!std::isfinite(current[i]) || !std::isfinite(target[i])) {return false;}
      bound[i] = std::max(std::abs(current[i]), std::abs(target[i]));
    }
    std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(*map.getMutex());
    const double resolution = map.getResolution();
    if (!std::isfinite(resolution) || resolution <= 0.0) {return false;}
    // Include corner motion during rotation, so a thin obstacle cannot be
    // skipped by a fast translation or by an in-place turn.
    const double speed_bound = std::hypot(bound[0], bound[1]) + radius * bound[2];
    const double step = std::min(0.05, 0.5 * resolution / std::max(speed_bound, 1e-6));
    const double brake_time = std::max({bound[0] / deceleration[0],
      bound[1] / deceleration[1], bound[2] / deceleration[2]});
    const double duration = reaction_time + horizon + brake_time;
    if (duration / step > 4096.0) {return false;}
    auto footprint_free = [&]() {
        std::vector<nav2_costmap_2d::MapLocation> polygon;
        const double c = std::cos(yaw), s = std::sin(yaw);
        for (const auto & p : footprint) {
          nav2_costmap_2d::MapLocation cell;
          if (!map.worldToMap(x + c * p.x - s * p.y, y + s * p.x + c * p.y,
            cell.x, cell.y)) {return false;}
          polygon.push_back(cell);
        }
        // Check the filled footprint, not just its boundary: a small obstacle
        // wholly under the robot must also be detected.
        std::vector<nav2_costmap_2d::MapLocation> cells;
        map.convexFillCells(polygon, cells);
        if (cells.empty()) {return false;}
        for (const auto & cell : cells) {
          if (map.getCost(cell.x, cell.y) >= nav2_costmap_2d::LETHAL_OBSTACLE) {
            return false;  // Includes NO_INFORMATION; outside-map was rejected above.
          }
        }
        return true;
      };
    if (!footprint_free()) {return false;}
    const double initial_yaw=yaw,initial_x=x,initial_y=y;
    const bool bounded_turn=std::isfinite(remaining_turn) &&
      target[0]==0 && target[2]*remaining_turn>0;
    const bool bounded_lateral=std::isfinite(remaining_lateral) &&
      target[0]==0 && target[1]*remaining_lateral>0;
    bool lateral_braking=false;
    double elapsed = 0.0;
    while (elapsed < duration) {
      double dt = std::min(step, duration - elapsed);
      // Land exactly on phase boundaries to avoid extending the hold or
      // shortening the braking phase through time discretization.
      if (elapsed < reaction_time) {dt = std::min(dt, reaction_time - elapsed);}
      else if (elapsed < reaction_time + horizon) {
        dt = std::min(dt, reaction_time + horizon - elapsed);
      }
      Velocity next = current;
      if (elapsed >= reaction_time) {
        Velocity desired = elapsed < reaction_time + horizon ? target : Velocity{};
        // Allow stop-command latency after reaching the requested angle, then
        // check the complete braking tail. The initial hold also preserves
        // any residual motion before this command takes effect.
        if (bounded_turn && std::copysign(1.0,remaining_turn)*(yaw-initial_yaw)>=
          std::abs(remaining_turn)+bound[2]*reaction_time) {desired[2]=0;}
        if (bounded_lateral) {
          const double travelled=-std::sin(initial_yaw)*(x-initial_x)+std::cos(initial_yaw)*(y-initial_y);
          const double stopping=current[1]*current[1]/(2*deceleration[1]);
          if (std::copysign(1.0,remaining_lateral)*travelled>=
            std::max(0.0,std::abs(remaining_lateral)-stopping)) {lateral_braking=true;}
          if (lateral_braking) {desired[1]=0;}
        }
        for (size_t i = 0; i < 3; ++i) {
          // Brake to zero before reversing an axis.
          const double goal = current[i] * desired[i] < 0.0 ? 0.0 : desired[i];
          const double rate = std::abs(goal) > std::abs(current[i]) ?
            acceleration[i] : deceleration[i];
          next[i] += std::clamp(goal - current[i], -rate * dt, rate * dt);
        }
      }
      const double vx = (current[0] + next[0]) * 0.5;
      const double vy = (current[1] + next[1]) * 0.5;
      const double wz = (current[2] + next[2]) * 0.5;
      const double mid_yaw = yaw + wz * dt * 0.5;
      x += (std::cos(mid_yaw) * vx - std::sin(mid_yaw) * vy) * dt;
      y += (std::sin(mid_yaw) * vx + std::cos(mid_yaw) * vy) * dt;
      yaw += wz * dt;
      current = next;
      elapsed += dt;
      if (!footprint_free()) {return false;}
    }
    return true;
  }
};
}  // namespace lightweight_fusion_bringup
