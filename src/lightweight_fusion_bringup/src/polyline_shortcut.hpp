#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>
#include "nav_msgs/msg/path.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/cost_values.hpp"

namespace lightweight_fusion_bringup
{
using ShortcutDeadline = std::chrono::steady_clock::time_point;
using PoseClearanceCheck = std::function<bool(double,double,double)>;

struct ShortcutPreference
{
  const nav2_costmap_2d::Costmap2D * costs{nullptr};
  double distance_weight{1.0}, traversal_weight{5.0};
  PoseClearanceCheck pose_allowed;
};

// Integrate the same cost scale as Theta*, in metres. A simplification must
// not erase the search's reason for choosing a route away from a wall.
inline double straightTraversalCost(const geometry_msgs::msg::Point & a,
  const geometry_msgs::msg::Point & b, const ShortcutPreference & preference)
{
  const double length=std::hypot(b.x-a.x,b.y-a.y);
  if (!preference.costs) {return length;}
  const auto & map=*preference.costs;
  const int steps=std::max(1,static_cast<int>(std::ceil(length/(map.getResolution()*0.5))));
  double total=0;
  for (int i=0;i<steps;++i) {
    const double t=(i+0.5)/steps;
    unsigned int mx,my;
    if (!map.worldToMap(a.x+t*(b.x-a.x),a.y+t*(b.y-a.y),mx,my) ||
      map.getCost(mx,my)>=nav2_costmap_2d::LETHAL_OBSTACLE) {
      return std::numeric_limits<double>::infinity();
    }
    const double cost=(26.0+0.9*map.getCost(mx,my))/252.0;
    total+=preference.distance_weight+preference.traversal_weight*cost*cost;
  }
  return length*total/steps;
}

inline double polylineTraversalCost(const nav_msgs::msg::Path & path,
  const ShortcutPreference & preference)
{
  if (path.poses.empty()) {return std::numeric_limits<double>::infinity();}
  double total=0;
  for (size_t i=1;i<path.poses.size();++i) {
    total+=straightTraversalCost(path.poses[i-1].pose.position,path.poses[i].pose.position,preference);
  }
  return total;
}

inline double planarYaw(const geometry_msgs::msg::Quaternion & q)
{
  const double norm=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
  if (!std::isfinite(norm) || norm<1e-12) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::atan2(2*(q.w*q.z+q.x*q.y),q.w*q.w+q.x*q.x-q.y*q.y-q.z*q.z);
}

inline bool footprintPoseFree(nav2_costmap_2d::Costmap2D & map,
  const std::vector<geometry_msgs::msg::Point> & footprint,
  double x, double y, double yaw)
{
  if (footprint.size()<3 || !std::isfinite(x) || !std::isfinite(y) ||
    !std::isfinite(yaw)) {return false;}
  const double c=std::cos(yaw),s=std::sin(yaw);
  std::vector<nav2_costmap_2d::MapLocation> polygon,cells;
  for (const auto & p:footprint) {
    nav2_costmap_2d::MapLocation cell;
    if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
      !map.worldToMap(x+c*p.x-s*p.y,y+s*p.x+c*p.y,cell.x,cell.y)) {return false;}
    polygon.push_back(cell);
  }
  map.convexFillCells(polygon,cells);
  if (cells.empty()) {return false;}
  for (const auto & cell:cells) {
    if (map.getCost(cell.x,cell.y)>=nav2_costmap_2d::LETHAL_OBSTACLE) {return false;}
  }
  return true;
}

inline bool rotationCorridorFree(nav2_costmap_2d::Costmap2D & map,
  const std::vector<geometry_msgs::msg::Point> & footprint,
  const geometry_msgs::msg::Point & at, double from, double to,
  ShortcutDeadline deadline, const PoseClearanceCheck & pose_allowed={})
{
  if (!std::isfinite(from) || !std::isfinite(to) || map.getResolution()<=0) {return false;}
  double radius=0;
  for (const auto & p:footprint) {radius=std::max(radius,std::hypot(p.x,p.y));}
  if (!std::isfinite(radius) || radius<=0) {return false;}
  const double delta=std::atan2(std::sin(to-from),std::cos(to-from));
  const double step=std::min(0.05,0.5*map.getResolution()/radius);
  if (std::abs(delta)/step>20000) {return false;}
  const int steps=std::max(1,static_cast<int>(std::ceil(std::abs(delta)/step)));
  for (int i=0;i<=steps;++i) {
    if (std::chrono::steady_clock::now()>=deadline ||
      !footprintPoseFree(map,footprint,at.x,at.y,from+delta*i/steps) ||
      (pose_allowed && !pose_allowed(at.x,at.y,from+delta*i/steps))) {return false;}
  }
  return true;
}

// Caller holds the costmap mutex. Check the whole filled footprint along the
// proposed straight leg; centre-line visibility alone can clip a vehicle side.
inline bool translationCorridorFree(
  nav2_costmap_2d::Costmap2D & map,
  const std::vector<geometry_msgs::msg::Point> & footprint,
  const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b, double yaw,
  ShortcutDeadline deadline, const PoseClearanceCheck & pose_allowed={})
{
  const double dx=b.x-a.x, dy=b.y-a.y, length=std::hypot(dx,dy);
  if (footprint.size()<3 || !std::isfinite(length) || map.getResolution()<=0 ||
    length/map.getResolution()>20000) {return false;}
  if (!std::isfinite(yaw)) {return false;}
  const int steps=std::max(1,static_cast<int>(std::ceil(length/(0.5*map.getResolution()))));
  for (int i=0;i<=steps;++i) {
    if (std::chrono::steady_clock::now()>=deadline) {return false;}
    const double t=static_cast<double>(i)/steps, x=a.x+t*dx, y=a.y+t*dy;
    if (!footprintPoseFree(map,footprint,x,y,yaw)) {return false;}
    if (pose_allowed && !pose_allowed(x,y,yaw)) {return false;}
  }
  return true;
}

// Forward legs use their travel bearing; side steps retain the body heading.
inline bool straightCorridorFree(
  nav2_costmap_2d::Costmap2D & map,
  const std::vector<geometry_msgs::msg::Point> & footprint,
  const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b,
  ShortcutDeadline deadline, const PoseClearanceCheck & pose_allowed={})
{
  return translationCorridorFree(map,footprint,a,b,std::atan2(b.y-a.y,b.x-a.x),
    deadline,pose_allowed);
}

inline nav_msgs::msg::Path shortcutPolyline(
  const nav_msgs::msg::Path & input, nav2_costmap_2d::Costmap2D & map,
  const std::vector<geometry_msgs::msg::Point> & footprint, ShortcutDeadline deadline,
  const ShortcutPreference * preference=nullptr)
{
  auto result=input;result.poses.clear();
  const size_t n=input.poses.size();
  if (n==0 || n>1024) {return result;}
  const double start_yaw=planarYaw(input.poses.front().pose.orientation);
  const double goal_yaw=planarYaw(input.poses.back().pose.orientation);
  if (!std::isfinite(start_yaw) || !std::isfinite(goal_yaw)) {return result;}
  const PoseClearanceCheck pose_allowed=preference ? preference->pose_allowed : PoseClearanceCheck{};
  std::vector<double> prefix(n,0);
  if (preference && preference->costs) {
    for (size_t i=1;i<n;++i) {
      prefix[i]=prefix[i-1]+straightTraversalCost(input.poses[i-1].pose.position,
        input.poses[i].pose.position,*preference);
    }
  }
  // Heading is part of feasibility: a rectangle can fit each straight leg
  // but hit a wall while turning between them. Backtrack if a long shortcut
  // strands the next turn, rather than returning an unchecked adjacent edge.
  std::vector<unsigned char> failed((n+1)*n,0),edges(n*n,0);
  std::vector<size_t> route{0};
  auto bearing=[&](size_t a,size_t b) {
      const auto & p=input.poses[a].pose.position;
      const auto & q=input.poses[b].pose.position;
      return std::atan2(q.y-p.y,q.x-p.x);
    };
  std::function<bool(size_t,size_t)> search=[&](size_t previous,size_t current) {
      if (std::chrono::steady_clock::now()>=deadline || failed[previous*n+current]) {
        return false;
      }
      const double incoming=previous==n ? start_yaw : bearing(previous,current);
      const auto & at=input.poses[current].pose.position;
      if (current+1==n) {
        return rotationCorridorFree(map,footprint,at,incoming,goal_yaw,deadline,pose_allowed);
      }
      for (size_t next=n-1;next>current;--next) {
        if (std::chrono::steady_clock::now()>=deadline) {return false;}
        const auto & end=input.poses[next].pose.position;
        const double length=std::hypot(end.x-at.x,end.y-at.y);
        if (length<1e-9) {
          if (next==n-1 && rotationCorridorFree(map,footprint,at,incoming,goal_yaw,deadline,pose_allowed)) {
            route.push_back(next);return true;
          }
          continue;
        }
        if (preference && preference->costs && next>current+1) {
          // Exact endpoints can require a gentler entry/exit than the grid
          // search's first diagonal. Permit a bounded one-cell cost increase
          // there so the full rectangle can acquire the centred leg safely.
          const double endpoint_allowance=(current==0 || next==n-1) ?
            preference->traversal_weight*map.getResolution() : 0.0;
          if (straightTraversalCost(at,end,*preference)>
            1.02*(prefix[next]-prefix[current])+0.01+endpoint_allowance) {continue;}
        }
        if (!rotationCorridorFree(map,footprint,at,incoming,bearing(current,next),deadline,pose_allowed)) {
          continue;
        }
        auto & edge=edges[current*n+next];
        if (edge==0) {edge=straightCorridorFree(map,footprint,at,end,deadline,pose_allowed) ? 1 : 2;}
        if (edge!=1) {continue;}
        route.push_back(next);
        if (search(current,next)) {return true;}
        route.pop_back();
      }
      failed[previous*n+current]=1;
      return false;
    };
  if (!search(n,0)) {return result;}
  for (size_t index:route) {result.poses.push_back(input.poses[index]);}
  return result;
}

// Nav2 validates rectangular footprints using each path pose's orientation.
// Theta* supplies identity orientations; leaving them on a non-horizontal
// leg can make a valid narrow passage look blocked and trigger replanning.
inline nav_msgs::msg::Path orientPolyline(nav_msgs::msg::Path path)
{
  for (size_t i=0;i+1<path.poses.size();++i) {
    const auto & a=path.poses[i].pose.position;
    size_t next=i+1;
    while (next<path.poses.size() && std::hypot(
      path.poses[next].pose.position.x-a.x,path.poses[next].pose.position.y-a.y)<1e-9) {++next;}
    if (next==path.poses.size()) {continue;}
    const auto & b=path.poses[next].pose.position;
    const double yaw=std::atan2(b.y-a.y,b.x-a.x);
    auto & q=path.poses[i].pose.orientation;
    q.x=q.y=0;q.z=std::sin(yaw/2);q.w=std::cos(yaw/2);
  }
  return path; // Preserve the requested final goal orientation.
}
}  // namespace lightweight_fusion_bringup
