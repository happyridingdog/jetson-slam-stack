#pragma once

#include "polyline_shortcut.hpp"
#include "nav2_costmap_2d/footprint.hpp"

namespace lightweight_fusion_bringup
{
// A private search preference, not a change to obstacle inflation or collision
// semantics. Extending the soft gradient lets the search see a corridor's
// centre even when the public inflation band ends before that centre.
inline nav2_costmap_2d::Costmap2D clearanceSearchMap(
  const nav2_costmap_2d::Costmap2D & source, double preferred_distance, double body_radius)
{
  nav2_costmap_2d::Costmap2D result(source);
  const int width=source.getSizeInCellsX(),height=source.getSizeInCellsY();
  auto cells=result.getCharMap();
  for (int i=0;i<width*height;++i) {
    if (cells[i]>=252 && cells[i]<254) {cells[i]=251;}
  }
  struct Offset {int x,y;unsigned char cost;};
  std::vector<Offset> offsets;
  const int radius=static_cast<int>(std::ceil(preferred_distance/source.getResolution()));
  for (int y=-radius;y<=radius;++y) {
    for (int x=-radius;x<=radius;++x) {
      const double distance=std::hypot(x,y)*source.getResolution();
      if (distance>=preferred_distance) {continue;}
      const double ratio=std::clamp((preferred_distance-distance)/
        (preferred_distance-body_radius),0.0,1.0);
      offsets.push_back({x,y,static_cast<unsigned char>(std::ceil(251*ratio))});
    }
  }
  for (int y=0;y<height;++y) {
    for (int x=0;x<width;++x) {
      if (source.getCost(x,y)<nav2_costmap_2d::LETHAL_OBSTACLE) {continue;}
      for (const auto & offset:offsets) {
        const int nx=x+offset.x,ny=y+offset.y;
        if (nx>=0 && ny>=0 && nx<width && ny<height) {
          auto & cost=cells[ny*width+nx];cost=std::max(cost,offset.cost);
        }
      }
    }
  }
  return result;
}

inline PoseClearanceCheck trackingMarginCheck(nav2_costmap_2d::Costmap2D & map,
  const std::vector<geometry_msgs::msg::Point> & footprint,
  const geometry_msgs::msg::Pose & start, const geometry_msgs::msg::Pose & goal,
  double margin)
{
  auto padded=footprint;
  nav2_costmap_2d::padFootprint(padded,margin);
  const bool relax_start=!footprintPoseFree(map,padded,start.position.x,start.position.y,
      planarYaw(start.orientation));
  const bool relax_goal=!footprintPoseFree(map,padded,goal.position.x,goal.position.y,
      planarYaw(goal.orientation));
  return [&map,footprint,padded,start,goal,margin,relax_start,relax_goal](double x,double y,double yaw) {
      // Existing poses next to an obstacle must be able to leave it. Only
      // relax the EXTRA margin near such endpoints; the full physical body
      // and its turn sweep remain checked everywhere by the caller.
      double scale=1.0;
      if (relax_start) {scale=std::min(scale,std::hypot(x-start.position.x,y-start.position.y)/0.6);}
      if (relax_goal) {scale=std::min(scale,std::hypot(x-goal.position.x,y-goal.position.y)/0.6);}
      if (scale>=1.0) {return footprintPoseFree(map,padded,x,y,yaw);}
      auto local=footprint;
      nav2_costmap_2d::padFootprint(local,margin*scale);
      return footprintPoseFree(map,local,x,y,yaw);
    };
}
}  // namespace lightweight_fusion_bringup
