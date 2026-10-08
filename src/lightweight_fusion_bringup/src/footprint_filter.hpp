#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// Test in the vehicle base frame, before voxelization, keyframe caching or
// transformation to the world. Reject the whole vertical footprint column.
struct FootprintFilter
{
  std::vector<double> polygon{-0.30, -0.225, 0.30, -0.225, 0.30, 0.225, -0.30, 0.225};
  std::vector<double> base_from_sensor{0.21, 0.0, 0.0};  // x, y, yaw
  void validate() const
  {
    if (polygon.size() < 6 || polygon.size() % 2 || base_from_sensor.size() != 3) {
      throw std::invalid_argument("Invalid footprint or base_from_sensor [x,y,yaw]");
    }
    for (double v : polygon) {if (!std::isfinite(v)) {throw std::invalid_argument("Non-finite footprint");}}
    for (double v : base_from_sensor) {if (!std::isfinite(v)) {throw std::invalid_argument("Non-finite extrinsic");}}
  }
  bool contains(double sx, double sy) const
  {
    const double c = std::cos(base_from_sensor[2]), s = std::sin(base_from_sensor[2]);
    const double x = c * sx - s * sy + base_from_sensor[0];
    const double y = s * sx + c * sy + base_from_sensor[1];
    bool inside = false;
    const size_t n = polygon.size() / 2;
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
      const double ax = polygon[2*j], ay = polygon[2*j+1];
      const double bx = polygon[2*i], by = polygon[2*i+1];
      const double cross = (x-ax)*(by-ay)-(y-ay)*(bx-ax);
      if (std::abs(cross) < 1e-8 && x >= std::min(ax,bx)-1e-8 &&
        x <= std::max(ax,bx)+1e-8 && y >= std::min(ay,by)-1e-8 && y <= std::max(ay,by)+1e-8) {return true;}
      if ((ay > y) != (by > y) && x < ax + (y-ay)*(bx-ax)/(by-ay)) {inside = !inside;}
    }
    return inside;
  }
};
