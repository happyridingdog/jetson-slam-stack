#ifndef LIGHTWEIGHT_FUSION_PATH_RESAMPLING_HPP_
#define LIGHTWEIGHT_FUSION_PATH_RESAMPLING_HPP_
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "nav_msgs/msg/path.hpp"

namespace lightweight_fusion_bringup
{
// Keep every original vertex (including a reversal or in-place rotation),
// endpoint and orientation. Only add points on the original line segments.
inline nav_msgs::msg::Path resamplePath(
  const nav_msgs::msg::Path & input, double max_spacing = 0.05, int min_intervals = 32)
{
  if (!std::isfinite(max_spacing) || max_spacing <= 0 || min_intervals < 12 || min_intervals > 4096) {
    throw std::invalid_argument("Invalid path resampling limits");
  }
  double length = 0;
  for (std::size_t i = 0; i < input.poses.size(); ++i) {
    const auto & p = input.poses[i].pose.position;
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
      throw std::invalid_argument("Path contains non-finite positions");
    }
    if (i) {
      const auto & prev = input.poses[i-1].pose.position;
      length += std::hypot(p.x-prev.x, p.y-prev.y);
    }
  }
  if (input.poses.size() < 2 || length < 1e-6) {return input;}
  const double spacing = std::min(max_spacing, length/min_intervals);
  if (length/spacing + input.poses.size() > 10000) {
    throw std::invalid_argument("Path exceeds bounded resampling size");
  }
  auto output = input;
  output.poses.clear();
  output.poses.push_back(input.poses.front());
  for (std::size_t i = 1; i < input.poses.size(); ++i) {
    const auto & a = input.poses[i-1].pose.position;
    const auto & b = input.poses[i].pose.position;
    const int count = std::max(1, static_cast<int>(std::ceil(std::hypot(b.x-a.x, b.y-a.y)/spacing)));
    for (int j = 1; j < count; ++j) {
      auto pose = input.poses[i-1];
      const double t = static_cast<double>(j)/count;
      pose.pose.position.x = a.x + t*(b.x-a.x);
      pose.pose.position.y = a.y + t*(b.y-a.y);
      pose.pose.position.z = a.z + t*(b.z-a.z);
      output.poses.push_back(pose);
    }
    output.poses.push_back(input.poses[i]);
  }
  return output;
}
}  // namespace lightweight_fusion_bringup
#endif
