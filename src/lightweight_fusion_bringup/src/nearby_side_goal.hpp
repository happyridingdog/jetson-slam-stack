#pragma once
#include <cmath>

namespace lightweight_fusion_bringup
{
// A short side step, with at most 30 cm of fore/aft position correction.
// Express eligibility in the starting body frame, never in map axes.
inline bool nearbySideGoal(double dx, double dy, double yaw)
{
  const double forward=std::cos(yaw)*dx+std::sin(yaw)*dy;
  const double lateral=-std::sin(yaw)*dx+std::cos(yaw)*dy;
  return std::isfinite(forward) && std::isfinite(lateral) &&
    std::hypot(dx,dy)<=2.0+1e-9 && std::abs(forward)<=0.30+1e-9 &&
    std::abs(lateral)>std::abs(forward);
}
}  // namespace lightweight_fusion_bringup
