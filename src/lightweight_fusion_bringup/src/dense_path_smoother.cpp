#include "nav2_smoother/simple_smoother.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "path_resampling.hpp"

namespace lightweight_fusion_bringup
{
// Humble SimpleSmoother skips short directional segments. Theta* can return
// a valid but kinked 1-2 m path with too few points for it to do any work.
// Add geometric samples first, then retain Nav2's obstacle-aware smoothing
// and the smoother server's full-footprint collision validation.
class DensePathSmoother : public nav2_smoother::SimpleSmoother
{
public:
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::CostmapSubscriber> costmap,
    std::shared_ptr<nav2_costmap_2d::FootprintSubscriber> footprint) override
  {
    SimpleSmoother::configure(parent, name, tf, costmap, footprint);
    auto node = parent.lock();
    node->declare_parameter(name+".resample_spacing", 0.05);
    node->declare_parameter(name+".minimum_intervals", 32);
    spacing_ = node->get_parameter(name+".resample_spacing").as_double();
    intervals_ = static_cast<int>(node->get_parameter(name+".minimum_intervals").as_int());
    resamplePath(nav_msgs::msg::Path{}, spacing_, intervals_);  // validate configuration
  }

  bool smooth(nav_msgs::msg::Path & path, const rclcpp::Duration & max_time) override
  {
    auto candidate = resamplePath(path, spacing_, intervals_);
    const bool completed = SimpleSmoother::smooth(candidate, max_time);
    path = std::move(candidate);
    return completed;
  }
private:
  double spacing_{0.05};
  int intervals_{32};
};
}
PLUGINLIB_EXPORT_CLASS(lightweight_fusion_bringup::DensePathSmoother, nav2_core::Smoother)
