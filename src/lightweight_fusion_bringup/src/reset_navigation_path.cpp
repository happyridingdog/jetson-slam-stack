#include "behaviortree_cpp_v3/bt_factory.h"
#include "nav_msgs/msg/path.hpp"

// A failed controller must trigger planning even when a centre-line-only
// path validity check would still accept the old route.
class ResetNavigationPath : public BT::SyncActionNode
{
public:
  ResetNavigationPath(const std::string & name, const BT::NodeConfiguration & config)
  : BT::SyncActionNode(name, config) {}
  static BT::PortsList providedPorts()
  {return {BT::OutputPort<nav_msgs::msg::Path>("path")};}
  BT::NodeStatus tick() override
  {
    setOutput("path", nav_msgs::msg::Path{});
    return BT::NodeStatus::SUCCESS;
  }
};

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<ResetNavigationPath>("ResetNavigationPath");
}
