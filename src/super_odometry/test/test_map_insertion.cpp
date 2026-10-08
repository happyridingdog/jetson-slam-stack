#include "super_odometry/LidarProcess/LidarSlam.h"
#include <stdexcept>
#include <iostream>

static void require(bool pass, const char* reason) {
  if (!pass) throw std::runtime_error(reason);
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto slam = std::make_unique<super_odometry::LidarSLAM>();
  slam->node_ = rclcpp::Node::make_shared("map_insertion_regression");
  SAVE_PLY = false;
  slam->OptSet = {};
  slam->OptSet.velocity_failure_threshold = 10.0;
  slam->OptSet.map_accumulation_max_linear_speed = 1.0;
  slam->OptSet.map_accumulation_max_angular_speed = 1.5;
  slam->OptSet.map_accumulation_max_registration_rmse = 0.08;
  slam->OptSet.map_accumulation_min_correspondences = 500;
  slam->last_registration_rmse_ = 0.02;
  slam->last_registration_correspondences_ = 1000;
  pcl::PointXYZI point;
  point.x = 2.0; point.y = 0.0; point.z = 0.0; point.intensity = 1.0;
  slam->PlanarsPoints->push_back(point);
  slam->initializeMapping(1.0);
  double time = 1.0;
  auto step = [&](double x, double yaw) {
    slam->T_w_lidar.pos.x() = x;
    slam->T_w_lidar.rot = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    slam->T_w_initial_guess = slam->T_w_lidar;
    TicToc timer;
    super_odometry_msgs::msg::OptimizationStats stats;
    time += 0.1;
    slam->performPostOptimizationProcessing(time, timer, stats);
  };
  // 5 cm/s at 10 Hz: every individual step is below the insertion threshold.
  for (int i = 1; i <= 50; ++i) step(i * 0.005, 0.0);
  require(slam->WorldPlanarsPoints->front().x > 2.22f,
    "slow travel failed to update actual map features");
  const auto before_bad_scan = slam->last_map_insert_pose_;
  slam->last_registration_rmse_ = 0.5;
  for (int i = 51; i <= 60; ++i) step(i * 0.005, 0.0);
  require((slam->last_map_insert_pose_.pos - before_bad_scan.pos).norm() == 0.0,
    "bad registration advanced the map insertion reference");
  slam->last_registration_rmse_ = 0.02;
  step(0.305, 0.0);
  require(slam->WorldPlanarsPoints->front().x > 2.30f,
    "map did not recover after rejected scans");
  // Slow angular steps must accumulate as well.
  for (int i = 1; i <= 30; ++i) step(0.305, i * 0.001);
  require(slam->WorldPlanarsPoints->front().y > 0.045f,
    "slow rotation failed to update actual map features");
  const auto settled = slam->last_map_insert_pose_;
  for (int i = 0; i < 30; ++i) step(0.305, 0.030);
  require(slam->last_map_insert_pose_.rot.angularDistance(settled.rot) < 1e-12,
    "stationary scans unnecessarily updated the map");
  std::cout << "PASS slow translation, slow rotation, quality rejection and stationary map gates\n";
  slam.reset();
  rclcpp::shutdown();
}
