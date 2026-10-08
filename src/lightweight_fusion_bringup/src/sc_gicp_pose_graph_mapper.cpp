#include <algorithm>
#include "footprint_filter.hpp"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <fast_gicp/gicp/fast_gicp.hpp>
#ifdef USE_FAST_VGICP_CUDA
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace
{
using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using gtsam::symbol_shorthand::X;

constexpr double kPi = 3.14159265358979323846;

struct ScanContextDescriptor
{
  Eigen::MatrixXf descriptor;
  Eigen::VectorXf ring_key;
  int valid_points{0};
};

struct ScanContextMatch
{
  int keyframe_index{-1};
  double score{std::numeric_limits<double>::infinity()};
  double second_score{std::numeric_limits<double>::infinity()};
  int yaw_shift{0};
};

struct LoopEdge
{
  int from{0};
  int to{0};
  Eigen::Isometry3d measurement_from_to{Eigen::Isometry3d::Identity()};
  double fitness{0.0};
  double score{0.0};
};

struct Keyframe
{
  int id{0};
  rclcpp::Time stamp;
  Eigen::Isometry3d odom_pose{Eigen::Isometry3d::Identity()};
  gtsam::Pose3 optimized_pose;
  CloudT::Ptr cloud{new CloudT()};
  ScanContextDescriptor scan_context;
};

Eigen::Matrix4f toMatrix4f(const Eigen::Isometry3d & pose)
{
  return pose.matrix().cast<float>();
}

gtsam::Pose3 toPose3(const Eigen::Isometry3d & pose)
{
  return gtsam::Pose3(gtsam::Rot3(pose.rotation()), gtsam::Point3(pose.translation()));
}

Eigen::Isometry3d fromPose3(const gtsam::Pose3 & pose)
{
  Eigen::Isometry3d out = Eigen::Isometry3d::Identity();
  out.matrix() = pose.matrix();
  return out;
}

double rotationAngle(const Eigen::Matrix3d & rotation)
{
  const double trace = rotation.trace();
  const double value = std::clamp((trace - 1.0) * 0.5, -1.0, 1.0);
  return std::acos(value);
}

std::string timestampName()
{
  std::time_t now = std::time(nullptr);
  std::tm tm{};
  localtime_r(&now, &tm);
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm);
  return std::string(buffer);
}

}  // namespace

class ScGicpPoseGraphMapper : public rclcpp::Node
{
public:
  ScGicpPoseGraphMapper()
  : Node("sc_gicp_pose_graph_mapper"),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    declareParameters();
    readParameters();
#ifndef USE_FAST_VGICP_CUDA
    if (registration_use_cuda_ || gicp_backend_ == "cuda") {
      throw std::runtime_error("CUDA mapper requested but USE_FAST_VGICP_CUDA is disabled");
    }
#endif
    const auto seed_file = declare_parameter<std::string>("seed_keyframes_file", "");
    if (seed_file.empty()) {
      cleanupKeyframeCache();
    } else {
      // Explicit recovery within the same odometry frame only. A normal new
      // mapping session must start without a seed file.
      loadSeedKeyframes(seed_file);
    }

    auto cloud_qos = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort();
    auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_cloud_topic_, cloud_qos,
      std::bind(&ScGicpPoseGraphMapper::cloudCallback, this, std::placeholders::_1));
    map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(map_cloud_topic_, map_qos);
    // Keep the complete map on map_cloud_topic_; bounded chunk topics are used
    // by display clients so a large map is uploaded in manageable buffers.
    for (int i = 0; i < map_shard_count_; ++i) {
      map_shard_pubs_.push_back(create_publisher<sensor_msgs::msg::PointCloud2>(
        map_cloud_topic_ + "/chunk_" + std::to_string(i), map_qos));
    }
    current_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(current_cloud_topic_, cloud_qos);
    optimized_path_pub_ = create_publisher<nav_msgs::msg::Path>(
      optimized_path_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
    status_pub_ = create_publisher<std_msgs::msg::String>(status_topic_, 10);
    save_service_ = create_service<std_srvs::srv::Trigger>(
      "~/save_map", std::bind(&ScGicpPoseGraphMapper::saveMapService, this,
      std::placeholders::_1, std::placeholders::_2));

    map_timer_ = create_wall_timer(
      std::chrono::duration<double>(std::max(0.2, 1.0 / std::max(0.1, map_publish_hz_))),
      std::bind(&ScGicpPoseGraphMapper::publishMapTimer, this));
    status_timer_ = create_wall_timer(
      std::chrono::seconds(2), std::bind(&ScGicpPoseGraphMapper::publishStatus, this));
    if (!keyframes_.empty()) {publishOptimizedPath();}

    RCLCPP_INFO(
      get_logger(),
      "SC+FastGICP pose graph mapper input=%s map=%s frame=%s",
      input_cloud_topic_.c_str(), map_cloud_topic_.c_str(), output_frame_.c_str());
  }

  void saveMapOnShutdown()
  {
    if (!save_map_on_shutdown_) {
      return;
    }
    saveMapSnapshot("shutdown");
  }

private:
  void loadSeedKeyframes(const std::string & path)
  {
    // Snapshot format: SC_GICP_SEED_V1 frame count, then one row per
    // keyframe: x y z qx qy qz qw relative_cloud_path. These are fixed,
    // already optimized poses, not a restoration of historical loop edges.
    std::ifstream input(path);
    std::string version, frame;
    std::size_t count = 0;
    if (!(input >> version >> frame >> count) || version != "SC_GICP_SEED_V1" ||
      frame != output_frame_ || count == 0 || count > 100000) {
      throw std::runtime_error("invalid seed keyframes header/frame: " + path);
    }
    std::vector<Keyframe> loaded;
    loaded.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      Eigen::Vector3d p;
      double qx, qy, qz, qw;
      std::string cloud_path;
      if (!(input >> p.x() >> p.y() >> p.z() >> qx >> qy >> qz >> qw >> cloud_path)) {
        throw std::runtime_error("truncated seed keyframes: " + path);
      }
      Eigen::Quaterniond q(qw, qx, qy, qz);
      if (!p.allFinite() || !q.coeffs().allFinite() || std::abs(q.norm() - 1.0) > 1e-3) {
        throw std::runtime_error("invalid seed keyframe pose: " + path);
      }
      Keyframe keyframe;
      keyframe.id = static_cast<int>(i);
      keyframe.odom_pose.translation() = p;
      keyframe.odom_pose.linear() = q.normalized().toRotationMatrix();
      keyframe.optimized_pose = toPose3(keyframe.odom_pose);
      const auto cloud_file = std::filesystem::path(path).parent_path() / cloud_path;
      if (pcl::io::loadPLYFile(cloud_file.string(), *keyframe.cloud) < 0 ||
        keyframe.cloud->empty() ||
        !std::all_of(keyframe.cloud->begin(), keyframe.cloud->end(), [](const PointT & point) {
          return point.getVector3fMap().allFinite();
        })) {
        throw std::runtime_error("invalid seed cloud: " + cloud_file.string());
      }
      keyframe.scan_context = makeScanContext(keyframe.cloud);
      loaded.push_back(std::move(keyframe));
    }
    keyframes_ = std::move(loaded);
    for (const auto & keyframe : keyframes_) {
      optimized_poses_.push_back(keyframe.optimized_pose);
    }
    enforceCloudBudget();
    last_keyframe_pose_ = keyframes_.back().odom_pose;
    has_last_keyframe_pose_ = true;
    map_dirty_ = true;
    RCLCPP_INFO(get_logger(), "restored %zu seed keyframes in frame %s",
      keyframes_.size(), output_frame_.c_str());
  }

  void saveMapService(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (keyframes_.empty()) {
      response->success = false;
      response->message = "no keyframes available";
      return;
    }
    saveMapSnapshot("service");
    response->success = true;
    response->message = "pose graph map snapshot saved";
  }

  void declareParameters()
  {
    declare_parameter<std::string>("input_cloud_topic", "/body_points");
    declare_parameter<std::string>("map_cloud_topic", "/map_points_3d");
    declare_parameter<std::string>("current_cloud_topic", "/current_cloud_3d");
    declare_parameter<std::string>("status_topic", "~/status");
    declare_parameter<std::string>("output_frame", "map");
    declare_parameter<std::string>("source_frame_override", "");
    declare_parameter<double>("keyframe_voxel_size_m", 0.08);
    declare_parameter<double>("map_voxel_size_m", 0.10);
    declare_parameter<int>("max_input_points", 60000);
    declare_parameter<int>("max_keyframe_points", 50000);
    declare_parameter<int>("max_publish_points", 80000);
    declare_parameter<int>("map_shard_count", 8);
    declare_parameter<int>("map_shard_points", 50000);
    declare_parameter<int>("max_current_points", 24000);
    declare_parameter<double>("insert_hz", 8.0);
    declare_parameter<double>("map_publish_hz", 0.5);
    declare_parameter<double>("current_publish_hz", 5.0);
    declare_parameter<double>("min_range_m", 0.10);
    declare_parameter<std::vector<double>>("self_filter_footprint", self_filter_.polygon);
    declare_parameter<std::vector<double>>("self_filter_base_from_sensor", self_filter_.base_from_sensor);
    declare_parameter<double>("max_range_m", 80.0);
    declare_parameter<double>("local_radius_m", 18.0);
    declare_parameter<double>("min_keyframe_translation_m", 0.25);
    declare_parameter<double>("min_keyframe_rotation_deg", 8.0);
    declare_parameter<double>("max_keyframe_interval_s", 2.0);
    declare_parameter<double>("tf_timeout_s", 0.10);
    declare_parameter<bool>("tf_use_latest_if_stamp_unavailable", true);
    declare_parameter<double>("tf_latest_max_age_s", 0.80);
    declare_parameter<double>("max_keyframe_linear_speed_mps", 2.0);
    declare_parameter<double>("max_keyframe_angular_speed_rps", 3.0);
    declare_parameter<std::string>("optimized_path_topic", "/optimized_pose_graph_path");
    declare_parameter<double>("registration_check_hz", 1.0);
    declare_parameter<int>("registration_check_min_keyframes", 5);
    declare_parameter<int>("registration_submap_keyframes", 20);
    declare_parameter<int>("registration_max_points", 12000);
    declare_parameter<double>("registration_voxel_size_m", 0.20);
    declare_parameter<double>("registration_max_fitness_score", 0.35);
    declare_parameter<bool>("registration_accept_nonconverged_if_quality_ok", false);
    declare_parameter<double>("registration_nonconverged_fitness_score", 0.08);
    declare_parameter<double>("registration_max_translation_correction_m", 0.50);
    declare_parameter<double>("registration_max_rotation_correction_deg", 15.0);
    declare_parameter<bool>("registration_use_cuda", false);

    declare_parameter<int>("scan_context_num_rings", 20);
    declare_parameter<int>("scan_context_num_sectors", 60);
    declare_parameter<double>("scan_context_min_radius", 0.5);
    declare_parameter<double>("scan_context_max_radius", 45.0);
    declare_parameter<double>("scan_context_height_offset", 2.0);
    declare_parameter<double>("scan_context_max_height", 6.0);
    declare_parameter<int>("scan_context_min_points", 80);
    declare_parameter<int>("scan_context_exclude_recent", 30);
    declare_parameter<int>("scan_context_search_candidates", 96);
    declare_parameter<int>("scan_context_verify_candidates", 8);
    declare_parameter<double>("scan_context_score_thresh", 0.22);
    declare_parameter<double>("scan_context_score_margin", 0.03);

    declare_parameter<double>("loop_min_travel_distance_m", 8.0);
    declare_parameter<double>("loop_fitness_score_thresh", 0.45);
    declare_parameter<double>("loop_max_relative_translation_m", 8.0);
    declare_parameter<double>("loop_max_relative_z_m", 2.5);
    declare_parameter<double>("loop_max_relative_rotation_deg", 35.0);
    declare_parameter<int>("loop_submap_max_keyframes", 12);
    declare_parameter<double>("loop_submap_voxel_size_m", 0.18);
    declare_parameter<int>("gicp_num_threads", 4);
    declare_parameter<int>("gicp_correspondence_randomness", 20);
    declare_parameter<double>("gicp_max_correspondence_distance", 1.5);
    declare_parameter<int>("gicp_maximum_iterations", 64);
    declare_parameter<std::string>("gicp_backend", "auto");
    declare_parameter<std::string>("cuda_search_method", "GPU_BRUTEFORCE");
    declare_parameter<double>("gicp_transformation_epsilon", 0.01);
    declare_parameter<double>("gicp_euclidean_fitness_epsilon", 0.001);

    declare_parameter<int>("graph_optimize_period_keyframes", 5);
    declare_parameter<int>("graph_optimization_iterations", 32);
    declare_parameter<double>("prior_position_sigma", 0.05);
    declare_parameter<double>("prior_rotation_sigma", 0.03);
    declare_parameter<double>("odom_position_sigma", 0.08);
    declare_parameter<double>("odom_rotation_sigma", 0.04);
    declare_parameter<double>("loop_position_sigma", 0.15);
    declare_parameter<double>("loop_rotation_sigma", 0.08);

    declare_parameter<bool>("enable_keyframe_cloud_cache", true);
    declare_parameter<std::string>("keyframe_cloud_cache_dir", "/home/jszr/zsibot_maps/keyframe_cloud_cache");
    declare_parameter<int>("keyframe_cloud_cache_keep_recent", 240);
    declare_parameter<bool>("keyframe_cloud_cache_cleanup_on_start", true);
    declare_parameter<int>("max_keyframe_clouds_in_memory", 240);
    declare_parameter<int>("max_archived_map_points", 300000);

    declare_parameter<bool>("save_map_on_shutdown", true);
    declare_parameter<std::string>("save_map_path", "/home/jszr/zsibot_maps/managed_pointcloud_map_latest.ply");
    declare_parameter<bool>("save_timestamped_snapshots", true);
    declare_parameter<std::string>("shutdown_save_root", "/home/jszr/zsibot_maps");
    declare_parameter<std::string>("shutdown_save_name", "");
    declare_parameter<std::string>("latest_map_path", "/home/jszr/zsibot_maps/managed_pointcloud_map_latest.ply");
    declare_parameter<std::string>("latest_metadata_path", "/home/jszr/zsibot_maps/managed_pointcloud_map_latest.txt");
    declare_parameter<std::string>("latest_map_pcd_path", "/home/jetson/.jszr/map/pose_graph_map_latest.pcd");
  }

  void readParameters()
  {
    input_cloud_topic_ = get_parameter("input_cloud_topic").as_string();
    map_cloud_topic_ = get_parameter("map_cloud_topic").as_string();
    current_cloud_topic_ = get_parameter("current_cloud_topic").as_string();
    status_topic_ = get_parameter("status_topic").as_string();
    output_frame_ = get_parameter("output_frame").as_string();
    source_frame_override_ = get_parameter("source_frame_override").as_string();
    keyframe_voxel_size_m_ = std::max(0.01, get_parameter("keyframe_voxel_size_m").as_double());
    map_voxel_size_m_ = std::max(0.01, get_parameter("map_voxel_size_m").as_double());
    max_input_points_ = std::max(0, static_cast<int>(get_parameter("max_input_points").as_int()));
    max_keyframe_points_ = std::max(0, static_cast<int>(get_parameter("max_keyframe_points").as_int()));
    max_publish_points_ = std::max(0, static_cast<int>(get_parameter("max_publish_points").as_int()));
    map_shard_count_ = std::clamp(static_cast<int>(get_parameter("map_shard_count").as_int()), 1, 32);
    map_shard_points_ = std::max(1000, static_cast<int>(get_parameter("map_shard_points").as_int()));
    max_current_points_ = std::max(0, static_cast<int>(get_parameter("max_current_points").as_int()));
    insert_hz_ = get_parameter("insert_hz").as_double();
    map_publish_hz_ = get_parameter("map_publish_hz").as_double();
    current_publish_hz_ = get_parameter("current_publish_hz").as_double();
    min_range_m_ = std::max(0.0, get_parameter("min_range_m").as_double());
    self_filter_.polygon = get_parameter("self_filter_footprint").as_double_array();
    self_filter_.base_from_sensor = get_parameter("self_filter_base_from_sensor").as_double_array();
    self_filter_.validate();
    max_range_m_ = std::max(0.0, get_parameter("max_range_m").as_double());
    local_radius_m_ = get_parameter("local_radius_m").as_double();
    min_keyframe_translation_m_ = std::max(0.0, get_parameter("min_keyframe_translation_m").as_double());
    min_keyframe_rotation_rad_ = get_parameter("min_keyframe_rotation_deg").as_double() * kPi / 180.0;
    max_keyframe_interval_s_ = std::max(0.0, get_parameter("max_keyframe_interval_s").as_double());
    tf_timeout_s_ = std::max(0.01, get_parameter("tf_timeout_s").as_double());
    tf_use_latest_if_stamp_unavailable_ = get_parameter(
      "tf_use_latest_if_stamp_unavailable").as_bool();
    tf_latest_max_age_s_ = std::max(0.0, get_parameter("tf_latest_max_age_s").as_double());
    max_keyframe_linear_speed_mps_ =
      std::max(0.0, get_parameter("max_keyframe_linear_speed_mps").as_double());
    max_keyframe_angular_speed_rps_ =
      std::max(0.0, get_parameter("max_keyframe_angular_speed_rps").as_double());
    optimized_path_topic_ = get_parameter("optimized_path_topic").as_string();
    registration_check_hz_ = std::max(0.0, get_parameter("registration_check_hz").as_double());
    registration_check_min_keyframes_ = std::max(0, static_cast<int>(
      get_parameter("registration_check_min_keyframes").as_int()));
    registration_submap_keyframes_ = std::max(1, static_cast<int>(
      get_parameter("registration_submap_keyframes").as_int()));
    registration_max_points_ = std::max(1000, static_cast<int>(
      get_parameter("registration_max_points").as_int()));
    registration_voxel_size_m_ = std::max(0.03, get_parameter("registration_voxel_size_m").as_double());
    registration_max_fitness_score_ = std::max(0.0, get_parameter(
      "registration_max_fitness_score").as_double());
    registration_accept_nonconverged_if_quality_ok_ = get_parameter(
      "registration_accept_nonconverged_if_quality_ok").as_bool();
    registration_nonconverged_fitness_score_ = std::max(0.0, get_parameter(
      "registration_nonconverged_fitness_score").as_double());
    registration_max_translation_correction_m_ = std::max(0.0, get_parameter(
      "registration_max_translation_correction_m").as_double());
    registration_max_rotation_correction_rad_ = get_parameter(
      "registration_max_rotation_correction_deg").as_double() * kPi / 180.0;
    registration_use_cuda_ = get_parameter("registration_use_cuda").as_bool();

    sc_num_rings_ = std::max(1, static_cast<int>(get_parameter("scan_context_num_rings").as_int()));
    sc_num_sectors_ = std::max(4, static_cast<int>(get_parameter("scan_context_num_sectors").as_int()));
    sc_min_radius_ = std::max(0.0, get_parameter("scan_context_min_radius").as_double());
    sc_max_radius_ = std::max(sc_min_radius_ + 1.0, get_parameter("scan_context_max_radius").as_double());
    sc_height_offset_ = get_parameter("scan_context_height_offset").as_double();
    sc_max_height_ = std::max(0.1, get_parameter("scan_context_max_height").as_double());
    sc_min_points_ = std::max(1, static_cast<int>(get_parameter("scan_context_min_points").as_int()));
    sc_exclude_recent_ = std::max(1, static_cast<int>(get_parameter("scan_context_exclude_recent").as_int()));
    sc_search_candidates_ = std::max(1, static_cast<int>(get_parameter("scan_context_search_candidates").as_int()));
    sc_verify_candidates_ = std::max(1, static_cast<int>(get_parameter("scan_context_verify_candidates").as_int()));
    sc_score_thresh_ = get_parameter("scan_context_score_thresh").as_double();
    sc_score_margin_ = get_parameter("scan_context_score_margin").as_double();

    loop_min_travel_distance_m_ = std::max(0.0, get_parameter("loop_min_travel_distance_m").as_double());
    loop_fitness_score_thresh_ = get_parameter("loop_fitness_score_thresh").as_double();
    loop_max_relative_translation_m_ = std::max(0.0, get_parameter("loop_max_relative_translation_m").as_double());
    loop_max_relative_z_m_ = std::max(0.0, get_parameter("loop_max_relative_z_m").as_double());
    loop_max_relative_rotation_rad_ = get_parameter("loop_max_relative_rotation_deg").as_double() * kPi / 180.0;
    loop_submap_max_keyframes_ = std::max(1, static_cast<int>(get_parameter("loop_submap_max_keyframes").as_int()));
    loop_submap_voxel_size_m_ = std::max(0.01, get_parameter("loop_submap_voxel_size_m").as_double());
    gicp_num_threads_ = std::max(1, static_cast<int>(get_parameter("gicp_num_threads").as_int()));
    gicp_correspondence_randomness_ = std::max(5, static_cast<int>(get_parameter("gicp_correspondence_randomness").as_int()));
    gicp_max_correspondence_distance_ = std::max(0.1, get_parameter("gicp_max_correspondence_distance").as_double());
    gicp_maximum_iterations_ = std::max(1, static_cast<int>(get_parameter("gicp_maximum_iterations").as_int()));
    gicp_backend_ = get_parameter("gicp_backend").as_string();
    gicp_transformation_epsilon_ = std::max(1e-6, get_parameter("gicp_transformation_epsilon").as_double());
    gicp_euclidean_fitness_epsilon_ =
      std::max(1e-9, get_parameter("gicp_euclidean_fitness_epsilon").as_double());

    graph_optimize_period_keyframes_ =
      std::max(1, static_cast<int>(get_parameter("graph_optimize_period_keyframes").as_int()));
    graph_optimization_iterations_ =
      std::max(1, static_cast<int>(get_parameter("graph_optimization_iterations").as_int()));
    prior_position_sigma_ = std::max(1e-6, get_parameter("prior_position_sigma").as_double());
    prior_rotation_sigma_ = std::max(1e-6, get_parameter("prior_rotation_sigma").as_double());
    odom_position_sigma_ = std::max(1e-6, get_parameter("odom_position_sigma").as_double());
    odom_rotation_sigma_ = std::max(1e-6, get_parameter("odom_rotation_sigma").as_double());
    loop_position_sigma_ = std::max(1e-6, get_parameter("loop_position_sigma").as_double());
    loop_rotation_sigma_ = std::max(1e-6, get_parameter("loop_rotation_sigma").as_double());

    enable_keyframe_cloud_cache_ = get_parameter("enable_keyframe_cloud_cache").as_bool();
    keyframe_cloud_cache_dir_ = get_parameter("keyframe_cloud_cache_dir").as_string();
    keyframe_cloud_cache_keep_recent_ =
      std::max(0, static_cast<int>(get_parameter("keyframe_cloud_cache_keep_recent").as_int()));
    keyframe_cloud_cache_cleanup_on_start_ = get_parameter("keyframe_cloud_cache_cleanup_on_start").as_bool();
    max_keyframe_clouds_in_memory_ =
      std::max(10, static_cast<int>(get_parameter("max_keyframe_clouds_in_memory").as_int()));
    max_archived_map_points_ =
      std::max(10000, static_cast<int>(get_parameter("max_archived_map_points").as_int()));

    save_map_on_shutdown_ = get_parameter("save_map_on_shutdown").as_bool();
    save_map_path_ = get_parameter("save_map_path").as_string();
    save_timestamped_snapshots_ = get_parameter("save_timestamped_snapshots").as_bool();
    shutdown_save_root_ = get_parameter("shutdown_save_root").as_string();
    shutdown_save_name_ = get_parameter("shutdown_save_name").as_string();
    latest_map_path_ = get_parameter("latest_map_path").as_string();
    latest_metadata_path_ = get_parameter("latest_metadata_path").as_string();
    latest_map_pcd_path_ = get_parameter("latest_map_pcd_path").as_string();
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    const auto now = now_seconds();
    if (insert_hz_ > 0.0 && now - last_insert_wall_s_ < 1.0 / insert_hz_) {
      return;
    }
    last_insert_wall_s_ = now;

    const std::string source_frame = source_frame_override_.empty() ?
      msg->header.frame_id : source_frame_override_;
    if (source_frame.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "input cloud has empty frame_id");
      return;
    }

    const auto pose = lookupPose(source_frame, msg->header.stamp);
    if (!pose) {
      return;
    }

    // A correct pose at the scan reference time does not guarantee that every
    // point in a moving scan is usable.  Reject scans whose pose delta implies
    // an excessive motion rate; this prevents a distorted frame from being
    // permanently fused into the graph map while leaving odometry untouched.
    const double stamp_s = rclcpp::Time(msg->header.stamp).seconds();
    if (has_last_input_pose_ && stamp_s > last_input_stamp_s_ + 1e-4) {
      const double dt = stamp_s - last_input_stamp_s_;
      const double linear_speed =
        (pose->translation() - last_input_pose_.translation()).norm() / dt;
      const double angular_speed =
        rotationAngle(last_input_pose_.rotation().transpose() * pose->rotation()) / dt;
      if ((max_keyframe_linear_speed_mps_ > 0.0 &&
        linear_speed > max_keyframe_linear_speed_mps_) ||
        (max_keyframe_angular_speed_rps_ > 0.0 &&
        angular_speed > max_keyframe_angular_speed_rps_))
      {
        skipped_keyframes_++;
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "skip distorted cloud: linear_speed=%.2f m/s angular_speed=%.2f rad/s",
          linear_speed, angular_speed);
        return;
      }
    }
    last_input_pose_ = *pose;
    last_input_stamp_s_ = stamp_s;
    has_last_input_pose_ = true;

    if (!acceptKeyframe(*pose, now)) {
      skipped_keyframes_++;
      return;
    }

    CloudT::Ptr cloud(new CloudT());
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty()) {
      return;
    }

    cloud = filterInputCloud(cloud);
    cloud = voxelDownsample(cloud, keyframe_voxel_size_m_);
    // Points outside different sides of a footprint corner can occupy one
    // voxel; their centroid can fall back inside the body. Enforce the same
    // exclusion again before caching, registration and global accumulation.
    cloud = filterInputCloud(cloud);
    cloud = limitCloud(cloud, max_keyframe_points_);
    if (static_cast<int>(cloud->size()) < sc_min_points_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000,
        "skip keyframe: cloud too small after filtering (%zu points)", cloud->size());
      return;
    }

    Eigen::Isometry3d accepted_pose = *pose;
    double registration_fitness = std::numeric_limits<double>::infinity();
    try {
      if (!checkScanRegistration(*pose, cloud, accepted_pose, registration_fitness)) {
        skipped_keyframes_++;
        registration_rejected_++;
        return;
      }
    } catch (const std::exception & exc) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "registration check failed, dropping frame: %s", exc.what());
      skipped_keyframes_++;
      registration_rejected_++;
      return;
    }

    // Commit the keyframe reference only after all quality gates pass.  A
    // rejected scan must not become the baseline used by the next candidate.
    last_keyframe_pose_ = accepted_pose;
    last_keyframe_wall_s_ = now;
    has_last_keyframe_pose_ = true;

    Keyframe keyframe;
    keyframe.id = static_cast<int>(keyframes_.size());
    keyframe.stamp = rclcpp::Time(msg->header.stamp);
    keyframe.odom_pose = accepted_pose;
    keyframe.optimized_pose = toPose3(accepted_pose);
    keyframe.cloud = cloud;
    keyframe.scan_context = makeScanContext(cloud);
    keyframes_.push_back(keyframe);
    optimized_poses_.push_back(keyframe.optimized_pose);
    cacheKeyframeCloud(keyframe);
    enforceCloudBudget();

    publishCurrentCloud(keyframe);

    const int new_index = static_cast<int>(keyframes_.size()) - 1;
    const auto loop = detectAndVerifyLoop(new_index);
    if (loop) {
      loop_edges_.push_back(*loop);
      accepted_loops_++;
      RCLCPP_INFO(
        get_logger(),
        "accepted SC+FastGICP loop %d -> %d score=%.3f fitness=%.3f",
        loop->from, loop->to, loop->score, loop->fitness);
      optimizePoseGraph();
    } else if (new_index > 0 && new_index % graph_optimize_period_keyframes_ == 0) {
      optimizePoseGraph();
    }

    map_dirty_ = true;
    publishOptimizedPath();
  }

  bool checkScanRegistration(
    const Eigen::Isometry3d & odom_pose, const CloudT::Ptr & source,
    Eigen::Isometry3d & corrected_pose, double & fitness)
  {
    // Bootstrap the map with a few keyframes.  Once the submap exists, every
    // candidate that reaches this function must be checked; silently bypassing
    // a check would allow the exact failure mode this gate is meant to stop.
    if (static_cast<int>(keyframes_.size()) < registration_check_min_keyframes_) {
      return true;
    }

    CloudT::Ptr target(new CloudT());
    // A temporal window becomes permanently stale after rejected scans or a
    // revisit. Select the map around the current pose, not the last insertion.
    std::vector<std::pair<double, int>> nearby;
    nearby.reserve(keyframes_.size());
    for (int i = 0; i < static_cast<int>(keyframes_.size()); ++i) {
      nearby.emplace_back(
        (currentPose(i).translation() - odom_pose.translation()).squaredNorm(), i);
    }
    const auto count = std::min(nearby.size(),
      static_cast<std::size_t>(std::max(1, registration_submap_keyframes_)));
    std::partial_sort(nearby.begin(), nearby.begin() + count, nearby.end());
    bool needs_archive = false;
    for (std::size_t n = 0; n < count; ++n) {
      const int i = nearby[n].second;
      if (!keyframes_[i].cloud || keyframes_[i].cloud->empty()) {
        needs_archive = true;
        continue;
      }
      CloudT transformed;
      const Eigen::Isometry3d target_from_keyframe = odom_pose.inverse() * currentPose(i);
      pcl::transformPointCloud(*keyframes_[i].cloud, transformed, toMatrix4f(target_from_keyframe));
      *target += transformed;
    }
    // Old keyframes may have been merged into the archive by the RAM budget.
    // Keep those areas usable for registration when the robot returns to them.
    if (needs_archive && archived_map_ && !archived_map_->empty()) {
      CloudT transformed;
      pcl::transformPointCloud(*archived_map_, transformed, toMatrix4f(odom_pose.inverse()));
      const double radius = local_radius_m_ > 0.0 ? local_radius_m_ : max_range_m_;
      const double bound = radius + gicp_max_correspondence_distance_;
      for (const auto & point : transformed) {
        if (radius <= 0.0 || point.getVector3fMap().squaredNorm() <= bound * bound) {
          target->push_back(point);
        }
      }
    }
    target = voxelDownsample(target, registration_voxel_size_m_);
    target = limitCloud(target, registration_max_points_);
    CloudT::Ptr input = limitCloud(source, registration_max_points_);
    if (target->size() < 100 || input->size() < 100) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "drop scan registration: insufficient points target=%zu input=%zu",
        target->size(), input->size());
      return false;
    }

    Eigen::Isometry3d correction = Eigen::Isometry3d::Identity();
    bool converged = false;
    bool cuda_registration_used = false;
#ifdef USE_FAST_VGICP_CUDA
    const bool use_cuda = registration_use_cuda_ &&
      (gicp_backend_ == "cuda" || gicp_backend_ == "auto");
    if (use_cuda) {
      cuda_registration_used = true;
      RCLCPP_INFO_ONCE(get_logger(), "mapper keyframe registration executing CUDA FastVGICP");
      fast_gicp::FastVGICPCuda<PointT, PointT> gicp;
      gicp.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      gicp.setTransformationEpsilon(0.005);
      gicp.setRotationEpsilon(0.005);
      // Regularize near-degenerate planar covariances before the GPU solve.
      gicp.setRegularizationMethod(fast_gicp::RegularizationMethod::FROBENIUS);
      gicp.setCorrespondenceRandomness(gicp_correspondence_randomness_);
      gicp.setResolution(std::max(0.10, registration_voxel_size_m_));
      const auto cuda_search = get_parameter("cuda_search_method").as_string();
      gicp.setNearestNeighborSearchMethod(
        cuda_search == "GPU_RBF_KERNEL" ? fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL :
        fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
      gicp.setMaxCorrespondenceDistance(gicp_max_correspondence_distance_);
      gicp.setMaximumIterations(std::min(gicp_maximum_iterations_, 32));
      gicp.setInputSource(input);
      gicp.setInputTarget(target);
      CloudT aligned;
      gicp.align(aligned, Eigen::Matrix4f::Identity());
      converged = gicp.hasConverged();
      correction.matrix() = gicp.getFinalTransformation().cast<double>();
      if (!correction.matrix().allFinite()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "drop CUDA registration: FastVGICP returned a non-finite transform");
        return false;
      }
      fitness = gicp.getFitnessScore(gicp_max_correspondence_distance_ * gicp_max_correspondence_distance_);
    } else
#endif
    {
      fast_gicp::FastGICP<PointT, PointT> gicp;
      gicp.setNumThreads(gicp_num_threads_);
      gicp.setCorrespondenceRandomness(gicp_correspondence_randomness_);
      gicp.setMaxCorrespondenceDistance(gicp_max_correspondence_distance_);
      gicp.setMaximumIterations(std::min(gicp_maximum_iterations_, 32));
      gicp.setInputSource(input);
      gicp.setInputTarget(target);
      CloudT aligned;
      gicp.align(aligned, Eigen::Matrix4f::Identity());
      converged = gicp.hasConverged();
      correction.matrix() = gicp.getFinalTransformation().cast<double>();
      if (!correction.matrix().allFinite()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "drop CPU registration: FastGICP returned a non-finite transform");
        return false;
      }
      fitness = gicp.getFitnessScore(gicp_max_correspondence_distance_ * gicp_max_correspondence_distance_);
    }

    const bool quality_limited_nonconverged = !converged &&
      registration_accept_nonconverged_if_quality_ok_ && cuda_registration_used &&
      registration_nonconverged_fitness_score_ > 0.0 &&
      std::isfinite(fitness) && fitness <= registration_nonconverged_fitness_score_;
    if ((!converged && !quality_limited_nonconverged) || !std::isfinite(fitness) ||
      (registration_max_fitness_score_ > 0.0 && fitness > registration_max_fitness_score_)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "drop scan registration: converged=%d fitness=%.3f", converged, fitness);
      return false;
    }
    const double correction_translation = correction.translation().norm();
    const double correction_rotation = rotationAngle(correction.rotation());
    if ((registration_max_translation_correction_m_ > 0.0 &&
      correction_translation > registration_max_translation_correction_m_) ||
      (registration_max_rotation_correction_rad_ > 0.0 &&
      correction_rotation > registration_max_rotation_correction_rad_)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "drop scan correction: fitness=%.3f translation=%.3f rotation=%.3f",
        fitness, correction_translation, correction_rotation);
      return false;
    }
    if (correction_translation > 0.01 || correction_rotation > 0.01) {
      corrected_pose = odom_pose * correction;
      registration_corrected_++;
    }
    return true;
  }

  void publishOptimizedPath()
  {
    if (!optimized_path_pub_) {
      return;
    }
    nav_msgs::msg::Path path;
    path.header.frame_id = output_frame_;
    path.header.stamp = get_clock()->now();
    path.poses.reserve(keyframes_.size());
    for (std::size_t i = 0; i < keyframes_.size(); ++i) {
      const Eigen::Isometry3d pose = currentPose(static_cast<int>(i));
      geometry_msgs::msg::PoseStamped pose_msg;
      pose_msg.header.frame_id = output_frame_;
      pose_msg.header.stamp = keyframes_[i].stamp;
      pose_msg.pose.position.x = pose.translation().x();
      pose_msg.pose.position.y = pose.translation().y();
      pose_msg.pose.position.z = pose.translation().z();
      const Eigen::Quaterniond q(pose.rotation());
      pose_msg.pose.orientation.x = q.x();
      pose_msg.pose.orientation.y = q.y();
      pose_msg.pose.orientation.z = q.z();
      pose_msg.pose.orientation.w = q.w();
      path.poses.push_back(pose_msg);
    }
    optimized_path_pub_->publish(path);
  }

  std::optional<Eigen::Isometry3d> lookupPose(
    const std::string & source_frame,
    const builtin_interfaces::msg::Time & stamp)
  {
    try {
      geometry_msgs::msg::TransformStamped transform;
      if (stamp.sec == 0 && stamp.nanosec == 0) {
        transform = tf_buffer_.lookupTransform(
          output_frame_, source_frame, tf2::TimePointZero, tf2::durationFromSec(tf_timeout_s_));
      } else {
        transform = tf_buffer_.lookupTransform(
          output_frame_, source_frame, stamp, tf2::durationFromSec(tf_timeout_s_));
      }
      return tf2::transformToEigen(transform);
    } catch (const tf2::TransformException & exc) {
      if (tf_use_latest_if_stamp_unavailable_) {
        try {
          const auto latest = tf_buffer_.lookupTransform(
            output_frame_, source_frame, tf2::TimePointZero,
            tf2::durationFromSec(tf_timeout_s_));
          const double requested_s = rclcpp::Time(stamp).seconds();
          const double latest_s = rclcpp::Time(latest.header.stamp).seconds();
          const double age_s = requested_s - latest_s;
          // A small positive age is expected here: the driver publishes a
          // cloud stamped ahead of the TF bridge.  Never accept an old TF or
          // an unrelated future transform.
          if (tf_latest_max_age_s_ <= 0.0 ||
            (age_s >= -0.05 && age_s <= tf_latest_max_age_s_))
          {
            tf_latest_fallbacks_++;
            RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 3000,
              "exact TF %s <- %s unavailable; using latest TF age=%.3f s",
              output_frame_.c_str(), source_frame.c_str(), age_s);
            return tf2::transformToEigen(latest);
          }
        } catch (const tf2::TransformException &) {
          // Fall through to the throttled diagnostic below.
        }
      }
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "TF %s <- %s failed: %s", output_frame_.c_str(), source_frame.c_str(), exc.what());
      return std::nullopt;
    }
  }

  bool acceptKeyframe(const Eigen::Isometry3d & pose, double now)
  {
    if (!has_last_keyframe_pose_) {
      return true;
    }

    if (max_keyframe_interval_s_ > 0.0 && now - last_keyframe_wall_s_ >= max_keyframe_interval_s_) {
      return true;
    }

    const double translation = (pose.translation() - last_keyframe_pose_.translation()).norm();
    const Eigen::Matrix3d relative_rotation = last_keyframe_pose_.rotation().transpose() * pose.rotation();
    const double rotation = rotationAngle(relative_rotation);

    if (translation >= min_keyframe_translation_m_ || rotation >= min_keyframe_rotation_rad_) {
      return true;
    }
    return false;
  }

  CloudT::Ptr filterInputCloud(const CloudT::Ptr & input)
  {
    CloudT::Ptr filtered(new CloudT());
    filtered->reserve(input->size());
    const double min_range_sq = min_range_m_ * min_range_m_;
    const double max_range_sq = max_range_m_ > 0.0 ? max_range_m_ * max_range_m_ : 0.0;
    const double local_radius_sq = local_radius_m_ > 0.0 ? local_radius_m_ * local_radius_m_ : 0.0;

    for (const auto & point : input->points) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
        continue;
      }
      const double range_sq = point.x * point.x + point.y * point.y + point.z * point.z;
      if (self_filter_.contains(point.x, point.y)) {continue;}
      if (min_range_m_ > 0.0 && range_sq < min_range_sq) {
        continue;
      }
      if (max_range_m_ > 0.0 && range_sq > max_range_sq) {
        continue;
      }
      if (local_radius_m_ > 0.0 && range_sq > local_radius_sq) {
        continue;
      }
      filtered->push_back(point);
    }
    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;
    return limitCloud(filtered, max_input_points_);
  }

  CloudT::Ptr limitCloud(const CloudT::Ptr & input, int max_points) const
  {
    if (max_points <= 0 || static_cast<int>(input->size()) <= max_points) {
      return input;
    }
    const int stride = static_cast<int>(std::ceil(static_cast<double>(input->size()) / max_points));
    CloudT::Ptr limited(new CloudT());
    limited->reserve(max_points);
    for (std::size_t i = 0; i < input->size(); i += stride) {
      limited->push_back(input->points[i]);
    }
    limited->width = limited->size();
    limited->height = 1;
    limited->is_dense = false;
    return limited;
  }

  CloudT::Ptr voxelDownsample(const CloudT::Ptr & input, double voxel_size) const
  {
    if (input->empty() || voxel_size <= 0.0) {
      return input;
    }
    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize(
      static_cast<float>(voxel_size),
      static_cast<float>(voxel_size),
      static_cast<float>(voxel_size));
    voxel.setInputCloud(input);
    CloudT::Ptr output(new CloudT());
    voxel.filter(*output);
    return output;
  }

  ScanContextDescriptor makeScanContext(const CloudT::Ptr & cloud) const
  {
    ScanContextDescriptor context;
    context.descriptor = Eigen::MatrixXf::Zero(sc_num_rings_, sc_num_sectors_);
    context.ring_key = Eigen::VectorXf::Zero(sc_num_rings_);
    const double radius_range = sc_max_radius_ - sc_min_radius_;
    if (radius_range <= 0.0) {
      return context;
    }

    for (const auto & point : cloud->points) {
      const double radius = std::hypot(point.x, point.y);
      if (radius < sc_min_radius_ || radius > sc_max_radius_) {
        continue;
      }
      double theta = std::atan2(point.y, point.x);
      if (theta < 0.0) {
        theta += 2.0 * kPi;
      }
      const int ring = std::clamp(
        static_cast<int>((radius - sc_min_radius_) / radius_range * sc_num_rings_),
        0, sc_num_rings_ - 1);
      const int sector = std::clamp(
        static_cast<int>(theta / (2.0 * kPi) * sc_num_sectors_),
        0, sc_num_sectors_ - 1);
      const float height = static_cast<float>(
        std::clamp(static_cast<double>(point.z) + sc_height_offset_, 0.0, sc_max_height_));
      context.descriptor(ring, sector) = std::max(context.descriptor(ring, sector), height);
      context.valid_points++;
    }

    for (int ring = 0; ring < sc_num_rings_; ++ring) {
      context.ring_key(ring) = context.descriptor.row(ring).mean();
    }
    return context;
  }

  std::optional<LoopEdge> detectAndVerifyLoop(int new_index)
  {
    if (new_index <= sc_exclude_recent_) {
      return std::nullopt;
    }
    if (keyframes_[new_index].scan_context.valid_points < sc_min_points_) {
      return std::nullopt;
    }

    std::vector<std::pair<double, int>> ring_candidates;
    ring_candidates.reserve(new_index);
    for (int i = 0; i < new_index - sc_exclude_recent_; ++i) {
      if (keyframes_[i].scan_context.valid_points < sc_min_points_) {
        continue;
      }
      if (travelDistanceBetween(i, new_index) < loop_min_travel_distance_m_) {
        continue;
      }
      const double ring_distance =
        (keyframes_[new_index].scan_context.ring_key - keyframes_[i].scan_context.ring_key).norm();
      ring_candidates.emplace_back(ring_distance, i);
    }
    if (ring_candidates.empty()) {
      return std::nullopt;
    }
    std::sort(ring_candidates.begin(), ring_candidates.end());
    if (static_cast<int>(ring_candidates.size()) > sc_search_candidates_) {
      ring_candidates.resize(sc_search_candidates_);
    }

    std::vector<ScanContextMatch> matches;
    matches.reserve(ring_candidates.size());
    for (const auto & candidate : ring_candidates) {
      auto match = matchScanContexts(
        keyframes_[new_index].scan_context,
        keyframes_[candidate.second].scan_context);
      match.keyframe_index = candidate.second;
      matches.push_back(match);
    }
    std::sort(
      matches.begin(), matches.end(),
      [](const ScanContextMatch & lhs, const ScanContextMatch & rhs) {
        return lhs.score < rhs.score;
      });
    if (matches.empty()) {
      return std::nullopt;
    }
    for (std::size_t i = 0; i < matches.size(); ++i) {
      matches[i].second_score = (i + 1 < matches.size()) ? matches[i + 1].score :
        std::numeric_limits<double>::infinity();
    }

    const int verify_count = std::min<int>(sc_verify_candidates_, matches.size());
    for (int i = 0; i < verify_count; ++i) {
      const auto & match = matches[i];
      if (match.score > sc_score_thresh_) {
        continue;
      }
      if (std::isfinite(match.second_score) && match.second_score - match.score < sc_score_margin_) {
        continue;
      }
      auto loop = verifyLoopWithGicp(match.keyframe_index, new_index, match);
      if (loop) {
        return loop;
      }
    }
    return std::nullopt;
  }

  ScanContextMatch matchScanContexts(
    const ScanContextDescriptor & query,
    const ScanContextDescriptor & reference) const
  {
    ScanContextMatch best;
    for (int shift = 0; shift < sc_num_sectors_; ++shift) {
      double cosine_sum = 0.0;
      int used_columns = 0;
      for (int sector = 0; sector < sc_num_sectors_; ++sector) {
        const auto query_column = query.descriptor.col(sector);
        const auto reference_column = reference.descriptor.col((sector + shift) % sc_num_sectors_);
        const double query_norm = query_column.norm();
        const double reference_norm = reference_column.norm();
        if (query_norm < 1e-6 || reference_norm < 1e-6) {
          continue;
        }
        cosine_sum += query_column.dot(reference_column) / (query_norm * reference_norm);
        used_columns++;
      }
      if (used_columns == 0) {
        continue;
      }
      const double score = 1.0 - cosine_sum / used_columns;
      if (score < best.score) {
        best.score = score;
        best.yaw_shift = shift;
      }
    }
    return best;
  }

  std::optional<LoopEdge> verifyLoopWithGicp(
    int candidate_index,
    int new_index,
    const ScanContextMatch & match)
  {
    CloudT::Ptr target = makeCandidateSubmap(candidate_index);
    if (!target || target->empty()) {
      return std::nullopt;
    }
    target = voxelDownsample(target, loop_submap_voxel_size_m_);

    CloudT::Ptr source = keyframes_[new_index].cloud;
    if (source->empty() || target->empty()) {
      return std::nullopt;
    }

    const Eigen::Isometry3d candidate_pose = currentPose(candidate_index);
    const Eigen::Isometry3d new_pose = currentPose(new_index);
    const Eigen::Isometry3d initial_target_from_source = candidate_pose.inverse() * new_pose;

    Eigen::Isometry3d target_from_source = Eigen::Isometry3d::Identity();
    double fitness = std::numeric_limits<double>::infinity();
    bool converged = false;
#ifdef USE_FAST_VGICP_CUDA
    const bool use_cuda = gicp_backend_ == "cuda" || gicp_backend_ == "auto";
    if (use_cuda) {
      fast_gicp::FastVGICPCuda<PointT, PointT> gicp;
      gicp.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      gicp.setTransformationEpsilon(0.005);
      gicp.setRotationEpsilon(0.005);
      RCLCPP_INFO_ONCE(get_logger(), "mapper loop registration executing CUDA FastVGICP");
      // Use the same covariance regularizer as keyframe registration.
      gicp.setRegularizationMethod(fast_gicp::RegularizationMethod::FROBENIUS);
      gicp.setCorrespondenceRandomness(gicp_correspondence_randomness_);
      gicp.setResolution(std::max(0.50, map_voxel_size_m_));
      const auto cuda_search = get_parameter("cuda_search_method").as_string();
      gicp.setNearestNeighborSearchMethod(
        cuda_search == "GPU_RBF_KERNEL" ? fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL :
        fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
      gicp.setMaxCorrespondenceDistance(gicp_max_correspondence_distance_);
      gicp.setMaximumIterations(gicp_maximum_iterations_);
      gicp.setTransformationEpsilon(gicp_transformation_epsilon_);
      gicp.setEuclideanFitnessEpsilon(gicp_euclidean_fitness_epsilon_);
      gicp.setInputSource(source);
      gicp.setInputTarget(target);
      CloudT aligned;
      gicp.align(aligned, toMatrix4f(initial_target_from_source));
      converged = gicp.hasConverged();
      target_from_source.matrix() = gicp.getFinalTransformation().cast<double>();
      if (!target_from_source.matrix().allFinite()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "reject CUDA loop registration: FastVGICP returned a non-finite transform");
        return std::nullopt;
      }
      fitness = gicp.getFitnessScore(gicp_max_correspondence_distance_ * gicp_max_correspondence_distance_);
    } else
#endif
    {
      fast_gicp::FastGICP<PointT, PointT> gicp;
      gicp.setNumThreads(gicp_num_threads_);
      gicp.setCorrespondenceRandomness(gicp_correspondence_randomness_);
      gicp.setMaxCorrespondenceDistance(gicp_max_correspondence_distance_);
      gicp.setMaximumIterations(gicp_maximum_iterations_);
      gicp.setTransformationEpsilon(gicp_transformation_epsilon_);
      gicp.setEuclideanFitnessEpsilon(gicp_euclidean_fitness_epsilon_);
      gicp.setInputSource(source);
      gicp.setInputTarget(target);
      CloudT aligned;
      gicp.align(aligned, toMatrix4f(initial_target_from_source));
      fitness = gicp.getFitnessScore(gicp_max_correspondence_distance_ * gicp_max_correspondence_distance_);
      converged = gicp.hasConverged();
      target_from_source.matrix() = gicp.getFinalTransformation().cast<double>();
    }
    if (!converged || !std::isfinite(fitness) || fitness > loop_fitness_score_thresh_) {
      return std::nullopt;
    }
    Eigen::Isometry3d measurement_candidate_to_new = target_from_source.inverse();

    if (!validateLoopMeasurement(candidate_index, new_index, measurement_candidate_to_new)) {
      return std::nullopt;
    }

    LoopEdge edge;
    edge.from = candidate_index;
    edge.to = new_index;
    edge.measurement_from_to = measurement_candidate_to_new;
    edge.fitness = fitness;
    edge.score = match.score;
    return edge;
  }

  CloudT::Ptr makeCandidateSubmap(int candidate_index) const
  {
    CloudT::Ptr submap(new CloudT());
    const int half_window = std::max(0, loop_submap_max_keyframes_ / 2);
    const int begin = std::max(0, candidate_index - half_window);
    const int end = std::min<int>(keyframes_.size() - 1, candidate_index + half_window);
    const Eigen::Isometry3d target_pose = currentPose(candidate_index);

    for (int i = begin; i <= end; ++i) {
      const Eigen::Isometry3d target_from_i = target_pose.inverse() * currentPose(i);
      CloudT transformed;
      pcl::transformPointCloud(*keyframes_[i].cloud, transformed, toMatrix4f(target_from_i));
      *submap += transformed;
    }
    submap->width = submap->size();
    submap->height = 1;
    submap->is_dense = false;
    return submap;
  }

  bool validateLoopMeasurement(
    int candidate_index,
    int new_index,
    const Eigen::Isometry3d & measurement_candidate_to_new) const
  {
    const Eigen::Isometry3d odom_relative =
      currentPose(candidate_index).inverse() * currentPose(new_index);
    const Eigen::Isometry3d delta = odom_relative.inverse() * measurement_candidate_to_new;
    const double translation = delta.translation().norm();
    const double dz = std::abs(delta.translation().z());
    const double rotation = rotationAngle(delta.rotation());

    if (loop_max_relative_translation_m_ > 0.0 && translation > loop_max_relative_translation_m_) {
      return false;
    }
    if (loop_max_relative_z_m_ > 0.0 && dz > loop_max_relative_z_m_) {
      return false;
    }
    if (loop_max_relative_rotation_rad_ > 0.0 && rotation > loop_max_relative_rotation_rad_) {
      return false;
    }
    return true;
  }

  double travelDistanceBetween(int from, int to) const
  {
    if (from >= to) {
      return 0.0;
    }
    double distance = 0.0;
    for (int i = from + 1; i <= to; ++i) {
      distance += (keyframes_[i].odom_pose.translation() - keyframes_[i - 1].odom_pose.translation()).norm();
    }
    return distance;
  }

  Eigen::Isometry3d currentPose(int index) const
  {
    if (index >= 0 && index < static_cast<int>(optimized_poses_.size())) {
      return fromPose3(optimized_poses_[index]);
    }
    return keyframes_[index].odom_pose;
  }

  void optimizePoseGraph()
  {
    if (keyframes_.empty()) {
      return;
    }

    try {
      gtsam::NonlinearFactorGraph graph;
      gtsam::Values initial;

      graph.add(gtsam::PriorFactor<gtsam::Pose3>(
        X(0), toPose3(keyframes_.front().odom_pose), noise(prior_rotation_sigma_, prior_position_sigma_)));

      for (std::size_t i = 0; i < keyframes_.size(); ++i) {
        if (i < optimized_poses_.size()) {
          initial.insert(X(i), optimized_poses_[i]);
        } else {
          initial.insert(X(i), toPose3(keyframes_[i].odom_pose));
        }
        if (i == 0) {
          continue;
        }
        const Eigen::Isometry3d odom_relative =
          keyframes_[i - 1].odom_pose.inverse() * keyframes_[i].odom_pose;
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
          X(i - 1), X(i), toPose3(odom_relative), noise(odom_rotation_sigma_, odom_position_sigma_)));
      }

      for (const auto & edge : loop_edges_) {
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
          X(edge.from), X(edge.to), toPose3(edge.measurement_from_to),
          noise(loop_rotation_sigma_, loop_position_sigma_)));
      }

      gtsam::LevenbergMarquardtParams params;
      params.setMaxIterations(graph_optimization_iterations_);
      params.setVerbosityLM("SILENT");
      gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
      const auto result = optimizer.optimize();

      // Treat graph optimization as a transaction.  GTSAM can return a
      // numerically valid Values object even when the solve diverged or made
      // the factor graph worse.  Never replace the last accepted trajectory
      // in that case; the next keyframe can still be processed safely.
      const double initial_error = graph.error(initial);
      const double result_error = graph.error(result);
      if (!std::isfinite(initial_error) || !std::isfinite(result_error) ||
        result_error > initial_error + std::max(1e-6, 1e-5 * std::abs(initial_error)))
      {
        RCLCPP_WARN(
          get_logger(),
          "reject pose graph optimization: initial_error=%.6g result_error=%.6g",
          initial_error, result_error);
        return;
      }

      std::vector<gtsam::Pose3> candidate_poses;
      candidate_poses.reserve(keyframes_.size());
      for (std::size_t i = 0; i < keyframes_.size(); ++i) {
        const auto pose = result.at<gtsam::Pose3>(X(i));
        if (!pose.matrix().allFinite()) {
          RCLCPP_WARN(get_logger(), "reject pose graph optimization: non-finite pose at %zu", i);
          return;
        }
        candidate_poses.push_back(pose);
      }

      optimized_poses_.clear();
      optimized_poses_ = std::move(candidate_poses);
      graph_optimizations_++;
      map_dirty_ = true;
      publishOptimizedPath();
    } catch (const std::exception & exc) {
      RCLCPP_WARN(get_logger(), "pose graph optimization failed: %s", exc.what());
    }
  }

  gtsam::SharedNoiseModel noise(double rotation_sigma, double position_sigma) const
  {
    gtsam::Vector6 sigmas;
    sigmas << rotation_sigma, rotation_sigma, rotation_sigma,
      position_sigma, position_sigma, position_sigma;
    return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
  }

  void publishCurrentCloud(const Keyframe & keyframe)
  {
    const double now = now_seconds();
    if (current_publish_hz_ > 0.0 && now - last_current_publish_wall_s_ < 1.0 / current_publish_hz_) {
      return;
    }
    last_current_publish_wall_s_ = now;

    CloudT::Ptr current = limitCloud(keyframe.cloud, max_current_points_);
    CloudT transformed;
    pcl::transformPointCloud(*current, transformed, toMatrix4f(currentPose(keyframe.id)));
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(transformed, msg);
    msg.header.frame_id = output_frame_;
    msg.header.stamp = keyframe.stamp;
    current_pub_->publish(msg);
  }

  void publishMapTimer()
  {
    if (!map_dirty_ && !last_map_cloud_->empty()) {
      publishMapCloud(last_map_cloud_);
      return;
    }
    if (keyframes_.empty()) {
      return;
    }
    last_map_cloud_ = buildMapCloud(max_publish_points_);
    publishMapCloud(last_map_cloud_);
    map_dirty_ = false;
  }

  void publishMapCloud(const CloudT::Ptr & cloud)
  {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.frame_id = output_frame_;
    msg.header.stamp = get_clock()->now();
    map_pub_->publish(msg);

    // Publish deterministic bounded chunks as well.  RViz can subscribe to
    // these topics and keep the full map density without one giant render
    // buffer or a global point-count reduction.
    for (int shard = 0; shard < map_shard_count_; ++shard) {
      sensor_msgs::msg::PointCloud2 shard_msg;
      const std::size_t begin = static_cast<std::size_t>(shard);
      if (begin < cloud->size()) {
        CloudT part;
        // Spatially neutral round-robin partition: all points are retained,
        // while every message stays close to cloud_size / shard_count.  This
        // avoids a fixed global cap as the map grows over long runs.
        part.points.reserve((cloud->size() + static_cast<std::size_t>(map_shard_count_) - 1) /
          static_cast<std::size_t>(map_shard_count_));
        for (std::size_t i = begin; i < cloud->size(); i += static_cast<std::size_t>(map_shard_count_)) {
          part.points.push_back(cloud->points[i]);
        }
        part.width = part.points.size();
        part.height = 1;
        part.is_dense = false;
        pcl::toROSMsg(part, shard_msg);
      } else {
        CloudT empty;
        pcl::toROSMsg(empty, shard_msg);
      }
      shard_msg.header.frame_id = output_frame_;
      shard_msg.header.stamp = msg.header.stamp;
      map_shard_pubs_[shard]->publish(shard_msg);
    }
  }

  CloudT::Ptr buildMapCloud(int max_points) const
  {
    CloudT::Ptr map(new CloudT());
    if (archived_map_ && !archived_map_->empty()) {
      *map += *archived_map_;
    }
    for (const auto & keyframe : keyframes_) {
      if (!keyframe.cloud || keyframe.cloud->empty()) {
        continue;
      }
      CloudT transformed;
      pcl::transformPointCloud(*keyframe.cloud, transformed, toMatrix4f(currentPose(keyframe.id)));
      *map += transformed;
    }
    map = voxelDownsample(map, map_voxel_size_m_);
    map = limitCloud(map, max_points);
    map->width = map->size();
    map->height = 1;
    map->is_dense = false;
    return map;
  }

  void enforceCloudBudget()
  {
    if (static_cast<int>(keyframes_.size()) <= max_keyframe_clouds_in_memory_) {
      return;
    }
    const std::size_t keep_from = keyframes_.size() - static_cast<std::size_t>(max_keyframe_clouds_in_memory_);
    for (std::size_t i = 0; i < keep_from; ++i) {
      auto & cloud = keyframes_[i].cloud;
      if (!cloud || cloud->empty()) {
        continue;
      }
      CloudT transformed;
      pcl::transformPointCloud(*cloud, transformed, toMatrix4f(currentPose(static_cast<int>(i))));
      *archived_map_ += transformed;
      cloud.reset(new CloudT());
    }
    if (static_cast<int>(archived_map_->size()) > max_archived_map_points_) {
      archived_map_ = voxelDownsample(archived_map_, std::max(0.15, map_voxel_size_m_));
      archived_map_ = limitCloud(archived_map_, max_archived_map_points_);
    }
  }

  void cacheKeyframeCloud(const Keyframe & keyframe)
  {
    if (!enable_keyframe_cloud_cache_ || keyframe_cloud_cache_dir_.empty()) {
      return;
    }
    try {
      std::filesystem::create_directories(keyframe_cloud_cache_dir_);
      const std::string path = keyframe_cloud_cache_dir_ + "/sc_gicp_keyframe_" +
        zeroPad(keyframe.id, 6) + ".ply";
      writePly(path, *keyframe.cloud);
      keyframe_cache_paths_.push_back(path);
      while (
        keyframe_cloud_cache_keep_recent_ > 0 &&
        static_cast<int>(keyframe_cache_paths_.size()) > keyframe_cloud_cache_keep_recent_)
      {
        std::error_code ec;
        std::filesystem::remove(keyframe_cache_paths_.front(), ec);
        keyframe_cache_paths_.erase(keyframe_cache_paths_.begin());
      }
    } catch (const std::exception & exc) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "failed to cache keyframe cloud: %s", exc.what());
    }
  }

  void cleanupKeyframeCache()
  {
    if (!keyframe_cloud_cache_cleanup_on_start_ || keyframe_cloud_cache_dir_.empty()) {
      return;
    }
    std::error_code ec;
    std::filesystem::remove_all(keyframe_cloud_cache_dir_, ec);
  }

  void saveMapSnapshot(const std::string & reason)
  {
    if (keyframes_.empty()) {
      RCLCPP_INFO(get_logger(), "no SC+GICP map to save");
      return;
    }
    const std::string path = resolveSavePath(reason);
    try {
      auto map = buildMapCloud(0);
      writePly(path, *map);
      // PCD is the localization hand-off artifact. Binary PCD preserves the
      // full retained cloud and loads much faster than the ASCII PLY used for
      // lightweight visualization compatibility.
      if (!latest_map_pcd_path_.empty()) {
        const auto pcd_parent = std::filesystem::path(latest_map_pcd_path_).parent_path();
        if (!pcd_parent.empty()) {std::filesystem::create_directories(pcd_parent);}
        const auto tmp_pcd = latest_map_pcd_path_ + ".tmp-" + std::to_string(getpid());
        pcl::io::savePCDFileBinary(tmp_pcd, *map);
        std::filesystem::rename(tmp_pcd, latest_map_pcd_path_);
      }
      updateLatestOutputs(path, reason, map->size());
      RCLCPP_INFO(get_logger(), "saved SC+GICP 3D map %s points=%zu", path.c_str(), map->size());
    } catch (const std::exception & exc) {
      RCLCPP_WARN(get_logger(), "failed to save SC+GICP map to %s: %s", path.c_str(), exc.what());
    }
  }

  std::string resolveSavePath(const std::string & reason) const
  {
    if (!save_timestamped_snapshots_) {
      return save_map_path_;
    }
    const std::string run_name = shutdown_save_name_.empty() ? timestampName() : shutdown_save_name_;
    std::filesystem::path root(shutdown_save_root_.empty() ? "/home/jszr/zsibot_maps" : shutdown_save_root_);
    std::filesystem::path path = root / run_name;
    if (!shutdown_save_name_.empty() && reason != "shutdown") {
      path += "_" + reason;
    }
    return (path / "map.ply").string();
  }

  void updateLatestOutputs(
    const std::string & target_path,
    const std::string & reason,
    std::size_t point_count) const
  {
    if (!latest_map_path_.empty()) {
      try {
        std::filesystem::create_directories(std::filesystem::path(latest_map_path_).parent_path());
        std::filesystem::path tmp = latest_map_path_ + ".tmp-" + std::to_string(getpid());
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        std::filesystem::create_symlink(target_path, tmp);
        std::filesystem::rename(tmp, latest_map_path_);
      } catch (const std::exception &) {
        try {
          std::filesystem::copy_file(
            target_path, latest_map_path_, std::filesystem::copy_options::overwrite_existing);
        } catch (const std::exception &) {
        }
      }
    }
    if (!latest_metadata_path_.empty()) {
      std::filesystem::create_directories(std::filesystem::path(latest_metadata_path_).parent_path());
      std::ofstream metadata(latest_metadata_path_);
      metadata << "saved_at=" << timestampName() << "\n";
      metadata << "reason=" << reason << "\n";
      metadata << "map_ply=" << target_path << "\n";
      metadata << "map_pcd=" << latest_map_pcd_path_ << "\n";
      metadata << "points=" << point_count << "\n";
      metadata << "keyframes=" << keyframes_.size() << "\n";
      metadata << "loop_edges=" << loop_edges_.size() << "\n";
      metadata << "graph_optimizations=" << graph_optimizations_ << "\n";
      metadata << "backend=scan_context_fast_gicp_gtsam\n";
    }
  }

  void writePly(const std::string & path, const CloudT & cloud) const
  {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("cannot open output file");
    }
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << cloud.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "end_header\n";
    out.setf(std::ios::fixed);
    out.precision(4);
    for (const auto & point : cloud.points) {
      out << point.x << ' ' << point.y << ' ' << point.z << '\n';
    }
  }

  void publishStatus()
  {
    std_msgs::msg::String msg;
    msg.data =
      "keyframes=" + std::to_string(keyframes_.size()) +
      " loops=" + std::to_string(loop_edges_.size()) +
      " accepted_loops=" + std::to_string(accepted_loops_) +
      " optimizations=" + std::to_string(graph_optimizations_) +
      " skipped_keyframes=" + std::to_string(skipped_keyframes_) +
      " registration_rejected=" + std::to_string(registration_rejected_) +
      " registration_corrected=" + std::to_string(registration_corrected_) +
      " tf_latest_fallbacks=" + std::to_string(tf_latest_fallbacks_);
    status_pub_->publish(msg);
  }

  static std::string zeroPad(int value, int width)
  {
    std::string text = std::to_string(value);
    if (static_cast<int>(text.size()) >= width) {
      return text;
    }
    return std::string(width - text.size(), '0') + text;
  }

  double now_seconds() const
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> map_shard_pubs_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr current_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr optimized_path_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_service_;
  rclcpp::TimerBase::SharedPtr map_timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;

  std::string input_cloud_topic_;
  std::string map_cloud_topic_;
  std::string current_cloud_topic_;
  std::string status_topic_;
  std::string output_frame_;
  std::string source_frame_override_;
  std::string optimized_path_topic_{"/optimized_pose_graph_path"};

  double keyframe_voxel_size_m_{0.08};
  FootprintFilter self_filter_;
  double map_voxel_size_m_{0.10};
  int max_input_points_{60000};
  int max_keyframe_points_{50000};
  int max_publish_points_{80000};
  int map_shard_count_{8};
  int map_shard_points_{50000};
  int max_current_points_{24000};
  double insert_hz_{8.0};
  double map_publish_hz_{0.5};
  double current_publish_hz_{5.0};
  double min_range_m_{0.10};
  double max_range_m_{80.0};
  double local_radius_m_{18.0};
  double min_keyframe_translation_m_{0.25};
  double min_keyframe_rotation_rad_{8.0 * kPi / 180.0};
  double max_keyframe_interval_s_{2.0};
  double tf_timeout_s_{0.10};
  bool tf_use_latest_if_stamp_unavailable_{true};
  double tf_latest_max_age_s_{0.80};
  double max_keyframe_linear_speed_mps_{2.0};
  double max_keyframe_angular_speed_rps_{3.0};
  double registration_check_hz_{1.0};
  int registration_check_min_keyframes_{5};
  int registration_submap_keyframes_{20};
  int registration_max_points_{12000};
  double registration_voxel_size_m_{0.20};
  double registration_max_fitness_score_{0.35};
  bool registration_accept_nonconverged_if_quality_ok_{false};
  double registration_nonconverged_fitness_score_{0.08};
  double registration_max_translation_correction_m_{0.50};
  double registration_max_rotation_correction_rad_{15.0 * kPi / 180.0};
  bool registration_use_cuda_{false};

  int sc_num_rings_{20};
  int sc_num_sectors_{60};
  double sc_min_radius_{0.5};
  double sc_max_radius_{45.0};
  double sc_height_offset_{2.0};
  double sc_max_height_{6.0};
  int sc_min_points_{80};
  int sc_exclude_recent_{30};
  int sc_search_candidates_{96};
  int sc_verify_candidates_{8};
  double sc_score_thresh_{0.22};
  double sc_score_margin_{0.03};

  double loop_min_travel_distance_m_{8.0};
  double loop_fitness_score_thresh_{0.45};
  double loop_max_relative_translation_m_{8.0};
  double loop_max_relative_z_m_{2.5};
  double loop_max_relative_rotation_rad_{35.0 * kPi / 180.0};
  int loop_submap_max_keyframes_{12};
  double loop_submap_voxel_size_m_{0.18};
  int gicp_num_threads_{4};
  std::string gicp_backend_{"auto"};
  int gicp_correspondence_randomness_{20};
  double gicp_max_correspondence_distance_{1.5};
  int gicp_maximum_iterations_{64};
  double gicp_transformation_epsilon_{0.01};
  double gicp_euclidean_fitness_epsilon_{0.001};

  int graph_optimize_period_keyframes_{5};
  int graph_optimization_iterations_{32};
  double prior_position_sigma_{0.05};
  double prior_rotation_sigma_{0.03};
  double odom_position_sigma_{0.08};
  double odom_rotation_sigma_{0.04};
  double loop_position_sigma_{0.15};
  double loop_rotation_sigma_{0.08};

  bool enable_keyframe_cloud_cache_{true};
  std::string keyframe_cloud_cache_dir_;
  int keyframe_cloud_cache_keep_recent_{240};
  bool keyframe_cloud_cache_cleanup_on_start_{true};
  int max_keyframe_clouds_in_memory_{240};
  int max_archived_map_points_{300000};

  bool save_map_on_shutdown_{true};
  std::string save_map_path_;
  bool save_timestamped_snapshots_{true};
  std::string shutdown_save_root_;
  std::string shutdown_save_name_;
  std::string latest_map_path_;
  std::string latest_metadata_path_;
  std::string latest_map_pcd_path_;

  bool has_last_keyframe_pose_{false};
  Eigen::Isometry3d last_keyframe_pose_{Eigen::Isometry3d::Identity()};
  double last_keyframe_wall_s_{0.0};
  bool has_last_input_pose_{false};
  Eigen::Isometry3d last_input_pose_{Eigen::Isometry3d::Identity()};
  double last_input_stamp_s_{0.0};
  double last_insert_wall_s_{0.0};
  double last_current_publish_wall_s_{0.0};
  double last_registration_check_wall_s_{0.0};

  std::vector<Keyframe> keyframes_;
  std::vector<gtsam::Pose3> optimized_poses_;
  std::vector<LoopEdge> loop_edges_;
  std::vector<std::string> keyframe_cache_paths_;
  CloudT::Ptr archived_map_{new CloudT()};
  CloudT::Ptr last_map_cloud_{new CloudT()};

  bool map_dirty_{false};
  std::uint64_t accepted_loops_{0};
  std::uint64_t graph_optimizations_{0};
  std::uint64_t skipped_keyframes_{0};
  std::uint64_t registration_rejected_{0};
  std::uint64_t registration_corrected_{0};
  std::uint64_t tf_latest_fallbacks_{0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ScGicpPoseGraphMapper>();
  rclcpp::spin(node);
  node->saveMapOnShutdown();
  node.reset();
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
