#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fast_gicp/gicp/fast_gicp.hpp>
#ifdef USE_FAST_VGICP_CUDA
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

namespace
{
using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
constexpr double kPi = 3.14159265358979323846;

struct Descriptor
{
  Eigen::MatrixXf sectors;
  Eigen::VectorXf rings;
  int points{0};
};

struct Tile
{
  Eigen::Vector3f center{Eigen::Vector3f::Zero()};
  Descriptor descriptor;
};

struct Candidate
{
  int index{-1};
  double score{std::numeric_limits<double>::infinity()};
  int yaw_shift{0};
};

Eigen::Isometry3d fromMsg(const geometry_msgs::msg::Transform & t)
{
  Eigen::Isometry3d out = Eigen::Isometry3d::Identity();
  out.translation() = Eigen::Vector3d(t.translation.x, t.translation.y, t.translation.z);
  out.linear() = Eigen::Quaterniond(t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z)
    .normalized().toRotationMatrix();
  return out;
}

geometry_msgs::msg::Transform transformToMsg(const Eigen::Isometry3d & pose)
{
  geometry_msgs::msg::Transform out;
  out.translation.x = pose.translation().x();
  out.translation.y = pose.translation().y();
  out.translation.z = pose.translation().z();
  Eigen::Quaterniond q(pose.rotation());
  out.rotation.x = q.x(); out.rotation.y = q.y(); out.rotation.z = q.z(); out.rotation.w = q.w();
  return out;
}

Eigen::Isometry3d fromPose(const geometry_msgs::msg::Pose & pose)
{
  Eigen::Isometry3d out = Eigen::Isometry3d::Identity();
  out.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  out.linear() = Eigen::Quaterniond(
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z)
    .normalized().toRotationMatrix();
  return out;
}

double yawOf(const Eigen::Isometry3d & pose)
{
  return std::atan2(pose.rotation()(1, 0), pose.rotation()(0, 0));
}

CloudT::Ptr downsample(const CloudT::ConstPtr & input, double leaf)
{
  CloudT::Ptr out(new CloudT());
  pcl::VoxelGrid<PointT> filter;
  filter.setLeafSize(static_cast<float>(leaf), static_cast<float>(leaf), static_cast<float>(leaf));
  filter.setInputCloud(input);
  filter.filter(*out);
  return out;
}

}  // namespace

class ScanContextGlobalLocalizer final : public rclcpp::Node
{
public:
  ScanContextGlobalLocalizer()
  : Node("scan_context_global_localizer"),
    tf_buffer_(get_clock()), tf_listener_(tf_buffer_), tf_broadcaster_(*this)
  {
    declare_parameter<std::string>("map_path", "/home/jetson/.jszr/map/pose_graph_map_latest.pcd");
    declare_parameter<std::string>("cloud_topic", "/body_points");
    declare_parameter<std::string>("map_cloud_topic", "/map_points_3d");
    declare_parameter<std::string>("odom_frame", "odom");
    declare_parameter<std::string>("sensor_frame", "sensor");
    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<std::string>("map_frame", "map");
    declare_parameter<std::string>("initial_pose_topic", "/initialpose");
    declare_parameter<double>("map_tile_spacing_m", 4.0);
    declare_parameter<double>("tile_radius_m", 9.0);
    declare_parameter<double>("scan_context_max_radius_m", 9.0);
    declare_parameter<double>("map_z_offset_m", 0.35);
    declare_parameter<double>("map_ground_height_quantile", 0.05);
    declare_parameter<double>("scan_rate_hz", 1.0);
    declare_parameter<int>("scan_context_rings", 20);
    declare_parameter<int>("scan_context_sectors", 60);
    declare_parameter<double>("scan_context_min_radius_m", 0.5);
    declare_parameter<double>("scan_context_height_offset_m", 2.0);
    declare_parameter<double>("scan_context_max_height_m", 6.0);
    declare_parameter<double>("scan_context_max_score", 0.30);
    declare_parameter<double>("scan_context_min_margin", 0.015);
    declare_parameter<int>("verify_candidates", 5);
    declare_parameter<int>("max_scan_points", 14000);
    declare_parameter<double>("scan_voxel_m", 0.18);
    declare_parameter<double>("map_voxel_m", 0.10);
    declare_parameter<double>("gicp_resolution_m", 0.50);
    declare_parameter<double>("gicp_max_correspondence_m", 2.0);
    declare_parameter<double>("gicp_max_fitness", 0.25);
    declare_parameter<double>("gicp_min_inlier_ratio", 0.35);
    declare_parameter<int>("gicp_iterations", 64);
    declare_parameter<double>("registration_voxel_m", 0.25);
    declare_parameter<double>("registration_translation_epsilon_m", 0.005);
    declare_parameter<double>("registration_rotation_epsilon_rad", 0.005);
    declare_parameter<double>("registration_inlier_distance_m", 0.5);
    declare_parameter<double>("global_search_budget_s", 0.8);
    declare_parameter<double>("global_fitness_margin", 0.02);
    declare_parameter<int>("global_candidate_count", 128);
    declare_parameter<double>("global_max_fitness", 0.12);
    declare_parameter<double>("global_min_inlier_ratio", 0.65);
    declare_parameter<bool>("use_cuda", true);
    declare_parameter<std::string>("cuda_search_method", "GPU_BRUTEFORCE");
    declare_parameter<bool>("cpu_refine_on_cuda_failure", true);
    declare_parameter<double>("tracking_search_radius_m", 10.0);
    declare_parameter<double>("correction_smoothing", 0.35);
    declare_parameter<double>("max_tracking_translation_correction_m", 1.5);
    declare_parameter<double>("max_tracking_rotation_correction_deg", 35.0);
    declare_parameter<int>("max_tracking_failures_before_global_search", 5);
    declare_parameter<int>("max_tracking_failures_before_lost", 30);
    declare_parameter<bool>("enable_automatic_global_recovery", false);
    declare_parameter<double>("max_global_recovery_translation_m", 3.0);
    declare_parameter<double>("max_global_recovery_rotation_deg", 35.0);
    declare_parameter<int>("global_confirmation_scans", 3);
    declare_parameter<double>("global_confirmation_translation_m", 0.35);
    declare_parameter<double>("global_confirmation_rotation_deg", 12.0);
    // RViz's initial pose is a search centre, not a final map pose.  The
    // current scan is matched against the saved map inside this radius.
    declare_parameter<double>("manual_initial_search_radius_m", 2.0);
    declare_parameter<double>("manual_initial_max_correction_m", 2.2);
    declare_parameter<double>("manual_initial_max_correction_deg", 75.0);

    map_frame_ = get_parameter("map_frame").as_string();
    odom_frame_ = get_parameter("odom_frame").as_string();
    sensor_frame_ = get_parameter("sensor_frame").as_string();
    base_frame_ = get_parameter("base_frame").as_string();
    scan_rate_hz_ = std::max(0.2, get_parameter("scan_rate_hz").as_double());
    rings_ = std::max(4, static_cast<int>(get_parameter("scan_context_rings").as_int()));
    sectors_ = std::max(12, static_cast<int>(get_parameter("scan_context_sectors").as_int()));
    map_radius_ = get_parameter("tile_radius_m").as_double();
    sc_radius_ = get_parameter("scan_context_max_radius_m").as_double();
    min_radius_ = get_parameter("scan_context_min_radius_m").as_double();
    height_offset_ = get_parameter("scan_context_height_offset_m").as_double();
    max_height_ = get_parameter("scan_context_max_height_m").as_double();
    scan_voxel_ = get_parameter("scan_voxel_m").as_double();
    map_voxel_ = get_parameter("map_voxel_m").as_double();
    map_path_ = get_parameter("map_path").as_string();
    use_cuda_ = get_parameter("use_cuda").as_bool();
#ifndef USE_FAST_VGICP_CUDA
    if (use_cuda_) {throw std::runtime_error("use_cuda=true but binary was built without USE_FAST_VGICP_CUDA");}
#endif
    RCLCPP_INFO(get_logger(), "registration backend requested: %s", use_cuda_ ? "CUDA FastVGICP" : "CPU FastGICP");

    loadMapAndBuildIndex();

    auto cloud_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      get_parameter("map_cloud_topic").as_string(), map_qos);
    for (int i = 0; i < 8; ++i) {
      map_chunk_pubs_.push_back(create_publisher<sensor_msgs::msg::PointCloud2>(
        get_parameter("map_cloud_topic").as_string() + "/chunk_" + std::to_string(i), map_qos));
    }
    status_pub_ = create_publisher<std_msgs::msg::String>("~/status", 10);
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      get_parameter("cloud_topic").as_string(), cloud_qos,
      std::bind(&ScanContextGlobalLocalizer::cloudCallback, this, std::placeholders::_1));
    initial_pose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      get_parameter("initial_pose_topic").as_string(), rclcpp::QoS(10).reliable(),
      std::bind(&ScanContextGlobalLocalizer::initialPoseCallback, this, std::placeholders::_1));
    map_timer_ = create_wall_timer(std::chrono::seconds(2), [this]() {
      sensor_msgs::msg::PointCloud2 msg;
      pcl::toROSMsg(*map_, msg);
      msg.header.frame_id = map_frame_;
      msg.header.stamp = now();
      map_pub_->publish(msg);
      for (std::size_t shard = 0; shard < map_chunk_pubs_.size(); ++shard) {
        CloudT part;
        for (std::size_t i = shard; i < map_->size(); i += map_chunk_pubs_.size()) {
          part.push_back(map_->points[i]);
        }
        sensor_msgs::msg::PointCloud2 chunk;
        pcl::toROSMsg(part, chunk);
        chunk.header = msg.header;
        map_chunk_pubs_[shard]->publish(chunk);
      }
      std_msgs::msg::String state;
      {std::lock_guard<std::mutex> lock(pose_mutex_);
        state.data = localized_ ? "localized" : "searching";}
      status_pub_->publish(state);
    });
    tf_timer_ = create_wall_timer(std::chrono::milliseconds(33), [this]() {publishMapOdom();});
    worker_ = std::thread(&ScanContextGlobalLocalizer::workerLoop, this);
    RCLCPP_INFO(get_logger(), "loaded %zu map points and %zu Scan Context tiles from %s",
      map_->size(), tiles_.size(), map_path_.c_str());
  }

  ~ScanContextGlobalLocalizer() override
  {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      stopping_ = true;
      pending_cloud_.reset();
    }
    queue_cv_.notify_all();
    if (worker_.joinable()) {worker_.join();}
  }

private:
  void initialPoseCallback(
    geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg)
  {
    std::string frame = msg->header.frame_id;
    while (!frame.empty() && frame.front() == '/') {frame.erase(frame.begin());}
    if (frame != map_frame_) {
      RCLCPP_WARN(get_logger(),
        "reject /initialpose in frame '%s'; expected '%s'", frame.c_str(), map_frame_.c_str());
      return;
    }

    Eigen::Isometry3d odom_from_sensor;
    Eigen::Isometry3d sensor_from_base;
    try {
      const auto stamp = rclcpp::Time(msg->header.stamp);
      geometry_msgs::msg::TransformStamped odom_tf;
      if (stamp.nanoseconds() > 0) {
        odom_tf = tf_buffer_.lookupTransform(
          odom_frame_, sensor_frame_, stamp, rclcpp::Duration::from_seconds(0.20));
      } else {
        odom_tf = tf_buffer_.lookupTransform(odom_frame_, sensor_frame_, tf2::TimePointZero);
      }
      odom_from_sensor = tf2::transformToEigen(odom_tf);
      const auto sensor_base_tf = tf_buffer_.lookupTransform(
        sensor_frame_, base_frame_, rclcpp::Time(0), rclcpp::Duration::from_seconds(0.20));
      sensor_from_base = tf2::transformToEigen(sensor_base_tf);
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "cannot apply /initialpose yet: %s", e.what());
      return;
    }

    // RViz publishes a coarse map->base_link hypothesis. Convert it to a
    // map->odom prior, but do not publish it yet: the next scan must confirm
    // it with local GICP first.
    const Eigen::Isometry3d map_from_base = fromPose(msg->pose.pose);
    const Eigen::Isometry3d map_from_sensor = map_from_base * sensor_from_base.inverse();
    const Eigen::Isometry3d map_from_odom = map_from_sensor * odom_from_sensor.inverse();
    std::uint64_t revision = 0;
    {
      std::lock_guard<std::mutex> lock(pose_mutex_);
      pending_manual_map_odom_ = map_from_odom;
      pending_global_pose_.reset();
      pending_global_confirmations_ = 0;
      revision = ++pose_revision_;
    }
    RCLCPP_INFO(get_logger(),
      "manual /initialpose queued for local GICP: map->base_link x=%.2f y=%.2f yaw=%.1f deg; search radius=%.1f m; revision=%llu",
      map_from_base.translation().x(), map_from_base.translation().y(),
      yawOf(map_from_base) * 180.0 / kPi,
      get_parameter("manual_initial_search_radius_m").as_double(),
      static_cast<unsigned long long>(revision));
  }

  void loadMapAndBuildIndex()
  {
    map_.reset(new CloudT());
    const std::string ext = std::filesystem::path(map_path_).extension().string();
    const int result = ext == ".pcd" ? pcl::io::loadPCDFile(map_path_, *map_) :
      pcl::io::loadPLYFile(map_path_, *map_);
    if (result < 0 || map_->empty()) {
      throw std::runtime_error("cannot load point map: " + map_path_);
    }
    map_ = downsample(map_, map_voxel_);
    registration_map_ = downsample(map_, get_parameter("registration_voxel_m").as_double());
    map_->width = map_->size(); map_->height = 1; map_->is_dense = false;
    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(map_);
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*map_, min_pt, max_pt);
    map_tree_.setInputCloud(map_);
    const double spacing = std::max(2.0, get_parameter("map_tile_spacing_m").as_double());
    const double r2 = map_radius_ * map_radius_;
    for (double x = std::floor(min_pt.x() / spacing) * spacing;
      x <= max_pt.x(); x += spacing)
    {
      for (double y = std::floor(min_pt.y() / spacing) * spacing;
        y <= max_pt.y(); y += spacing)
      {
        PointT center; center.x = x; center.y = y; center.z = 0.0f;
        std::vector<int> ids; std::vector<float> ds;
        if (tree.radiusSearch(center, map_radius_, ids, ds) < 100) {continue;}
        CloudT local;
        local.reserve(ids.size());
        std::vector<float> heights;
        heights.reserve(ids.size());
        for (int id : ids) {
          const auto & p = map_->points[id];
          PointT centered = p;
          centered.x -= static_cast<float>(x);
          centered.y -= static_cast<float>(y);
          local.push_back(centered);
          heights.push_back(p.z);
        }
        // Wall/ceiling returns dominate dense maps. Their median is not ground.
        const size_t ground_index = static_cast<size_t>(std::clamp(
          get_parameter("map_ground_height_quantile").as_double(), 0.0, 0.25) * (heights.size() - 1));
        std::nth_element(heights.begin(), heights.begin() + ground_index, heights.end());
        Tile tile;
        tile.center = Eigen::Vector3f(static_cast<float>(x), static_cast<float>(y),
          heights[ground_index] + static_cast<float>(get_parameter("map_z_offset_m").as_double()));
        tile.descriptor = makeDescriptor(local, tile.center.z());
        if (tile.descriptor.points >= 100) {tiles_.push_back(std::move(tile));}
      }
    }
    if (tiles_.empty()) {throw std::runtime_error("map produced no usable Scan Context tiles");}
  }

  Descriptor makeDescriptor(const CloudT & cloud, float center_z) const
  {
    Descriptor d;
    d.sectors = Eigen::MatrixXf::Zero(rings_, sectors_);
    d.rings = Eigen::VectorXf::Zero(rings_);
    const double range = sc_radius_ - min_radius_;
    for (const auto & p : cloud.points) {
      const double r = std::hypot(p.x, p.y);
      if (r < min_radius_ || r > sc_radius_) {continue;}
      double theta = std::atan2(p.y, p.x);
      if (theta < 0.0) {theta += 2.0 * kPi;}
      const int ring = std::clamp(static_cast<int>((r - min_radius_) / range * rings_), 0, rings_ - 1);
      const int sector = std::clamp(static_cast<int>(theta / (2.0 * kPi) * sectors_), 0, sectors_ - 1);
      const float h = std::clamp(p.z - center_z + static_cast<float>(height_offset_), 0.0f,
        static_cast<float>(max_height_));
      d.sectors(ring, sector) = std::max(d.sectors(ring, sector), h);
      d.points++;
    }
    for (int r = 0; r < rings_; ++r) {d.rings(r) = d.sectors.row(r).mean();}
    return d;
  }

  std::vector<std::pair<double, int>> descriptorScores(const Descriptor & query, const Descriptor & ref) const
  {
    std::vector<std::pair<double, int>> scores;
    for (int shift = 0; shift < sectors_; ++shift) {
      double sum = 0.0; int used = 0;
      for (int sector = 0; sector < sectors_; ++sector) {
        const auto a = query.sectors.col(sector);
        const auto b = ref.sectors.col((sector + shift) % sectors_);
        const double an = a.norm(), bn = b.norm();
        if (an < 1e-6 || bn < 1e-6) {continue;}
        sum += a.dot(b) / (an * bn); ++used;
      }
      if (used > 0) {scores.emplace_back(1.0 - sum / used, shift);}
    }
    std::sort(scores.begin(), scores.end());
    std::vector<std::pair<double, int>> peaks;
    for (const auto & score : scores) {
      if (peaks.empty()) {peaks.push_back(score); continue;}
      const int distance = std::abs(score.second - peaks.front().second);
      if (std::min(distance, sectors_ - distance) >= std::max(1, sectors_ / 6)) {
        peaks.push_back(score); break;
      }
    }
    return peaks;
  }

  void cloudCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    const auto stamp = rclcpp::Time(msg->header.stamp);
    if (last_queued_stamp_.nanoseconds() != 0 &&
      (stamp - last_queued_stamp_).seconds() < 1.0 / scan_rate_hz_) {return;}
    last_queued_stamp_ = stamp;
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      pending_cloud_ = msg;  // bounded latest-only queue; sensor callback never waits on CUDA.
    }
    queue_cv_.notify_one();
  }

  void workerLoop()
  {
    while (true) {
      sensor_msgs::msg::PointCloud2::ConstSharedPtr msg;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this]() {return stopping_ || static_cast<bool>(pending_cloud_);});
        if (stopping_) {return;}
        msg = std::move(pending_cloud_);
      }
      try {processScan(msg);} catch (const std::exception & e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "localization scan rejected: %s", e.what());
      }
    }
  }

  void processScan(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
  {
    const auto scan_started = std::chrono::steady_clock::now();
    struct Timer {
      ScanContextGlobalLocalizer * node;
      std::chrono::steady_clock::time_point start;
      ~Timer() {RCLCPP_INFO_THROTTLE(node->get_logger(), *node->get_clock(), 3000,
        "registration scan processing %.1f ms", 1000.0 * std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count());}
    } timer{this, scan_started};
    auto raw = CloudT::Ptr(new CloudT()); pcl::fromROSMsg(*msg, *raw);
    CloudT::Ptr scan(new CloudT());
    scan->reserve(raw->size());
    for (const auto & p : raw->points) {
      if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
        std::hypot(p.x, p.y) >= 0.5 && std::hypot(p.x, p.y) <= sc_radius_)
      {scan->push_back(p);}
    }
    scan = downsample(scan, scan_voxel_);
    const int max_scan_points = static_cast<int>(get_parameter("max_scan_points").as_int());
    if (max_scan_points > 0 && static_cast<int>(scan->size()) > max_scan_points) {
      CloudT::Ptr limited(new CloudT());
      const std::size_t stride = static_cast<std::size_t>(std::ceil(
        static_cast<double>(scan->size()) / max_scan_points));
      for (std::size_t i = 0; i < scan->size(); i += stride) {
        limited->push_back(scan->points[i]);
      }
      scan = limited;
    }
    if (scan->size() < 150) {return;}

    Eigen::Isometry3d odom_from_sensor;
    try {
      const auto tf = tf_buffer_.lookupTransform(
        odom_frame_, msg->header.frame_id.empty() ? sensor_frame_ : msg->header.frame_id,
        rclcpp::Time(msg->header.stamp), rclcpp::Duration::from_seconds(0.12));
      odom_from_sensor = tf2::transformToEigen(tf);
    } catch (const std::exception &) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "waiting for odom->sensor TF at scan timestamp");
      return;
    }

    std::optional<Eigen::Isometry3d> map_from_sensor;
    bool currently_localized = false;
    bool recovery_blocked = false;
    Eigen::Isometry3d predicted_map_sensor = Eigen::Isometry3d::Identity();
    std::optional<Eigen::Isometry3d> manual_map_odom;
    std::optional<Eigen::Isometry3d> confirmation_prior;
    std::uint64_t pose_revision = 0;
    { 
      std::lock_guard<std::mutex> lock(pose_mutex_);
      currently_localized = localized_;
      recovery_blocked = recovery_blocked_;
      pose_revision = pose_revision_;
      manual_map_odom = pending_manual_map_odom_;
      if (pending_global_pose_) {confirmation_prior = *pending_global_pose_ * odom_from_sensor;}
      if (currently_localized) {predicted_map_sensor = map_from_odom_ * odom_from_sensor;}
    }
    if (search_revision_ != pose_revision) {
      search_revision_ = pose_revision;
      search_candidates_.clear(); search_results_.clear(); global_candidate_cursor_ = 0;
      manual_candidate_cursor_ = 0;
    }
    if (manual_map_odom) {
      // Use the current odometry sample so a click-to-scan delay does not
      // turn the manual pose into a stale sensor position.
      const Eigen::Isometry3d manual_guess = *manual_map_odom * odom_from_sensor;
      Eigen::Isometry3d aligned;
      double fitness = std::numeric_limits<double>::infinity();
      if (manualInitialSearch(scan, manual_guess, aligned, fitness)) {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        if (pose_revision_ == pose_revision && pending_manual_map_odom_) {
          map_from_odom_ = aligned * odom_from_sensor.inverse();
          pending_manual_map_odom_.reset();
          localized_ = true;
          recovery_blocked_ = false;
          tracking_failures_ = 0;
          RCLCPP_INFO(get_logger(),
            "manual local GICP accepted: correction=%.2f m fitness=%.4f; map->odom x=%.2f y=%.2f",
            (aligned.translation() - manual_guess.translation()).norm(), fitness,
            map_from_odom_.translation().x(), map_from_odom_.translation().y());
        }
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "manual pose GICP not converged inside %.1f m; retaining previous localization",
          get_parameter("manual_initial_search_radius_m").as_double());
      }
      return;
    }
    if (recovery_blocked) {return;}
    if (currently_localized) {
      map_from_sensor = track(scan, odom_from_sensor);
      if (map_from_sensor) {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        tracking_failures_ = 0;
      } else {
        int failures = 0;
        {
          std::lock_guard<std::mutex> lock(pose_mutex_);
          if (pose_revision_ != pose_revision) {return;}
          failures = ++tracking_failures_;
        }
        const int search_after = std::max(1, static_cast<int>(
          get_parameter("max_tracking_failures_before_global_search").as_int()));
        const bool automatic_recovery = get_parameter(
          "enable_automatic_global_recovery").as_bool();
        const int lost_after = static_cast<int>(get_parameter("max_tracking_failures_before_lost").as_int());
        if (failures >= lost_after) {
          std::lock_guard<std::mutex> lock(pose_mutex_);
          if (pose_revision_ != pose_revision) {return;}
          localized_ = false;
          recovery_blocked_ = !automatic_recovery;
          pending_global_pose_.reset();
          pending_global_confirmations_ = 0;
          RCLCPP_WARN(get_logger(), "localization lost after %d tracking failures; stop publishing map->odom", failures);
        }
        if (!automatic_recovery || failures < search_after) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
            "tracking rejected (failure %d/%d, automatic recovery %s)",
            failures, search_after, automatic_recovery ? "enabled" : "disabled");
          return;
        }
        map_from_sensor = globalSearch(scan, predicted_map_sensor, odom_from_sensor);
      }
    } else {
      // Fast-LIO's odom frame has an arbitrary local origin.  It must never
      // be interpreted as the map origin during startup: doing so creates a
      // false local prior and can place the robot near (0, 0) of the saved
      // map.  Startup and recovery therefore use Scan Context + bounded GICP
      // only.  A local predicted prior is used exclusively while tracking
      // is already established (the branch above).
      // Confirm the same geometric hypothesis against a fresh scan instead
      // of starting another expensive global search that may select a new tile.
      if (confirmation_prior) {
        Eigen::Isometry3d aligned; double fitness;
        if (registerCloud(scan, registration_map_, *confirmation_prior, aligned, fitness, false, false, true)) {
          map_from_sensor = aligned;
        }
      } else {
        map_from_sensor = globalSearch(scan, std::nullopt, odom_from_sensor);
      }
    }
    if (!map_from_sensor) {
      std::lock_guard<std::mutex> lock(pose_mutex_);
      if (pose_revision_ != pose_revision) {return;}
      const int lost_after = std::max(1, static_cast<int>(
        get_parameter("max_tracking_failures_before_lost").as_int()));
      if (!currently_localized || tracking_failures_ >= lost_after) {
        localized_ = false;
        pending_global_pose_.reset();
        pending_global_confirmations_ = 0;
      }
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "no Scan Context + GICP localization accepted; searching");
      return;
    }
    const Eigen::Isometry3d candidate_map_odom = *map_from_sensor * odom_from_sensor.inverse();
    {
      std::lock_guard<std::mutex> lock(pose_mutex_);
      if (pose_revision_ != pose_revision) {
        RCLCPP_INFO(get_logger(), "discard registration result from before manual /initialpose");
        return;
      }
      if (!localized_) {
        const int required = std::max(1, static_cast<int>(
          get_parameter("global_confirmation_scans").as_int()));
        const double max_translation = get_parameter(
          "global_confirmation_translation_m").as_double();
        const double max_rotation = get_parameter(
          "global_confirmation_rotation_deg").as_double() * kPi / 180.0;
        bool consistent = false;
        if (pending_global_pose_) {
          const Eigen::Isometry3d delta = pending_global_pose_->inverse() * candidate_map_odom;
          consistent = delta.translation().norm() <= max_translation &&
            Eigen::AngleAxisd(delta.rotation()).angle() <= max_rotation;
        }
        if (!consistent) {
          pending_global_pose_ = candidate_map_odom;
          pending_global_confirmations_ = 1;
        } else {
          pending_global_pose_ = candidate_map_odom;
          pending_global_confirmations_++;
        }
        if (pending_global_confirmations_ < required) {
          RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "global registration passed, awaiting confirmation %d/%d",
            pending_global_confirmations_, required);
          return;
        }
        map_from_odom_ = *pending_global_pose_;
        pending_global_pose_.reset();
        pending_global_confirmations_ = 0;
        localized_ = true;
        tracking_failures_ = 0;
      } else if (get_parameter("correction_smoothing").as_double() < 0.999) {
        const double alpha = std::clamp(get_parameter("correction_smoothing").as_double(), 0.01, 1.0);
        const Eigen::Quaterniond qa(map_from_odom_.rotation()), qb(candidate_map_odom.rotation());
        map_from_odom_.translation() = (1.0 - alpha) * map_from_odom_.translation() + alpha * candidate_map_odom.translation();
        map_from_odom_.linear() = qa.slerp(alpha, qb).normalized().toRotationMatrix();
      } else {
        map_from_odom_ = candidate_map_odom;
      }
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
      "global localization accepted; map->odom x=%.2f y=%.2f", candidate_map_odom.translation().x(), candidate_map_odom.translation().y());
  }

  std::optional<Eigen::Isometry3d> track(const CloudT::Ptr & scan, const Eigen::Isometry3d & odom_from_sensor)
  {
    Eigen::Isometry3d predicted;
    {std::lock_guard<std::mutex> lock(pose_mutex_); predicted = map_from_odom_ * odom_from_sensor;}
    // Static registration target: cache covariances/voxels across scans.
    auto target = registration_map_;
    if (!target || target->size() < 200) {return std::nullopt;}
    Eigen::Isometry3d aligned;
    double fitness = 0.0;
    // Tracking stays CUDA-only so a transient CUDA failure cannot turn every
    // scan into a CPU workload.  The caller will fall back to the bounded
    // global relocalization path below.
    if (!registerCloud(scan, target, predicted, aligned, fitness, false)) {return std::nullopt;}
    const double dt = (aligned.translation() - predicted.translation()).norm();
    Eigen::AngleAxisd aa(predicted.rotation().transpose() * aligned.rotation());
    if (dt > get_parameter("max_tracking_translation_correction_m").as_double() ||
      std::abs(aa.angle()) > get_parameter("max_tracking_rotation_correction_deg").as_double() * kPi / 180.0)
    {return std::nullopt;}
    return aligned;
  }

  std::optional<Eigen::Isometry3d> globalSearch(
    const CloudT::Ptr & scan,
    const std::optional<Eigen::Isometry3d> & predicted_map_sensor,
    const Eigen::Isometry3d & odom_from_sensor)
  {
    // Complete a candidate round across fresh scans before selecting a pose.
    // Store each hypothesis in map->odom, so motion between batches is handled.
    if (search_candidates_.empty()) {
      const Descriptor query = makeDescriptor(*scan, 0.0f);
      std::vector<Candidate> candidates;
      // Partial live scans and complete map tiles have different ring keys.
      // Rank all tile descriptors instead of discarding the correct place
      // with a ring-key shortlist. Keep two separated heading hypotheses.
      for (std::size_t i = 0; i < tiles_.size(); ++i) {
        for (const auto & peak : descriptorScores(query, tiles_[i].descriptor)) {
          if (peak.first <= get_parameter("scan_context_max_score").as_double()) {
            candidates.push_back({static_cast<int>(i), peak.first, peak.second});
          }
        }
      }
      std::sort(candidates.begin(), candidates.end(), [](const Candidate & a, const Candidate & b) {return a.score < b.score;});
      const size_t limit = static_cast<size_t>(std::max<int64_t>(1, get_parameter("global_candidate_count").as_int()));
      if (candidates.size() > limit) {candidates.resize(limit);}
      if (candidates.empty()) {return std::nullopt;}
      search_candidates_ = std::move(candidates);
      search_results_.clear();
      search_odom_start_ = odom_from_sensor;
      global_candidate_cursor_ = 0;
    }
    auto & candidates = search_candidates_;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(
      get_parameter("global_search_budget_s").as_double());
    const int verify = std::min<int>(std::max(1, static_cast<int>(get_parameter("verify_candidates").as_int())),
      static_cast<int>(candidates.size()) - global_candidate_cursor_);
    int attempted = 0;
    const int offset = global_candidate_cursor_;
    for (int i = 0; i < verify; ++i) {
      if (i > 0 && std::chrono::steady_clock::now() >= deadline) {break;}
      ++attempted;
      const auto & c = candidates[(offset + i) % candidates.size()];
      if (c.score > get_parameter("scan_context_max_score").as_double()) {continue;}
      auto target = registration_map_;
      const double yaw = c.yaw_shift * 2.0 * kPi / sectors_;
      Eigen::Isometry3d guess = Eigen::Isometry3d::Identity();
      guess.translation() = tiles_[c.index].center.cast<double>();
      guess.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
      guess = guess * search_odom_start_.inverse() * odom_from_sensor;
      Eigen::Isometry3d aligned; double fitness = 0.0;
      if (target && registerCloud(scan, target, guess, aligned, fitness, false, true)) {
        if (predicted_map_sensor) {
          const double recovery_translation =
            (aligned.translation() - predicted_map_sensor->translation()).norm();
          const Eigen::AngleAxisd recovery_rotation(
            predicted_map_sensor->rotation().transpose() * aligned.rotation());
          const double recovery_rotation_deg = recovery_rotation.angle() * 180.0 / kPi;
          if (recovery_translation > get_parameter("max_global_recovery_translation_m").as_double() ||
            recovery_rotation_deg > get_parameter("max_global_recovery_rotation_deg").as_double()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
              "drop discontinuous global recovery: translation=%.2f m rotation=%.1f deg",
              recovery_translation, recovery_rotation_deg);
            continue;
          }
        }
        search_results_.emplace_back(fitness, aligned * odom_from_sensor.inverse());
      }
    }
    global_candidate_cursor_ += attempted;
    RCLCPP_INFO(get_logger(), "global candidate round %d/%zu, valid=%zu",
      global_candidate_cursor_, candidates.size(), search_results_.size());
    if (global_candidate_cursor_ < static_cast<int>(candidates.size())) {return std::nullopt;}
    search_candidates_.clear();
    if (search_results_.empty()) {return std::nullopt;}
    std::sort(search_results_.begin(), search_results_.end(),
      [](const auto & a, const auto & b) {return a.first < b.first;});
    const auto best = search_results_.front();
    for (const auto & candidate : search_results_) {
      const auto delta = best.second.inverse() * candidate.second;
      const bool distinct = delta.translation().norm() > get_parameter("global_confirmation_translation_m").as_double() ||
        Eigen::AngleAxisd(delta.rotation()).angle() > get_parameter("global_confirmation_rotation_deg").as_double()*kPi/180.0;
      if (distinct && candidate.first - best.first < get_parameter("global_fitness_margin").as_double()) {
        RCLCPP_WARN(get_logger(), "geometrically ambiguous global candidates; retaining searching state");
        return std::nullopt;
      }
    }
    RCLCPP_INFO(get_logger(), "completed global candidate round; best GICP fitness=%.4f", best.first);
    return best.second * odom_from_sensor;
  }

  bool manualInitialSearch(
    const CloudT::Ptr & scan, const Eigen::Isometry3d & centre,
    Eigen::Isometry3d & best_aligned, double & best_fitness)
  {
    const double radius = std::max(0.5,
      get_parameter("manual_initial_search_radius_m").as_double());
    auto target = registration_map_;
    if (!target || target->size() < 200) {return false;}

    // A small deterministic lattice covers a two-metre click error without
    // opening an unconstrained global search.  Three yaw seeds handle a
    // coarse RViz arrow while GICP supplies the final orientation.
    std::vector<Eigen::Vector2d> offsets;
    for (int ix = -2; ix <= 2; ++ix) {
      for (int iy = -2; iy <= 2; ++iy) {
        const Eigen::Vector2d offset(0.5 * radius * ix, 0.5 * radius * iy);
        if (offset.norm() <= radius + 1e-6) {offsets.push_back(offset);}
      }
    }
    const double yaw_step = 30.0 * kPi / 180.0;
    std::sort(offsets.begin(), offsets.end(), [](const auto & a, const auto & b) {return a.squaredNorm() < b.squaredNorm();});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(
      get_parameter("global_search_budget_s").as_double());
    bool found = false;
    double best_correction = std::numeric_limits<double>::infinity();
    const int seed_count = static_cast<int>(offsets.size()) * 3;
    int attempted = 0;
    for (int i = 0; i < seed_count; ++i) {
        if (i > 0 && std::chrono::steady_clock::now() >= deadline) {break;}
        const int seed = (manual_candidate_cursor_ + i) % seed_count;
        const auto & offset = offsets[seed / 3];
        const int yaw_index = (seed % 3 == 0) ? 0 : (seed % 3 == 1 ? -1 : 1);
        ++attempted;
        Eigen::Isometry3d guess = centre;
        guess.translation().x() += offset.x();
        guess.translation().y() += offset.y();
        guess.linear() = centre.linear() *
          Eigen::AngleAxisd(yaw_index * yaw_step, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        Eigen::Isometry3d aligned;
        double fitness = std::numeric_limits<double>::infinity();
        if (!registerCloud(scan, target, guess, aligned, fitness, false)) {continue;}
        const double correction =
          (aligned.translation() - centre.translation()).norm();
        const Eigen::AngleAxisd correction_rotation(centre.rotation().transpose() * aligned.rotation());
        if (correction > get_parameter("manual_initial_max_correction_m").as_double() ||
          correction_rotation.angle() > get_parameter("manual_initial_max_correction_deg").as_double() * kPi / 180.0)
        {continue;}
        if (!found || fitness < best_fitness) {
          found = true;
          best_fitness = fitness;
          best_aligned = aligned;
          best_correction = correction;
        }
    }
    manual_candidate_cursor_ = (manual_candidate_cursor_ + attempted) % seed_count;
    if (found) {
      RCLCPP_INFO(get_logger(),
        "manual GICP candidate selected correction=%.2f m fitness=%.4f", best_correction, best_fitness);
    }
    return found;
  }

  CloudT::Ptr cropMap(const Eigen::Vector3d & center, double radius)
  {
    PointT p; p.x = center.x(); p.y = center.y(); p.z = center.z();
    std::vector<int> ids; std::vector<float> ds;
    if (map_tree_.radiusSearch(p, radius, ids, ds) <= 0) {return CloudT::Ptr(new CloudT());}
    CloudT::Ptr crop(new CloudT()); crop->reserve(ids.size());
    const int stride = std::max(1, static_cast<int>(std::ceil(static_cast<double>(ids.size()) / 30000.0)));
    for (std::size_t i = 0; i < ids.size(); i += stride) {crop->push_back(map_->points[ids[i]]);}
    return downsample(crop, map_voxel_);
  }

  double registrationInlierRatio(
    const CloudT::ConstPtr & source, const CloudT::ConstPtr & target,
    const Eigen::Isometry3d & transform)
  {
    if (!source || !target || source->empty() || target->empty()) {return 0.0;}
    if (validation_target_ != target) {validation_tree_.setInputCloud(target); validation_target_ = target;}
    auto & tree = validation_tree_;
    const double max_distance = get_parameter("registration_inlier_distance_m").as_double();
    const float max_distance_sq = static_cast<float>(max_distance * max_distance);
    const std::size_t stride = std::max<std::size_t>(1, source->size() / 2000);
    std::size_t tested = 0;
    std::size_t inliers = 0;
    std::vector<int> ids(1);
    std::vector<float> distances(1);
    for (std::size_t i = 0; i < source->size(); i += stride) {
      const auto & p = source->points[i];
      const Eigen::Vector3d q = transform * Eigen::Vector3d(p.x, p.y, p.z);
      PointT query; query.x = static_cast<float>(q.x());
      query.y = static_cast<float>(q.y()); query.z = static_cast<float>(q.z());
      if (tree.nearestKSearch(query, 1, ids, distances) > 0) {
        ++tested;
        if (distances[0] <= max_distance_sq) {++inliers;}
      }
    }
    return tested > 0 ? static_cast<double>(inliers) / static_cast<double>(tested) : 0.0;
  }

  bool registerCloud(const CloudT::Ptr & source, const CloudT::Ptr & target,
    const Eigen::Isometry3d & guess, Eigen::Isometry3d & aligned, double & fitness,
    bool allow_cpu_refine, bool coarse = false, bool require_global_quality = false)
  {
    if (!source || !target || source->size() < 100 || target->size() < 200) {return false;}
    bool converged = false;
    const char * backend = "cpu";
    const bool global_quality = coarse || require_global_quality;
    const double fitness_limit = get_parameter(global_quality ? "global_max_fitness" : "gicp_max_fitness").as_double();
    auto run_cpu = [&]() {
      backend = "cpu";
      if (!cpu_registration_) {cpu_registration_ = std::make_unique<fast_gicp::FastGICP<PointT, PointT>>();}
      auto & gicp = *cpu_registration_;
      gicp.setNumThreads(2); gicp.setCorrespondenceRandomness(12);
      gicp.setMaxCorrespondenceDistance(get_parameter("gicp_max_correspondence_m").as_double());
      gicp.setMaximumIterations(static_cast<int>(get_parameter("gicp_iterations").as_int()));
      gicp.setTransformationEpsilon(get_parameter("registration_translation_epsilon_m").as_double());
      gicp.setRotationEpsilon(get_parameter("registration_rotation_epsilon_rad").as_double());
      gicp.setInputSource(source); gicp.setInputTarget(target);
      CloudT output; gicp.align(output, guess.matrix().cast<float>());
      const Eigen::Matrix4f matrix = gicp.getFinalTransformation();
      if (!matrix.allFinite()) {return false;}
      aligned.matrix() = matrix.cast<double>();
      converged = gicp.hasConverged();
      fitness = gicp.getFitnessScore(std::pow(get_parameter("gicp_max_correspondence_m").as_double(), 2));
      return converged && std::isfinite(fitness) && fitness <= fitness_limit;
    };
#ifdef USE_FAST_VGICP_CUDA
    if (use_cuda_) {
      backend = "cuda";
      auto & engine = coarse ? cuda_global_registration_ : cuda_registration_;
      if (!engine) {engine = std::make_unique<fast_gicp::FastVGICPCuda<PointT, PointT>>();}
      auto & gicp = *engine;
      gicp.setRegularizationMethod(fast_gicp::RegularizationMethod::FROBENIUS);
      gicp.setCorrespondenceRandomness(12);
      gicp.setResolution(coarse ? 1.0 : get_parameter("gicp_resolution_m").as_double());
      gicp.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      const auto cuda_search = get_parameter("cuda_search_method").as_string();
      gicp.setNearestNeighborSearchMethod(
        cuda_search == "GPU_RBF_KERNEL" ? fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL :
        fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
      gicp.setMaxCorrespondenceDistance(get_parameter("gicp_max_correspondence_m").as_double());
      gicp.setMaximumIterations(static_cast<int>(get_parameter("gicp_iterations").as_int()));
      gicp.setTransformationEpsilon(get_parameter("registration_translation_epsilon_m").as_double());
      gicp.setRotationEpsilon(get_parameter("registration_rotation_epsilon_rad").as_double());
      gicp.setInputSource(source); gicp.setInputTarget(target);
      CloudT output; gicp.align(output, guess.matrix().cast<float>());
      const Eigen::Matrix4f matrix = gicp.getFinalTransformation();
      if (!matrix.allFinite()) {return false;}
      aligned.matrix() = matrix.cast<double>();
      converged = gicp.hasConverged();
      fitness = gicp.getFitnessScore(std::pow(get_parameter("gicp_max_correspondence_m").as_double(), 2));
      if ((!converged || !std::isfinite(fitness) || fitness > fitness_limit) &&
        allow_cpu_refine && get_parameter("cpu_refine_on_cuda_failure").as_bool())
      {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
          "CUDA map registration did not pass; using one bounded CPU refinement for global init");
        run_cpu();
      }
    } else
#endif
    {
      run_cpu();
    }
    const double inlier_ratio = (converged && std::isfinite(fitness)) ?
      registrationInlierRatio(source, target, aligned) : 0.0;
    const double min_inlier_ratio = get_parameter(global_quality ? "global_min_inlier_ratio" : "gicp_min_inlier_ratio").as_double();
    const bool accepted = converged && std::isfinite(fitness) &&
      fitness <= fitness_limit && inlier_ratio >= min_inlier_ratio;
    RCLCPP_DEBUG(get_logger(),
      "candidate seed=(%.2f %.2f %.2f %.1f) aligned=(%.2f %.2f %.2f %.1f) conv=%d fitness=%.4f inliers=%.3f",
      guess.translation().x(), guess.translation().y(), guess.translation().z(), yawOf(guess)*180.0/kPi,
      aligned.translation().x(), aligned.translation().y(), aligned.translation().z(), yawOf(aligned)*180.0/kPi,
      converged, fitness, inlier_ratio);
    if (!accepted) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "map registration rejected: backend=%s converged=%d fitness=%.5f/%0.5f inliers=%.2f/%0.2f",
        backend, converged, fitness, fitness_limit,
        inlier_ratio, min_inlier_ratio);
    } else {
      RCLCPP_DEBUG(get_logger(),
        "map registration accepted: backend=%s fitness=%.5f inliers=%.2f",
        backend, fitness, inlier_ratio);
    }
    return accepted;
  }

  void publishMapOdom()
  {
    Eigen::Isometry3d transform;
    {std::lock_guard<std::mutex> lock(pose_mutex_); if (!localized_) {return;} transform = map_from_odom_;}
    geometry_msgs::msg::TransformStamped msg;
    msg.header.stamp = now(); msg.header.frame_id = map_frame_; msg.child_frame_id = odom_frame_;
    msg.transform = transformToMsg(transform); tf_broadcaster_.sendTransform(msg);
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> map_chunk_pubs_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::TimerBase::SharedPtr map_timer_, tf_timer_;
  CloudT::Ptr map_;
  CloudT::Ptr registration_map_;
  int global_candidate_cursor_{0}, manual_candidate_cursor_{0};
  std::uint64_t search_revision_{0};
  std::vector<Candidate> search_candidates_;
  std::vector<std::pair<double, Eigen::Isometry3d>> search_results_;
  Eigen::Isometry3d search_odom_start_{Eigen::Isometry3d::Identity()};
  std::unique_ptr<fast_gicp::FastGICP<PointT, PointT>> cpu_registration_;
#ifdef USE_FAST_VGICP_CUDA
  std::unique_ptr<fast_gicp::FastVGICPCuda<PointT, PointT>> cuda_registration_, cuda_global_registration_;
#endif
  pcl::KdTreeFLANN<PointT> map_tree_, validation_tree_;
  CloudT::ConstPtr validation_target_;
  std::vector<Tile> tiles_;
  std::string map_path_, map_frame_, odom_frame_, sensor_frame_, base_frame_;
  double map_radius_{25.0}, sc_radius_{20.0}, min_radius_{0.5}, height_offset_{2.0}, max_height_{6.0};
  double scan_rate_hz_{1.0}, scan_voxel_{0.18}, map_voxel_{0.2};
  int rings_{20}, sectors_{60};
  bool use_cuda_{true};
  rclcpp::Time last_queued_stamp_{0, 0, RCL_ROS_TIME};
  std::mutex queue_mutex_, pose_mutex_;
  std::condition_variable queue_cv_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr pending_cloud_;
  std::thread worker_;
  bool stopping_{false}, localized_{false}, recovery_blocked_{false};
  int tracking_failures_{0};
  std::optional<Eigen::Isometry3d> pending_global_pose_;
  std::optional<Eigen::Isometry3d> pending_manual_map_odom_;
  int pending_global_confirmations_{0};
  std::uint64_t pose_revision_{0};
  Eigen::Isometry3d map_from_odom_{Eigen::Isometry3d::Identity()};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<ScanContextGlobalLocalizer>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("scan_context_global_localizer"), "%s", e.what());
    rclcpp::shutdown(); return 2;
  }
  rclcpp::shutdown();
  return 0;
}
