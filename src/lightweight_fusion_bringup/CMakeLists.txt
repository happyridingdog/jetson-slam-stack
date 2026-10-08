cmake_minimum_required(VERSION 3.8)
project(lightweight_fusion_bringup)

option(USE_FAST_VGICP_CUDA "Use FastVGICPCuda when the CUDA fast_gicp library is built" ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(nav2_core REQUIRED)
find_package(behaviortree_cpp_v3 REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_smoother REQUIRED)
find_package(nav2_theta_star_planner REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters io kdtree)
find_package(fast_gicp REQUIRED)
find_package(GTSAM REQUIRED)

add_executable(sc_gicp_pose_graph_mapper src/sc_gicp_pose_graph_mapper.cpp)
target_compile_features(sc_gicp_pose_graph_mapper PUBLIC cxx_std_17)
target_include_directories(sc_gicp_pose_graph_mapper PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)
ament_target_dependencies(sc_gicp_pose_graph_mapper
  rclcpp nav_msgs sensor_msgs std_msgs std_srvs geometry_msgs
  pcl_conversions tf2 tf2_eigen tf2_ros fast_gicp
)
target_link_libraries(sc_gicp_pose_graph_mapper ${PCL_LIBRARIES} gtsam)
if(USE_FAST_VGICP_CUDA)
  target_compile_definitions(sc_gicp_pose_graph_mapper PRIVATE USE_FAST_VGICP_CUDA)
endif()

add_executable(scan_context_global_localizer src/scan_context_global_localizer.cpp)
target_compile_features(scan_context_global_localizer PUBLIC cxx_std_17)
target_include_directories(scan_context_global_localizer PRIVATE ${PCL_INCLUDE_DIRS})
ament_target_dependencies(scan_context_global_localizer
  rclcpp sensor_msgs std_msgs geometry_msgs pcl_conversions
  tf2 tf2_eigen tf2_ros fast_gicp
)
target_link_libraries(scan_context_global_localizer ${PCL_LIBRARIES})
if(USE_FAST_VGICP_CUDA)
  target_compile_definitions(scan_context_global_localizer PRIVATE USE_FAST_VGICP_CUDA)
endif()

add_library(exclusive_velocity_controller SHARED src/exclusive_velocity_controller.cpp)
target_compile_features(exclusive_velocity_controller PUBLIC cxx_std_17)
ament_target_dependencies(exclusive_velocity_controller
  geometry_msgs nav_msgs sensor_msgs rclcpp rclcpp_lifecycle tf2 tf2_geometry_msgs
  nav2_core nav2_costmap_2d pluginlib)
target_link_options(exclusive_velocity_controller PRIVATE "-Wl,--no-undefined")
pluginlib_export_plugin_description_file(nav2_core exclusive_velocity_controller.xml)

add_library(stable_path_bt_nodes SHARED src/reset_navigation_path.cpp)
target_compile_features(stable_path_bt_nodes PUBLIC cxx_std_17)
target_compile_definitions(stable_path_bt_nodes PRIVATE BT_PLUGIN_EXPORT)
ament_target_dependencies(stable_path_bt_nodes behaviortree_cpp_v3 nav_msgs)

add_library(dense_path_smoother SHARED src/dense_path_smoother.cpp)
target_compile_features(dense_path_smoother PUBLIC cxx_std_17)
ament_target_dependencies(dense_path_smoother nav2_smoother nav2_core nav_msgs pluginlib)
target_link_options(dense_path_smoother PRIVATE "-Wl,--no-undefined")
pluginlib_export_plugin_description_file(nav2_core dense_path_smoother.xml)

add_library(polyline_theta_star_planner SHARED src/polyline_theta_star_planner.cpp)
target_compile_features(polyline_theta_star_planner PUBLIC cxx_std_17)
ament_target_dependencies(polyline_theta_star_planner nav2_theta_star_planner nav2_core
  nav2_costmap_2d nav_msgs pluginlib)
target_link_options(polyline_theta_star_planner PRIVATE "-Wl,--no-undefined")
pluginlib_export_plugin_description_file(nav2_core polyline_theta_star_planner.xml)

install(DIRECTORY config launch rviz behavior_trees DESTINATION share/${PROJECT_NAME})
install(TARGETS exclusive_velocity_controller dense_path_smoother stable_path_bt_nodes
  polyline_theta_star_planner LIBRARY DESTINATION lib)
install(TARGETS sc_gicp_pose_graph_mapper scan_context_global_localizer
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/superodom_feature_cloud_bridge.py
  scripts/laser_odometry_tf_bridge.py
  scripts/nav_cloud_retimestamp.py
  scripts/navigation_supervisor.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_navigation_odometry test/test_navigation_odometry.py)
  ament_add_pytest_test(test_navigation_cloud test/test_navigation_cloud.py)
  ament_add_gtest(test_footprint_filter test/test_footprint_filter.cpp)
  target_compile_features(test_footprint_filter PUBLIC cxx_std_17)
  target_include_directories(test_footprint_filter PRIVATE src)
  ament_add_gtest(test_path_tracking test/test_path_tracking.cpp)
  target_compile_features(test_path_tracking PUBLIC cxx_std_17)
  target_include_directories(test_path_tracking PRIVATE src)
  ament_add_gtest(test_path_resampling test/test_path_resampling.cpp)
  target_compile_features(test_path_resampling PUBLIC cxx_std_17)
  target_include_directories(test_path_resampling PRIVATE src)
  ament_target_dependencies(test_path_resampling nav_msgs)
  ament_add_gtest(test_collision_projection test/test_collision_projection.cpp)
  target_compile_features(test_collision_projection PUBLIC cxx_std_17)
  target_include_directories(test_collision_projection PRIVATE src)
  ament_target_dependencies(test_collision_projection geometry_msgs nav_msgs nav2_costmap_2d)
endif()

ament_package()
