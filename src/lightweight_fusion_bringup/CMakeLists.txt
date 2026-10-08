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
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
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

install(DIRECTORY config launch rviz DESTINATION share/${PROJECT_NAME})
install(TARGETS sc_gicp_pose_graph_mapper scan_context_global_localizer
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/superodom_feature_cloud_bridge.py
  scripts/laser_odometry_tf_bridge.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
