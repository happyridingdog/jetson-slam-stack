cmake_minimum_required(VERSION 3.5)
project(super_odometry)

set(SUPERODOM_NVCC /home/jetson/.local/cuda-11.5/usr/lib/nvidia-cuda-toolkit/bin/nvcc)
set(SUPERODOM_CUDA_INCLUDE /home/jetson/.local/cuda-11.5/usr/include)
set(SUPERODOM_CUDA_RUNTIME /home/jetson/.local/cuda-11.5/usr/lib/aarch64-linux-gnu/libcudart.so)
set(SUPERODOM_GPU_KNN_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/gpu_batch_knn.cu.o)
add_custom_command(
  OUTPUT ${SUPERODOM_GPU_KNN_OBJECT}
  COMMAND ${CMAKE_COMMAND} -E env
    PATH=/home/jetson/.local/cuda-11.5/usr/lib/nvidia-cuda-toolkit/bin:/home/jetson/.local/cuda-11.5/usr/bin:$ENV{PATH}
    ${SUPERODOM_NVCC} --allow-unsupported-compiler -std=c++14 -arch=sm_87 -Xcompiler=-fPIC -I${SUPERODOM_CUDA_INCLUDE}
    -I${CMAKE_CURRENT_SOURCE_DIR}/include -c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/LidarProcess/gpu_batch_knn.cu
    -o ${SUPERODOM_GPU_KNN_OBJECT}
  DEPENDS src/LidarProcess/gpu_batch_knn.cu
          include/super_odometry/LidarProcess/gpu_batch_knn.hpp
  VERBATIM
)
set_source_files_properties(${SUPERODOM_GPU_KNN_OBJECT} PROPERTIES
  GENERATED TRUE EXTERNAL_OBJECT TRUE)
add_custom_target(superodom_gpu_knn DEPENDS ${SUPERODOM_GPU_KNN_OBJECT})

set(SUPERODOM_GPU_RESIDUALS_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/gpu_residuals.cu.o)
add_custom_command(
  OUTPUT ${SUPERODOM_GPU_RESIDUALS_OBJECT}
  COMMAND ${CMAKE_COMMAND} -E env
    PATH=/home/jetson/.local/cuda-11.5/usr/lib/nvidia-cuda-toolkit/bin:/home/jetson/.local/cuda-11.5/usr/bin:$ENV{PATH}
    ${SUPERODOM_NVCC} --allow-unsupported-compiler -std=c++14 -arch=sm_87 -Xcompiler=-fPIC -I${SUPERODOM_CUDA_INCLUDE}
    -I${CMAKE_CURRENT_SOURCE_DIR}/include -c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/LidarProcess/gpu_residuals.cu
    -o ${SUPERODOM_GPU_RESIDUALS_OBJECT}
  DEPENDS src/LidarProcess/gpu_residuals.cu
          include/super_odometry/LidarProcess/gpu_residuals.hpp
  VERBATIM
)
set_source_files_properties(${SUPERODOM_GPU_RESIDUALS_OBJECT} PROPERTIES
  GENERATED TRUE EXTERNAL_OBJECT TRUE)
add_custom_target(superodom_gpu_residuals DEPENDS ${SUPERODOM_GPU_RESIDUALS_OBJECT})

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++1z")
# Keep Jetson builds responsive and avoid excessive compiler memory use.
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -Wall -g0")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
SET(TBB_INCLUDE_DIR "/usr/include/tbb")
set(SUPERODOM_LOCAL_LIBDIR "/home/jetson/.local/ros2-thirdparty/superodom/usr/lib/aarch64-linux-gnu")
add_compile_options(-Wall -Wextra -Wpedantic -Wunused-parameter)
add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")



# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra )
endif()

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(super_odometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Ceres REQUIRED)
find_package(GTSAM REQUIRED QUIET)
find_package(TBB REQUIRED)
find_package(livox_ros_driver2 REQUIRED)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${TBB_INCLUDE_DIR}
  ${Sophus_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIR}
  /home/jetson/.local/ros2-thirdparty/superodom/opt/ros/humble/include
  /home/jetson/.local/ros2-thirdparty/cvbridge/usr/include/opencv4
  ${pcl_conversions_INCLUDE_DIRS}
  ${livox_ros_driver2_INCLUDE_DIRS}
)

# Link directories
link_directories(
  include
  ${SUPERODOM_LOCAL_LIBDIR}
  ${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
)
add_link_options("-Wl,-rpath-link,${SUPERODOM_LOCAL_LIBDIR}" "-Wl,-rpath,${SUPERODOM_LOCAL_LIBDIR}")

set(GTSAM_LIBRARIES gtsam)


# SuperOdom Library
add_library(SuperOdomLib SHARED
  src/FeatureExtraction/featureExtraction.cpp
  src/ImuPreintegration/imuPreintegration.cpp
  src/LaserMapping/laserMapping.cpp
  src/LaserMapping/lidarOptimization.cpp
  src/LaserMapping/LocalMap.cpp
  src/LidarProcess/LidarSlam.cpp
  ${SUPERODOM_GPU_KNN_OBJECT}
  ${SUPERODOM_GPU_RESIDUALS_OBJECT}
  src/LidarProcess/pose_local_parameterization.cpp
  src/LidarProcess/SE3AbsolutatePoseFactor.cpp
  src/parameter/parameter.cpp
  src/utils/superodom_utils.cpp
)
add_dependencies(SuperOdomLib superodom_gpu_knn superodom_gpu_residuals)

# Set up library include directory
target_include_directories(SuperOdomLib PUBLIC  
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/SuperOdomLib>  
    $<INSTALL_INTERFACE:include/SuperOdomLib>  # <prefix>/include/mylib
)

ament_target_dependencies(SuperOdomLib
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs 
  sensor_msgs
  image_transport
  cv_bridge
  tf2
  tf2_ros
  tf2_geometry_msgs
  super_odometry_msgs
  visualization_msgs
  pcl_conversions
  livox_ros_driver2
)
target_link_libraries(SuperOdomLib
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  gtsam
 ${CERES_LIBRARIES}
 ${SUPERODOM_CUDA_RUNTIME}
)

# IMU PreIntegration node
add_executable(imu_preintegration_node src/imuPreintegration_node.cpp)
ament_target_dependencies(imu_preintegration_node rclcpp nav_msgs sensor_msgs tf2_ros)
target_link_libraries(imu_preintegration_node
    gtsam
    SuperOdomLib
)



# Feature Extraction node
add_executable(feature_extraction_node src/featureExtraction_node.cpp)
ament_target_dependencies(feature_extraction_node rclcpp nav_msgs sensor_msgs super_odometry_msgs livox_ros_driver2)
target_link_libraries(feature_extraction_node 
  ${PCL_LIBRARIES}
  ${TBB_LIBRARIES}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  SuperOdomLib
)

# LiDAR Mapping node
add_executable(laser_mapping_node src/laserMapping_node.cpp)
ament_target_dependencies(laser_mapping_node rclcpp std_msgs nav_msgs sensor_msgs geometry_msgs tf2 tf2_ros)
target_link_libraries(laser_mapping_node
    ${PCL_LIBRARIES}
    ${CERES_LIBRARIES}
    SuperOdomLib
)

# Install Exported Libraries
ament_export_targets(export_super_odometry HAS_LIBRARY_TARGET)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS SuperOdomLib
  EXPORT export_super_odometry
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

ament_export_include_directories(include)
ament_export_libraries(SuperOdomLib)

#Install executables
install(TARGETS
  imu_preintegration_node
  feature_extraction_node
  laser_mapping_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
        DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  enable_testing()
  add_executable(test_gpu_residuals test/test_gpu_residuals.cpp)
  target_link_libraries(test_gpu_residuals SuperOdomLib ${CERES_LIBRARIES})
  add_test(NAME gpu_residuals COMMAND test_gpu_residuals)
  add_executable(test_map_insertion test/test_map_insertion.cpp)
  target_link_libraries(test_map_insertion SuperOdomLib ${CERES_LIBRARIES})
  add_test(NAME map_insertion COMMAND test_map_insertion)
endif()

ament_package()
