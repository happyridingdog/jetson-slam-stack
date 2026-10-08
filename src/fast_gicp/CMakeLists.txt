cmake_minimum_required(VERSION 3.10.0)
project(fast_gicp)

option(BUILD_VGICP_CUDA "Build GPU-powered VGICP" ON)

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
else()
  add_definitions(-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)
  set(CMAKE_C_FLAGS "-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")
  set(CMAKE_CXX_FLAGS "-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")
endif()

set(CMAKE_BUILD_TYPE "Release")

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
add_definitions(${PCL_DEFINITIONS})

find_package(ament_cmake_ros REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(BUILD_VGICP_CUDA)
  if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR AND EXISTS "/usr/local/cuda")
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
  endif()
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  link_directories(${CUDA_LIBRARY_DIRS})
endif()

###########
## Build ##
###########

add_library(fast_gicp SHARED
  src/fast_gicp/gicp/lsq_registration.cpp
  src/fast_gicp/gicp/fast_gicp.cpp
  src/fast_gicp/gicp/fast_gicp_st.cpp
  src/fast_gicp/gicp/fast_vgicp.cpp
)
target_link_libraries(fast_gicp
  ${PCL_LIBRARIES}
)
target_include_directories(fast_gicp PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
set_target_properties(fast_gicp PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)

###################################
## ament specific configuration ##
###################################
if(ament_cmake_ros_FOUND)
  ament_export_include_directories(include)
  ament_export_libraries(fast_gicp)
  ament_export_dependencies(Eigen3)
  ament_export_dependencies(PCL)

  install(
    TARGETS fast_gicp
    EXPORT fast_gicp
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  install(
    DIRECTORY include/
    DESTINATION include
  )

  ament_package()
endif()

### CUDA ###
if(BUILD_VGICP_CUDA)
  # CUDA 11.5 on JetPack 6 uses the system GCC 11 host compiler.  Force the
  # CUDA translation units to C++14 and allow this supported Jetson/GCC pair;
  # otherwise nvcc 11.5 parses libstdc++'s C++17 std::function incorrectly.
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr;--allow-unsupported-compiler;-std=c++14")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_87,code=sm_87")
  endif()
  add_definitions(-DUSE_VGICP_CUDA)

  cuda_add_library(fast_vgicp_cuda SHARED
    src/fast_gicp/cuda/fast_vgicp_cuda.cu
    src/fast_gicp/cuda/brute_force_knn.cu
    src/fast_gicp/cuda/covariance_estimation.cu
    src/fast_gicp/cuda/covariance_estimation_rbf.cu
    src/fast_gicp/cuda/covariance_regularization.cu
    src/fast_gicp/cuda/gaussian_voxelmap.cu
    src/fast_gicp/cuda/find_voxel_correspondences.cu
    src/fast_gicp/cuda/compute_derivatives.cu
    src/fast_gicp/cuda/compute_mahalanobis.cu
    src/fast_gicp/cuda/ndt_cuda.cu
    src/fast_gicp/cuda/ndt_compute_derivatives.cu
  )
  target_include_directories(fast_vgicp_cuda PRIVATE
    include
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
  )
  set_target_properties(fast_vgicp_cuda PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )
  target_link_libraries(fast_vgicp_cuda ${PCL_LIBRARIES})
  cuda_add_cublas_to_target(fast_vgicp_cuda)

  # add vgicp_cuda to libfast_gicp
  target_sources(fast_gicp PRIVATE
    src/fast_gicp/gicp/fast_vgicp_cuda.cpp
    src/fast_gicp/ndt/ndt_cuda.cpp
  )
  target_link_libraries(fast_gicp
    fast_vgicp_cuda
  )
  add_dependencies(fast_gicp fast_vgicp_cuda)
  if(ament_cmake_ros_FOUND)
    install(
      TARGETS fast_vgicp_cuda
      LIBRARY DESTINATION lib
    )
  endif()
endif()

if(BUILD_TESTING AND BUILD_VGICP_CUDA)
  enable_testing()
  cuda_add_executable(cuda_knn_test test/cuda_knn_test.cu)
  target_include_directories(cuda_knn_test PRIVATE include ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(cuda_knn_test fast_vgicp_cuda)
  add_test(NAME cuda_knn COMMAND cuda_knn_test)
  add_executable(cuda_registration_test test/cuda_registration_test.cpp)
  target_include_directories(cuda_registration_test PRIVATE include ${PCL_INCLUDE_DIRS})
  target_link_libraries(cuda_registration_test fast_gicp ${PCL_LIBRARIES})
  add_test(NAME cuda_registration COMMAND cuda_registration_test)
endif()
