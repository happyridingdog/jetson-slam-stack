#pragma once

#include <vector>

namespace super_odometry {

// Exact batched 3-D kNN restricted to the same map voxel as each query.
// Coordinates are packed xyz floats; results use -1 for missing neighbors.
void gpuBatchKnnSameCell(
  const std::vector<float> & queries_xyz,
  const std::vector<int> & query_cells,
  const std::vector<float> & targets_xyz,
  const std::vector<int> & target_cells,
  int k,
  std::vector<int> & output_indices,
  std::vector<float> & output_squared_distances);

}  // namespace super_odometry
