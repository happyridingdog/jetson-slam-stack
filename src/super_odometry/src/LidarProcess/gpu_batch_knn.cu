#include "super_odometry/LidarProcess/gpu_batch_knn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <string>

namespace super_odometry {
namespace {

struct DeviceWorkspace {
  float * queries{nullptr};
  int * query_cells{nullptr};
  float * targets{nullptr};
  int * target_cells{nullptr};
  int * indices{nullptr};
  float * distances{nullptr};
  size_t query_capacity{0};
  size_t query_cell_capacity{0};
  size_t target_capacity{0};
  size_t target_cell_capacity{0};
  size_t result_capacity{0};
  size_t distance_capacity{0};

  ~DeviceWorkspace()
  {
    if (queries) cudaFree(queries);
    if (query_cells) cudaFree(query_cells);
    if (targets) cudaFree(targets);
    if (target_cells) cudaFree(target_cells);
    if (indices) cudaFree(indices);
    if (distances) cudaFree(distances);
  }
};

DeviceWorkspace workspace;
std::mutex workspace_mutex;

void checkCuda(cudaError_t status, const char * operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template<typename T>
void reserveDevice(T * & pointer, size_t & capacity, size_t requested)
{
  if (requested <= capacity) return;
  if (pointer) checkCuda(cudaFree(pointer), "cudaFree");
  pointer = nullptr;
  checkCuda(cudaMalloc(reinterpret_cast<void **>(&pointer), requested * sizeof(T)), "cudaMalloc");
  capacity = requested;
}

__global__ void sameCellKnnKernel(
  const float * queries, const int * query_cells, int query_count,
  const float * targets, const int * target_cells, int target_count,
  int k, int * output_indices, float * output_distances)
{
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= query_count) return;

  const float infinity = __int_as_float(0x7f800000);
  float best_distances[5] = {infinity, infinity, infinity, infinity, infinity};
  int best_indices[5] = {-1, -1, -1, -1, -1};
  const int cell = query_cells[query];
  if (cell >= 0) {
    const float qx = queries[3 * query];
    const float qy = queries[3 * query + 1];
    const float qz = queries[3 * query + 2];
    for (int target = 0; target < target_count; ++target) {
      if (target_cells[target] != cell) continue;
      const float dx = qx - targets[3 * target];
      const float dy = qy - targets[3 * target + 1];
      const float dz = qz - targets[3 * target + 2];
      const float distance = dx * dx + dy * dy + dz * dz;
      if (distance >= best_distances[k - 1]) continue;
      int insert = k - 1;
      while (insert > 0 && distance < best_distances[insert - 1]) {
        best_distances[insert] = best_distances[insert - 1];
        best_indices[insert] = best_indices[insert - 1];
        --insert;
      }
      best_distances[insert] = distance;
      best_indices[insert] = target;
    }
  }
  for (int neighbor = 0; neighbor < k; ++neighbor) {
    output_indices[query * k + neighbor] = best_indices[neighbor];
    output_distances[query * k + neighbor] = best_distances[neighbor];
  }
}

}  // namespace

void gpuBatchKnnSameCell(
  const std::vector<float> & queries_xyz,
  const std::vector<int> & query_cells,
  const std::vector<float> & targets_xyz,
  const std::vector<int> & target_cells,
  int k,
  std::vector<int> & output_indices,
  std::vector<float> & output_squared_distances)
{
  const size_t query_count = query_cells.size();
  const size_t target_count = target_cells.size();
  if (queries_xyz.size() != query_count * 3 || targets_xyz.size() != target_count * 3 ||
      k < 1 || k > 5) {
    throw std::invalid_argument("invalid GPU kNN input dimensions");
  }
  output_indices.assign(query_count * static_cast<size_t>(k), -1);
  output_squared_distances.assign(query_count * static_cast<size_t>(k),
                                   std::numeric_limits<float>::infinity());
  if (query_count == 0 || target_count == 0) return;

  std::lock_guard<std::mutex> lock(workspace_mutex);
  reserveDevice(workspace.queries, workspace.query_capacity, query_count * 3);
  reserveDevice(workspace.query_cells, workspace.query_cell_capacity, query_count);
  reserveDevice(workspace.targets, workspace.target_capacity, target_count * 3);
  reserveDevice(workspace.target_cells, workspace.target_cell_capacity, target_count);
  reserveDevice(workspace.indices, workspace.result_capacity, query_count * static_cast<size_t>(k));
  reserveDevice(workspace.distances, workspace.distance_capacity, query_count * static_cast<size_t>(k));

  checkCuda(cudaMemcpy(workspace.queries, queries_xyz.data(), queries_xyz.size() * sizeof(float),
                       cudaMemcpyHostToDevice), "copy queries to CUDA");
  checkCuda(cudaMemcpy(workspace.query_cells, query_cells.data(), query_cells.size() * sizeof(int),
                       cudaMemcpyHostToDevice), "copy query cells to CUDA");
  checkCuda(cudaMemcpy(workspace.targets, targets_xyz.data(), targets_xyz.size() * sizeof(float),
                       cudaMemcpyHostToDevice), "copy map points to CUDA");
  checkCuda(cudaMemcpy(workspace.target_cells, target_cells.data(), target_cells.size() * sizeof(int),
                       cudaMemcpyHostToDevice), "copy map cells to CUDA");

  constexpr int threads = 128;
  sameCellKnnKernel<<<static_cast<unsigned int>((query_count + threads - 1) / threads), threads>>>(
    workspace.queries, workspace.query_cells, static_cast<int>(query_count),
    workspace.targets, workspace.target_cells, static_cast<int>(target_count), k,
    workspace.indices, workspace.distances);
  checkCuda(cudaGetLastError(), "launch GPU kNN kernel");
  checkCuda(cudaDeviceSynchronize(), "synchronize GPU kNN");
  checkCuda(cudaMemcpy(output_indices.data(), workspace.indices,
                       output_indices.size() * sizeof(int), cudaMemcpyDeviceToHost),
            "copy kNN indices from CUDA");
  checkCuda(cudaMemcpy(output_squared_distances.data(), workspace.distances,
                       output_squared_distances.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "copy kNN distances from CUDA");
}

}  // namespace super_odometry
