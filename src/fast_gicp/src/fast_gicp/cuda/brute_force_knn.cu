#include <Eigen/Core>

#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

#include <stdexcept>

namespace fast_gicp {
  namespace cuda {

namespace {
  struct neighborsearch_kernel {
    neighborsearch_kernel(int k, const thrust::device_vector<Eigen::Vector3f>& target, thrust::device_vector<thrust::pair<float, int>>& k_neighbors)
        : k(k), num_target_points(target.size()), target_points_ptr(target.data()), k_neighbors_ptr(k_neighbors.data()) {}

    template<typename Tuple>
    __host__ __device__ void operator()(const Tuple& idx_x) const {
      // threadIdx doesn't work because thrust split for_each in two loops
      int idx = thrust::get<0>(idx_x);
      const Eigen::Vector3f& x = thrust::get<1>(idx_x);

      // target points buffer & nn output buffer
      const Eigen::Vector3f* pts = thrust::raw_pointer_cast(target_points_ptr);
      thrust::pair<float, int>* k_neighbors = thrust::raw_pointer_cast(k_neighbors_ptr) + idx * k;

      // A zero-based max heap with exactly k entries per query. Avoid the
      // external one-based queue and its pointer-before-allocation storage.
      for (int i = 0; i < k; ++i) {
        k_neighbors[i] = thrust::make_pair((pts[i] - x).squaredNorm(), i);
        int child = i;
        while (child > 0) {
          const int parent = (child - 1) / 2;
          if (k_neighbors[parent].first >= k_neighbors[child].first) {break;}
          const auto value = k_neighbors[parent];
          k_neighbors[parent] = k_neighbors[child]; k_neighbors[child] = value;
          child = parent;
        }
      }
      for (int i = k; i < num_target_points; ++i) {
        const float distance = (pts[i] - x).squaredNorm();
        if (distance >= k_neighbors[0].first) {continue;}
        const auto value = thrust::make_pair(distance, i);
        int parent = 0;
        while (2 * parent + 1 < k) {
          int child = 2 * parent + 1;
          if (child + 1 < k && k_neighbors[child + 1].first > k_neighbors[child].first) {++child;}
          if (k_neighbors[child].first <= value.first) {break;}
          k_neighbors[parent] = k_neighbors[child]; parent = child;
        }
        k_neighbors[parent] = value;
      }
    }

    const int k;
    const int num_target_points;
    thrust::device_ptr<const Eigen::Vector3f> target_points_ptr;

    thrust::device_ptr<thrust::pair<float, int>> k_neighbors_ptr;
  };

  struct sorting_kernel {
    sorting_kernel(int k, thrust::device_vector<thrust::pair<float, int>>& k_neighbors) : k(k), k_neighbors_ptr(k_neighbors.data()) {}

    __host__ __device__ void operator()(int idx) const {
      // target points buffer & nn output buffer
      thrust::pair<float, int>* k_neighbors = thrust::raw_pointer_cast(k_neighbors_ptr) + idx * k;

      // In-place heapsort; output is in ascending distance order.
      for (int end = k - 1; end > 0; --end) {
        const auto value = k_neighbors[end];
        k_neighbors[end] = k_neighbors[0];
        int parent = 0;
        while (2 * parent + 1 < end) {
          int child = 2 * parent + 1;
          if (child + 1 < end && k_neighbors[child + 1].first > k_neighbors[child].first) {++child;}
          if (k_neighbors[child].first <= value.first) {break;}
          k_neighbors[parent] = k_neighbors[child]; parent = child;
        }
        k_neighbors[parent] = value;
      }
    }

    const int k;
    thrust::device_ptr<thrust::pair<float, int>> k_neighbors_ptr;
  };
}

void brute_force_knn_search(const thrust::device_vector<Eigen::Vector3f>& source, const thrust::device_vector<Eigen::Vector3f>& target, int k, thrust::device_vector<thrust::pair<float, int>>& k_neighbors, bool do_sort=false) {
  if (k < 1 || static_cast<size_t>(k) > target.size()) {throw std::invalid_argument("CUDA KNN requires 1 <= k <= target size");}
  thrust::device_vector<int> d_indices(source.size());
  thrust::sequence(d_indices.begin(), d_indices.end());

  auto first = thrust::make_zip_iterator(thrust::make_tuple(d_indices.begin(), source.begin()));
  auto last = thrust::make_zip_iterator(thrust::make_tuple(d_indices.end(), source.end()));

  // Each query exclusively owns k entries.
  k_neighbors.resize(source.size() * k, thrust::make_pair(-1.0f, -1));
  thrust::for_each(first, last, neighborsearch_kernel(k, target, k_neighbors));

  if(do_sort) {
    thrust::for_each(d_indices.begin(), d_indices.end(), sorting_kernel(k, k_neighbors));
  }
}

  }
} // namespace fast_gicp
