#pragma once
#include <cstddef>
#include <vector>

namespace super_odometry {
// Plain double precision structures shared with NVCC; quaternion order is xyzw.
struct GpuFeature {
  int edge;
  double point[3], a[3], b[3], offset;
};
struct GpuEvaluation { double residual[3], jacobian[21]; };
class GpuResiduals {
 public:
  explicit GpuResiduals(const std::vector<GpuFeature>& features);
  ~GpuResiduals();
  GpuResiduals(const GpuResiduals&) = delete;
  GpuResiduals& operator=(const GpuResiduals&) = delete;
  void evaluate(const double* pose, std::vector<GpuEvaluation>& output);
 private:
  GpuFeature* features_{nullptr};
  GpuEvaluation* results_{nullptr};
  std::size_t count_{0};
};
}
