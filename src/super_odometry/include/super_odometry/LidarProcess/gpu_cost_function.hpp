#pragma once
#include "super_odometry/LidarProcess/gpu_residuals.hpp"
#include <ceres/ceres.h>
#include <ceres/evaluation_callback.h>
#include <algorithm>
#include <memory>
#include <iostream>
namespace super_odometry {
// Ceres calls this once before its parallel cost evaluations. The immutable
// result buffer is then shared by all residual blocks, preserving robust losses.
class GpuEvaluationBatch : public ceres::EvaluationCallback {
 public:
  GpuEvaluationBatch(const std::vector<GpuFeature>& features, const double* pose)
    : gpu_(features), pose_(pose) {}
  void PrepareForEvaluation(bool, bool new_point) override {
    if (new_point || !prepared_) {
      try {gpu_.evaluate(pose_, values);valid=true;}
      catch(const std::exception& e) {std::cerr << e.what() << std::endl;valid=false;}
      prepared_=true;
    }
  }
  std::vector<GpuEvaluation> values;
  bool valid{false};
 private:
  GpuResiduals gpu_;
  const double* pose_;
  bool prepared_{false};
};
class GpuFeatureCost : public ceres::CostFunction {
 public:
  GpuFeatureCost(std::shared_ptr<GpuEvaluationBatch> batch, std::size_t index, bool edge)
    : batch_(std::move(batch)), index_(index) {
    set_num_residuals(edge?3:1); mutable_parameter_block_sizes()->push_back(7);
  }
  bool Evaluate(double const* const*, double* residuals, double** jacobians) const override {
    if(!batch_->valid) return false;
    const auto& value=batch_->values.at(index_);
    std::copy_n(value.residual,num_residuals(),residuals);
    if(jacobians && jacobians[0]) std::copy_n(value.jacobian,7*num_residuals(),jacobians[0]);
    return true;
  }
 private:
  std::shared_ptr<GpuEvaluationBatch> batch_;
  std::size_t index_;
};
}
