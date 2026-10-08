#include "super_odometry/LidarProcess/gpu_residuals.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cmath>
namespace super_odometry {
namespace {
void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(std::string("SuperOdom CUDA residuals: ") + cudaGetErrorString(code));
}
struct Pose { double v[7]; };
__device__ void skew(const double* p, double* s) {
  s[0]=0; s[1]=-p[2]; s[2]=p[1]; s[3]=p[2]; s[4]=0;
  s[5]=-p[0]; s[6]=-p[1]; s[7]=p[0]; s[8]=0;
}
__global__ void evaluateFeatures(const GpuFeature* features, GpuEvaluation* results, int count, Pose pose) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=count) return;
  const auto& f=features[i]; auto& out=results[i];
  for(int k=0;k<3;++k) out.residual[k]=0;
  for(int k=0;k<21;++k) out.jacobian[k]=0;
  const double x=pose.v[3],y=pose.v[4],z=pose.v[5],w=pose.v[6];
  const double R[9]={1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
    2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
    2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)};
  double p[3],S[9],D[18]={}; skew(f.point,S);
  for(int r=0;r<3;++r) {
    p[r]=pose.v[r]; D[r*6+r]=1;
    for(int k=0;k<3;++k) p[r]+=R[r*3+k]*f.point[k];
    for(int c=0;c<3;++c) for(int k=0;k<3;++k) D[r*6+3+c]-=R[r*3+k]*S[k*3+c];
  }
  if(f.edge) {
    double u[3],v[3],d[3],B[9]; double norm=0;
    for(int k=0;k<3;++k) {u[k]=p[k]-f.a[k];v[k]=p[k]-f.b[k];d[k]=f.b[k]-f.a[k];norm+=d[k]*d[k];}
    norm=sqrt(norm); skew(d,B);
    out.residual[0]=(u[1]*v[2]-u[2]*v[1])/norm;
    out.residual[1]=(u[2]*v[0]-u[0]*v[2])/norm;
    out.residual[2]=(u[0]*v[1]-u[1]*v[0])/norm;
    for(int r=0;r<3;++r) for(int c=0;c<6;++c)
      for(int k=0;k<3;++k) out.jacobian[r*7+c]+=B[r*3+k]*D[k*6+c]/norm;
  } else {
    out.residual[0]=f.offset;
    for(int k=0;k<3;++k) out.residual[0]+=f.a[k]*p[k];
    for(int c=0;c<6;++c) for(int k=0;k<3;++k) out.jacobian[c]+=f.a[k]*D[k*6+c];
  }
}
}
GpuResiduals::GpuResiduals(const std::vector<GpuFeature>& features):count_(features.size()) {
  if (!count_) return;
  try {
    check(cudaMalloc(reinterpret_cast<void**>(&features_),count_*sizeof(GpuFeature)));
    check(cudaMalloc(reinterpret_cast<void**>(&results_),count_*sizeof(GpuEvaluation)));
    check(cudaMemcpy(features_,features.data(),count_*sizeof(GpuFeature),cudaMemcpyHostToDevice));
  } catch (...) {cudaFree(features_);cudaFree(results_);throw;}
}
GpuResiduals::~GpuResiduals() {cudaFree(features_);cudaFree(results_);}
void GpuResiduals::evaluate(const double* pose,std::vector<GpuEvaluation>& output) {
  output.resize(count_); if(!count_) return;
  Pose p;for(int k=0;k<7;++k)p.v[k]=pose[k];
  evaluateFeatures<<<(count_+127)/128,128>>>(features_,results_,static_cast<int>(count_),p);
  check(cudaGetLastError());
  check(cudaMemcpy(output.data(),results_,count_*sizeof(GpuEvaluation),cudaMemcpyDeviceToHost));
}
}
