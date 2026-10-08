#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#include <pcl/common/transforms.h>
#include <random>
#include <chrono>
#include <iostream>
#include <stdexcept>
int main() {
  using P=pcl::PointXYZ;using Cloud=pcl::PointCloud<P>;
  std::mt19937 rng(71);std::uniform_real_distribution<float> random(-5,5);
  Cloud::Ptr target(new Cloud);
  for(int i=0;i<5000;++i){P p;p.x=random(rng);p.y=random(rng);p.z=random(rng)*.4f;
    if(i%3==0)p.x=3.1f;if(i%3==1)p.y=-2.4f;if(i%3==2)p.z=-.37f;target->push_back(p);}
  // Includes the zero voxel, whose hash probe can encounter empty buckets.
  for(int i=0;i<30;++i){P p;p.x=.25f+.01f*i;p.y=.3;p.z=.4;target->push_back(p);}
  fast_gicp::FastVGICPCuda<P,P> registration;
  registration.setRegularizationMethod(fast_gicp::RegularizationMethod::FROBENIUS);
  registration.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
  registration.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
  registration.setCorrespondenceRandomness(12);registration.setResolution(.5);
  registration.setTransformationEpsilon(.005);registration.setRotationEpsilon(.005);
  registration.setInputTarget(target);
  double max_error=0;auto start=std::chrono::steady_clock::now();
  for(int i=0;i<200;++i) {
    Eigen::Isometry3f truth=Eigen::Isometry3f::Identity();truth.translation()<<.2+.001*i,-.1,.03;
    truth.linear()=Eigen::AngleAxisf(.06,Eigen::Vector3f::UnitZ()).toRotationMatrix();
    Cloud::Ptr scan(new Cloud);pcl::transformPointCloud(*target,*scan,truth.inverse().matrix());
    if(i%40==0)registration.setResolution(.6);if(i%40==1)registration.setResolution(.5);
    registration.setInputSource(scan);Cloud aligned;registration.align(aligned);
    const auto estimate=registration.getFinalTransformation();
    const double error=(estimate-truth.matrix()).norm();max_error=std::max(max_error,error);
    if(!registration.hasConverged() || !estimate.allFinite() || error>.06)throw std::runtime_error("CUDA registration accuracy/stability failure");
  }
  std::cout<<"PASS 200 CUDA registrations, persistent target, replaced source, resolution changes, origin voxel; max_transform_error="<<max_error<<" mean_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/200<<std::endl;
}
