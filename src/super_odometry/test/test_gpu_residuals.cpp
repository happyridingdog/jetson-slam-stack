#include "super_odometry/LidarProcess/gpu_cost_function.hpp"
#include "super_odometry/LidarProcess/factor/lidarOptimization.h"
#include "super_odometry/LidarProcess/factor/pose_local_parameterization.h"
#include <random>
#include <chrono>
#include <stdexcept>
using namespace super_odometry;
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
ceres::CostFunction* cpuCost(const GpuFeature& f) {
  Eigen::Vector3d p(f.point), a(f.a), b(f.b);
  if(f.edge) return new EdgeAnalyticCostFunction(p,a,b);
  return new SurfNormAnalyticCostFunction(p,a,f.offset);
}
int main() {
  std::mt19937 rng(92); std::uniform_real_distribution<double> random(-4,4);
  std::vector<GpuFeature> features(4096);
  for(size_t i=0;i<features.size();++i) {
    auto& f=features[i]; f={};f.edge=i%2;
    for(int k=0;k<3;++k){f.point[k]=random(rng);f.a[k]=random(rng);f.b[k]=random(rng);}
    if(!f.edge){Eigen::Map<Eigen::Vector3d> n(f.a);n.normalize();} f.offset=random(rng);
  }
  GpuResiduals gpu(features);std::vector<GpuEvaluation> results;
  std::vector<std::unique_ptr<ceres::CostFunction>> costs;
  for(auto& f:features)costs.emplace_back(cpuCost(f));
  double max_res=0,max_jac=0,max_fd=0;
  for(int sample=0;sample<12;++sample) {
    double pose[7]={random(rng),random(rng),random(rng),0,0,0,1};
    Eigen::Quaterniond q(Eigen::AngleAxisd(random(rng),Eigen::Vector3d(1,2,3).normalized()));
    Eigen::Map<Eigen::Quaterniond> pq(pose+3);pq=q;
    gpu.evaluate(pose,results);
    const double* params[]={pose};
    for(size_t i=0;i<features.size();++i){double r[3],j[21];double* js[]={j};costs[i]->Evaluate(params,r,js);
      for(int k=0;k<costs[i]->num_residuals();++k)max_res=std::max(max_res,std::abs(r[k]-results[i].residual[k]));
      for(int k=0;k<costs[i]->num_residuals()*7;++k)max_jac=std::max(max_jac,std::abs(j[k]-results[i].jacobian[k]));
    }
    // Independent finite differences in the actual six-dimensional pose update.
    std::unique_ptr<ceres::LocalParameterization> local(new PoseLocalParameterization);
    for(int c=0;c<6;++c){double plus[7],minus[7],d[6]={};d[c]=1e-6;local->Plus(pose,d,plus);d[c]=-1e-6;local->Plus(pose,d,minus);
      for(size_t i=0;i<16;++i){double rp[3],rm[3];const double* pp[]={plus};const double* pm[]={minus};costs[i]->Evaluate(pp,rp,nullptr);costs[i]->Evaluate(pm,rm,nullptr);
        for(int r=0;r<costs[i]->num_residuals();++r) max_fd=std::max(max_fd,std::abs((rp[r]-rm[r])/2e-6-results[i].jacobian[r*7+c]));}}
  }
  require(max_res<1e-10 && max_jac<1e-10 && max_fd<1e-7,"GPU residual/Jacobian parity failed");
  double solutions[2][7]={{.1,-.1,.1,0,0,0,1},{.1,-.1,.1,0,0,0,1}};
  double final_cost[2];
  for(int mode=0;mode<2;++mode){auto* pose=solutions[mode];
    std::shared_ptr<GpuEvaluationBatch> batch;
    ceres::Problem::Options po;if(mode){batch=std::make_shared<GpuEvaluationBatch>(features,pose);po.evaluation_callback=batch.get();}
    ceres::Problem problem(po);problem.AddParameterBlock(pose,7,new PoseLocalParameterization);
    for(size_t i=0;i<features.size();++i){auto* cost=mode?static_cast<ceres::CostFunction*>(new GpuFeatureCost(batch,i,features[i].edge)):cpuCost(features[i]);
      problem.AddResidualBlock(cost,new ceres::ScaledLoss(new ceres::TukeyLoss(4.0),0.7,ceres::TAKE_OWNERSHIP),pose);}
    ceres::Solver::Options options;options.max_num_iterations=12;options.num_threads=4;options.linear_solver_type=ceres::DENSE_QR;
    ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);require(summary.IsSolutionUsable(),"Ceres solve failed");final_cost[mode]=summary.final_cost;
    std::cout<<(mode?"GPU":"CPU")<<" solve_ms="<<summary.total_time_in_seconds*1000<<" cost="<<summary.final_cost<<std::endl;
  }
  double pose_error=0;for(int k=0;k<7;++k)pose_error=std::max(pose_error,std::abs(solutions[0][k]-solutions[1][k]));
  require(pose_error<1e-8 && std::abs(final_cost[0]-final_cost[1])<1e-8,"Ceres robust solve parity failed");
  // Stress repeated allocations and changed input/pose, catching asynchronous device failures.
  for(int i=0;i<200;++i){double pose[7]={i*.001,0,0,0,0,0,1};gpu.evaluate(pose,results);}
  std::cout<<"PASS 4096 features, 12 poses, 200 repeated batches; max_res="<<max_res<<" max_jac="<<max_jac<<" finite_difference="<<max_fd<<" pose_error="<<pose_error<<std::endl;
}
