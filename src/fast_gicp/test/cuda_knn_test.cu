#include <fast_gicp/cuda/brute_force_knn.cuh>
#include <thrust/host_vector.h>
#include <algorithm>
#include <random>
#include <iostream>
#include <stdexcept>
int main() {
  std::mt19937 rng(321); std::uniform_real_distribution<float> random(-20,20);
  thrust::host_vector<Eigen::Vector3f> target(1301),source(257);
  for(auto& p:target)p=Eigen::Vector3f(random(rng),random(rng),random(rng));
  for(int i=0;i<source.size();++i)source[i]=i%2 ? target[i*3] : Eigen::Vector3f(random(rng),random(rng),random(rng));
  target[0]=source[0]=Eigen::Vector3f::Zero();
  thrust::device_vector<Eigen::Vector3f> dt=target,ds=source;
  int checked=0;
  for(int k:{1,3,12,20,33})for(bool sorted:{false,true}) {
    thrust::device_vector<thrust::pair<float,int>> result;
    fast_gicp::cuda::brute_force_knn_search(ds,dt,k,result,sorted);
    thrust::host_vector<thrust::pair<float,int>> values=result;
    for(int row=0;row<source.size();++row) {
      std::vector<float> expected;for(auto& p:target)expected.push_back((p-source[row]).squaredNorm());std::sort(expected.begin(),expected.end());
      std::vector<float> got;std::vector<int> ids;
      for(int n=0;n<k;++n){auto p=values[row*k+n];if(p.second<0||p.second>=target.size())throw std::runtime_error("KNN out-of-range index");
        got.push_back((target[p.second]-source[row]).squaredNorm());ids.push_back(p.second);
        if(std::abs(got.back()-p.first)>1e-4*std::max(1.f,got.back()))throw std::runtime_error("KNN distance mismatch");}
      if(sorted && !std::is_sorted(got.begin(),got.end()))throw std::runtime_error("KNN unsorted output");
      std::sort(ids.begin(),ids.end());if(std::adjacent_find(ids.begin(),ids.end())!=ids.end())throw std::runtime_error("KNN duplicate index");
      std::sort(got.begin(),got.end());for(int n=0;n<k;++n)if(std::abs(got[n]-expected[n])>1e-4*std::max(1.f,expected[n]))throw std::runtime_error("KNN wrong neighbor");
      ++checked;
    }
  }
  bool rejected=false;thrust::device_vector<thrust::pair<float,int>> result;
  try{fast_gicp::cuda::brute_force_knn_search(ds,dt,1302,result,false);}catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)throw std::runtime_error("Oversized K was not rejected");
  std::cout<<"PASS "<<checked<<" exact CPU/GPU KNN comparisons; k=1,3,12,20,33 sorted/unsorted, distinct indices, bounds"<<std::endl;
}
