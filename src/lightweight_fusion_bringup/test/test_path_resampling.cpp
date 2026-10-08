#include <gtest/gtest.h>
#include <limits>
#include "path_resampling.hpp"
using lightweight_fusion_bringup::resamplePath;
nav_msgs::msg::Path path(std::initializer_list<std::pair<double,double>> points)
{
  nav_msgs::msg::Path p; p.header.frame_id="map";
  for (const auto & xy : points) {
    geometry_msgs::msg::PoseStamped v; v.header=p.header;
    v.pose.position.x=xy.first;v.pose.position.y=xy.second;v.pose.orientation.w=1.;
    p.poses.push_back(v);
  }
  return p;
}
TEST(PathResampling, SparseShortBentPathGetsEnoughSamplesAndExactEndpoints)
{
  auto p=path({{0,0},{.2,.04},{.4,0}});
  p.poses.back().pose.orientation.w=0.;p.poses.back().pose.orientation.z=1.;
  const auto r=resamplePath(p);
  EXPECT_GE(r.poses.size(),33u);
  EXPECT_EQ(r.header,p.header);EXPECT_EQ(r.poses.front(),p.poses.front());
  EXPECT_EQ(r.poses.back(),p.poses.back());
  bool original_corner=false;
  for (std::size_t i=1;i<r.poses.size();++i) {
    const auto & a=r.poses[i-1].pose.position;const auto & b=r.poses[i].pose.position;
    EXPECT_LE(std::hypot(a.x-b.x,a.y-b.y),.05+1e-9);
    original_corner |= r.poses[i]==p.poses[1];
  }
  EXPECT_TRUE(original_corner);
}
TEST(PathResampling, KeepsReversalCusp)
{
  auto p=path({{0,0},{1,0},{0,0}});auto r=resamplePath(p);std::size_t peak=0;
  for(std::size_t i=1;i<r.poses.size();++i)if(r.poses[i].pose.position.x>r.poses[peak].pose.position.x)peak=i;
  EXPECT_EQ(r.poses[peak],p.poses[1]);ASSERT_GT(peak,0u);ASSERT_LT(peak+1,r.poses.size());
  EXPECT_LT(r.poses[peak-1].pose.position.x,1.);EXPECT_LT(r.poses[peak+1].pose.position.x,1.);
}
TEST(PathResampling, KeepsInPlaceRotationAndZeroLength)
{
  auto p=path({{1,1},{1,1}});p.poses.back().pose.orientation.z=1.;p.poses.back().pose.orientation.w=0.;
  EXPECT_EQ(resamplePath(p),p);EXPECT_TRUE(resamplePath(path({})).poses.empty());
}
TEST(PathResampling, NeverMovesStraightGeometry)
{
  auto r=resamplePath(path({{0,0},{1,2}}));
  for(const auto & p:r.poses)EXPECT_NEAR(p.pose.position.y,2*p.pose.position.x,1e-12);
}
TEST(PathResampling, RejectsInvalidAndUnboundedInput)
{
  auto p=path({{0,0},{1,2}});p.poses.back().pose.position.x=std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(resamplePath(p),std::invalid_argument);
  EXPECT_THROW(resamplePath(path({{0,0},{1e6,0}})),std::invalid_argument);
  EXPECT_THROW(resamplePath(path({}),0.),std::invalid_argument);
}
