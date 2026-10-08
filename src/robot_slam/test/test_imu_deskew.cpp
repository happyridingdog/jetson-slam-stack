#include <gtest/gtest.h>
#include "process/imu_process.h"

TEST(ImuDeskew, SkippedLidarFramesRetainPropagationAndCompensateEachPointOnce)
{
  esekfom::esekf<state_ikfom,12,input_ikfom> filter;
  double epsilon[23]; std::fill(epsilon,epsilon+23,.001);
  filter.init_dyn_share(get_f,df_dx,df_dw,
    [](state_ikfom &, esekfom::dyn_share_datastruct<double> &){},3,epsilon);
  ImuProcess imu;
  imu.set_acc_cov(robot::slam::Vec3d::Constant(.1));
  imu.set_gyr_cov(robot::slam::Vec3d::Constant(.1));
  robot::slam::CloudPtr out(new robot::slam::PointCloudType);
  robot::slam::MeasureGroup init;
  init.lidar.reset(new robot::slam::PointCloudType);
  init.lidar_beg_time=0.;init.lidar_end_time=.1;
  for(int i=0;i<=20;++i) init.imu.push_back(std::make_shared<robot::slam::ImuMessage>(
    i*.005,robot::slam::Vec3d::Zero(),robot::slam::Vec3d(0,0,robot::slam::G_m_s2)));
  imu.Process(init,filter,out);
  auto state=filter.get_x();state.vel=robot::slam::Vec3d(1,0,0);filter.change_x(state);
  robot::slam::MeasureGroup scan;
  scan.lidar.reset(new robot::slam::PointCloudType);
  scan.lidar_beg_time=.5;scan.lidar_end_time=.6;
  for(float stamp:{0.f,50.f,100.f}) {
    robot::slam::PointType point;point.x=5;point.y=0;point.z=0;point.curvature=stamp;
    scan.lidar->push_back(point);
  }
  // Four skipped lidar scans; their IMU samples are all retained.
  for(int i=21;i<=120;++i) scan.imu.push_back(std::make_shared<robot::slam::ImuMessage>(
    i*.005,robot::slam::Vec3d::Zero(),robot::slam::Vec3d(0,0,robot::slam::G_m_s2)));
  imu.Process(scan,filter,out);
  EXPECT_NEAR(filter.get_x().pos.x(),.5,1e-6);
  ASSERT_EQ(out->size(),3u);
  EXPECT_NEAR(out->at(0).x,4.9,1e-5);
  EXPECT_NEAR(out->at(1).x,4.95,1e-5);
  EXPECT_NEAR(out->at(2).x,5.,1e-5);
}
