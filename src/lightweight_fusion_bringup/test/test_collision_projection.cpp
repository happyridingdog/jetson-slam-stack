#include <gtest/gtest.h>
#include <limits>
#include "collision_projection.hpp"
#include "polyline_shortcut.hpp"
#include "planner_clearance.hpp"

using lightweight_fusion_bringup::CollisionProjection;
using lightweight_fusion_bringup::Velocity;
using lightweight_fusion_bringup::shortcutPolyline;
using lightweight_fusion_bringup::straightCorridorFree;
using lightweight_fusion_bringup::rotationCorridorFree;
using lightweight_fusion_bringup::orientPolyline;
using lightweight_fusion_bringup::planarYaw;

class CollisionTest : public ::testing::Test
{
protected:
  nav2_costmap_2d::Costmap2D map{400, 400, 0.02, -4.0, -4.0, 0};
  CollisionProjection check;
  std::vector<geometry_msgs::msg::Point> footprint;
  void SetUp() override
  {
    for (const auto & xy : std::vector<std::pair<double, double>>{
        {0.30, 0.225}, {0.30, -0.225}, {-0.30, -0.225}, {-0.30, 0.225}})
    {
      geometry_msgs::msg::Point p; p.x = xy.first; p.y = xy.second;
      footprint.push_back(p);
    }
  }
  void obstacle(double x, double y, unsigned char cost = nav2_costmap_2d::LETHAL_OBSTACLE)
  {
    unsigned int mx, my;
    ASSERT_TRUE(map.worldToMap(x, y, mx, my));
    map.setCost(mx, my, cost);
  }
  bool free(Velocity target, Velocity current = {}, double yaw = 0.0)
  {
    return check.isFree(map, footprint, 0.0, 0.0, yaw, current, target);
  }
};

static nav_msgs::msg::Path route(std::initializer_list<std::pair<double,double>> points)
{
  nav_msgs::msg::Path p;p.header.frame_id="map";
  for (const auto & xy:points) {
    geometry_msgs::msg::PoseStamped pose;pose.pose.position.x=xy.first;
    pose.pose.position.y=xy.second;pose.pose.orientation.w=1.;p.poses.push_back(pose);
  }
  return p;
}

TEST_F(CollisionTest, OpenRouteBecomesOneStraightLegWithoutCurveFitting)
{
  const auto input=route({{0,0},{.8,.4},{1.2,.4},{2,.3}});
  const auto output=shortcutPolyline(input,map,footprint,
    std::chrono::steady_clock::now()+std::chrono::seconds(2));
  ASSERT_EQ(output.poses.size(),2u);
  EXPECT_EQ(output.poses.front(),input.poses.front());
  EXPECT_EQ(output.poses.back(),input.poses.back());
}

TEST_F(CollisionTest, ShortcutsKeepNecessaryCornersAndCheckVehicleSides)
{
  obstacle(1.,.15); // Away from the centre line, inside the swept body.
  const auto input=route({{0,0},{0,1},{2,1},{2,0}});
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  EXPECT_FALSE(straightCorridorFree(map,footprint,input.poses.front().pose.position,
    input.poses.back().pose.position,deadline));
  const auto output=shortcutPolyline(input,map,footprint,deadline);
  ASSERT_GE(output.poses.size(),3u);
  for (size_t i=1;i<output.poses.size();++i) {
    EXPECT_TRUE(straightCorridorFree(map,footprint,output.poses[i-1].pose.position,
      output.poses[i].pose.position,deadline));
  }
  EXPECT_TRUE(shortcutPolyline(input,map,footprint,std::chrono::steady_clock::now()).poses.empty());
}

TEST_F(CollisionTest, InflationIsSoftButRealAndUnknownObstaclesRemainHard)
{
  map.resetMapToValue(0,0,map.getSizeInCellsX(),map.getSizeInCellsY(),253);
  auto p=route({{0,0},{2,0}});
  auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  EXPECT_EQ(shortcutPolyline(p,map,footprint,deadline).poses.size(),2u);
  obstacle(1.,.15);
  EXPECT_TRUE(shortcutPolyline(p,map,footprint,deadline).poses.empty());
  obstacle(1.,.15,nav2_costmap_2d::NO_INFORMATION);
  EXPECT_TRUE(shortcutPolyline(p,map,footprint,deadline).poses.empty());
}

TEST_F(CollisionTest, ClearancePreferenceKeepsRouteAwayFromWallDuringSimplification)
{
  for (double x=-3.;x<3.;x+=.01) {obstacle(x,-.60);obstacle(x,1.0);}
  auto search=lightweight_fusion_bringup::clearanceSearchMap(map,.9,.285);
  lightweight_fusion_bringup::ShortcutPreference preference;
  preference.costs=&search;
  auto input=route({{-2,-.15},{-1,.20},{1,.20},{2,-.15}});
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  ASSERT_TRUE(straightCorridorFree(map,footprint,input.poses.front().pose.position,
    input.poses.back().pose.position,deadline)); // Original code accepted this near-wall line.
  const auto output=shortcutPolyline(input,map,footprint,deadline,&preference);
  ASSERT_GE(output.poses.size(),3u);
  EXPECT_LT(lightweight_fusion_bringup::polylineTraversalCost(output,preference),
    lightweight_fusion_bringup::polylineTraversalCost(route({{-2,-.15},{2,-.15}}),preference));
  EXPECT_EQ(output.poses.front(),input.poses.front());
  EXPECT_EQ(output.poses.back(),input.poses.back());
}

TEST_F(CollisionTest, ClearanceCostPrefersCorridorMiddleAndDoesNotMutatePublicMap)
{
  for (double x=-3.;x<3.;x+=.01) {obstacle(x,-.8);obstacle(x,.8);}
  obstacle(3,3,253);obstacle(3,2,255);
  const auto original=map;
  auto search=lightweight_fusion_bringup::clearanceSearchMap(map,.9,.285);
  lightweight_fusion_bringup::ShortcutPreference preference;preference.costs=&search;
  const double centre=lightweight_fusion_bringup::polylineTraversalCost(route({{-2,0},{2,0}}),preference);
  EXPECT_LT(centre,lightweight_fusion_bringup::polylineTraversalCost(route({{-2,.35},{2,.35}}),preference));
  EXPECT_LT(centre,lightweight_fusion_bringup::polylineTraversalCost(route({{-2,-.35},{2,-.35}}),preference));
  for (unsigned int y=0;y<map.getSizeInCellsY();++y) {
    for (unsigned int x=0;x<map.getSizeInCellsX();++x) {
      EXPECT_EQ(map.getCost(x,y),original.getCost(x,y));
      if (map.getCost(x,y)>=254) {EXPECT_EQ(search.getCost(x,y),map.getCost(x,y));}
    }
  }
}

TEST_F(CollisionTest, TrackingMarginRejectsGrazingLegButAllowsCentredPassage)
{
  for (double x=-3.;x<3.;x+=.01) {obstacle(x,-.50);obstacle(x,.50);}
  auto input=route({{-2,0},{2,0}});
  const auto allowed=lightweight_fusion_bringup::trackingMarginCheck(map,footprint,
    input.poses.front().pose,input.poses.back().pose,.06);
  EXPECT_TRUE(allowed(0,0,0));
  EXPECT_TRUE(lightweight_fusion_bringup::footprintPoseFree(map,footprint,0,.25,0));
  EXPECT_FALSE(allowed(0,.25,0));
  lightweight_fusion_bringup::ShortcutPreference preference;preference.pose_allowed=allowed;
  EXPECT_FALSE(shortcutPolyline(input,map,footprint,
    std::chrono::steady_clock::now()+std::chrono::seconds(2),&preference).poses.empty());
}

TEST_F(CollisionTest, EndpointMarginRelaxationNeverRemovesPhysicalCollisionCheck)
{
  for (double x=-3.;x<3.;x+=.01) {obstacle(x,-.27);}
  auto input=route({{0,0},{2,1}});
  const auto allowed=lightweight_fusion_bringup::trackingMarginCheck(map,footprint,
    input.poses.front().pose,input.poses.back().pose,.06);
  EXPECT_TRUE(allowed(0,0,0)); // Can start where physical footprint fits.
  EXPECT_FALSE(allowed(1,0,0)); // Extra margin must recover away from endpoint.
  EXPECT_FALSE(allowed(0,-.08,0)); // Never permit the actual body to hit the wall.
}

TEST_F(CollisionTest, NarrowVerticalPassageKeepsCorrectBodyHeading)
{
  for (double y=-3.;y<3.;y+=.01) {obstacle(-.30,y);obstacle(.30,y);}
  auto p=route({{0,0},{0,2}});
  for (auto & v:p.poses) {v.pose.orientation.z=std::sin(M_PI/4);v.pose.orientation.w=std::cos(M_PI/4);}
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  const auto result=shortcutPolyline(p,map,footprint,deadline);
  ASSERT_EQ(result.poses.size(),2u);
  auto oriented=orientPolyline(result);
  EXPECT_NEAR(planarYaw(oriented.poses.front().pose.orientation),M_PI/2,1e-9);
  EXPECT_EQ(oriented.poses.back(),p.poses.back());
  EXPECT_TRUE(free({.45,0,0},{},M_PI/2));
  p.poses.back().pose.orientation.z=0;p.poses.back().pose.orientation.w=1;
  EXPECT_TRUE(shortcutPolyline(p,map,footprint,deadline).poses.empty());
}

TEST_F(CollisionTest, TurnChecksTheSweepEvenWhenBothEndpointHeadingsFit)
{
  for (double x=-3.;x<3.;x+=.01) {obstacle(x,-.30);obstacle(x,.30);}
  geometry_msgs::msg::Point at;
  EXPECT_TRUE(free({}));EXPECT_TRUE(free({}, {}, M_PI));
  EXPECT_FALSE(rotationCorridorFree(map,footprint,at,0,M_PI,
    std::chrono::steady_clock::now()+std::chrono::seconds(2)));
}

TEST_F(CollisionTest, FinishingSmallTurnDoesNotProjectAnotherFullSecond)
{
  // This point is swept at ~0.3 rad, well after a 0.05 rad correction and
  // its conservative latency/braking tail. A long turn must still stop.
  obstacle(.21,.28);
  EXPECT_FALSE(free({0,0,.18},{0,0,.18}));
  EXPECT_TRUE(check.isFree(map,footprint,0,0,0,{0,0,.18},{0,0,.18},.05));
  EXPECT_FALSE(check.isFree(map,footprint,0,0,0,{0,0,.5},{0,0,.18},.05));
  obstacle(.25,.245);
  EXPECT_FALSE(check.isFree(map,footprint,0,0,0,{0,0,.18},{0,0,.18},.05));
}

TEST_F(CollisionTest, PathOrientationsFollowEachLegAndPreserveFinalHeading)
{
  auto p=route({{0,0},{1,1},{1,2}});
  p.poses.back().pose.orientation.z=1;p.poses.back().pose.orientation.w=0;
  auto q=orientPolyline(p);
  EXPECT_NEAR(planarYaw(q.poses[0].pose.orientation),M_PI/4,1e-9);
  EXPECT_NEAR(planarYaw(q.poses[1].pose.orientation),M_PI/2,1e-9);
  EXPECT_EQ(q.poses.back(),p.poses.back());
  for (size_t i=0;i<p.poses.size();++i) {EXPECT_EQ(p.poses[i].pose.position,q.poses[i].pose.position);}
}

TEST_F(CollisionTest, OpenSpaceTranslationRotationAndStop)
{
  EXPECT_TRUE(free({0.45, 0.08, 0.0}));
  EXPECT_TRUE(free({-0.1, -0.08, 0.0}));
  EXPECT_TRUE(free({0.0, 0.0, 0.5}));
  EXPECT_TRUE(free({}, {0.45, 0.08, 0.5}));
}

TEST_F(CollisionTest, ForwardObstacle)
{
  obstacle(0.60, 0.0);
  EXPECT_FALSE(free({0.45, 0.0, 0.0}));
}

TEST_F(CollisionTest, LateralObstacle)
{
  obstacle(0.0, 0.33);
  EXPECT_FALSE(free({0.0, 0.10, 0.0}, {0.0, 0.10, 0.0}));
}

TEST_F(CollisionTest, ReverseObstacle)
{
  obstacle(-0.45, 0.0);
  EXPECT_FALSE(free({-0.10, 0.0, 0.0}, {-0.10, 0.0, 0.0}));
}

TEST_F(CollisionTest, RotatingCornerSweepsObstacle)
{
  obstacle(0.0, 0.35);
  EXPECT_TRUE(free({}));
  EXPECT_FALSE(free({0.0, 0.0, 0.7}, {0.0, 0.0, 0.7}));
}

TEST_F(CollisionTest, ObstacleInsideFootprint)
{
  obstacle(0.06, 0.06);
  EXPECT_FALSE(free({}));
}

TEST_F(CollisionTest, UnknownAndOutsideMapFailClosed)
{
  obstacle(0.0, 0.0, nav2_costmap_2d::NO_INFORMATION);
  EXPECT_FALSE(free({}));
  EXPECT_FALSE(check.isFree(map, footprint, 3.9, 0.0, 0.0, {}, {}));
}

TEST_F(CollisionTest, FullStoppingDistanceBeyondConstantHorizon)
{
  // At the end of the hold + horizon the front is at x=1.0425. Braking
  // adds another 0.135 m, and must catch an obstacle beyond that endpoint.
  obstacle(1.12, 0.0);
  EXPECT_FALSE(free({0.45, 0.0, 0.0}, {0.45, 0.0, 0.0}));
}

TEST_F(CollisionTest, ResidualMotionCheckedEvenForZeroTarget)
{
  obstacle(0.55, 0.0);
  EXPECT_TRUE(free({}));
  EXPECT_FALSE(free({}, {0.45, 0.0, 0.0}));
}

TEST_F(CollisionTest, YawChangesWorldDirection)
{
  obstacle(0.0, 0.60);
  EXPECT_TRUE(free({0.45, 0.0, 0.0}));
  EXPECT_FALSE(free({0.45, 0.0, 0.0}, {}, 1.5707963267948966));
}

TEST_F(CollisionTest, MixedMotionDuringSmootherModeTransition)
{
  obstacle(0.0, 0.35);
  EXPECT_FALSE(free({0.0, 0.0, 0.7}, {0.0, 0.08, 0.0}));
}

TEST_F(CollisionTest, MalformedInputAndUnboundedWorkFailClosed)
{
  EXPECT_FALSE(free({std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}));
  EXPECT_FALSE(free({1000.0, 0.0, 0.0}));
  EXPECT_FALSE(check.isFree(map, {}, 0.0, 0.0, 0.0, {}, {}));
  check.deceleration[0] = 0.0;
  EXPECT_FALSE(free({}));
}

TEST_F(CollisionTest, SteeringArcChecksItsSideSweep)
{
  obstacle(.65,.40);
  EXPECT_TRUE(free({.45,0,0},{.45,0,0}));
  EXPECT_FALSE(free({.45,0,.4},{.45,0,.4}));
}

TEST_F(CollisionTest, SideStepChecksHeldRectangleInsteadOfTravelHeading)
{
  using lightweight_fusion_bringup::translationCorridorFree;
  geometry_msgs::msg::Point a,b;b.y=1.5;
  auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
  EXPECT_TRUE(translationCorridorFree(map,footprint,a,b,0,deadline));
  obstacle(.28,.75);
  EXPECT_TRUE(straightCorridorFree(map,footprint,a,b,deadline));
  EXPECT_FALSE(translationCorridorFree(map,footprint,a,b,0,deadline));
}

TEST_F(CollisionTest, SideStepRejectsIntermediateObstacleAndUnsafeFinalRotation)
{
  using lightweight_fusion_bringup::translationCorridorFree;
  geometry_msgs::msg::Point a,b;b.y=1.5;
  auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
  obstacle(0,.75);
  EXPECT_FALSE(translationCorridorFree(map,footprint,a,b,0,deadline));
  map.resetMap(0,0,map.getSizeInCellsX(),map.getSizeInCellsY());
  obstacle(.02,1.84);
  EXPECT_TRUE(translationCorridorFree(map,footprint,a,b,0,deadline));
  EXPECT_TRUE(lightweight_fusion_bringup::footprintPoseFree(map,footprint,0,1.5,M_PI/2));
  EXPECT_FALSE(rotationCorridorFree(map,footprint,b,0,M_PI/2,deadline));
}

TEST_F(CollisionTest, FiniteLateralAlignmentChecksStopAtLineInsteadOfCruisingAcrossPassage)
{
  for (double x=-1.;x<=1.;x+=.02) {obstacle(x,.30);obstacle(x,-.30);}
  const double nan=std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(check.isFree(map,footprint,0,.04,0,{}, {0,-.10,0}));
  EXPECT_TRUE(check.isFree(map,footprint,0,.04,0,{}, {0,-.10,0},nan,-.04));
  // The finite target must not hide real braking momentum or intervening obstacles.
  EXPECT_FALSE(check.isFree(map,footprint,0,.04,0,{0,-.30,0}, {0,-.10,0},nan,-.04));
  obstacle(0,-.21);  // Outside the starting footprint, inside the correction sweep.
  EXPECT_FALSE(check.isFree(map,footprint,0,.04,0,{}, {0,-.10,0},nan,-.04));
}
