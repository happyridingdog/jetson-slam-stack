#include <gtest/gtest.h>
#include "path_tracking.hpp"

using lightweight_fusion_bringup::PathTracking;
using lightweight_fusion_bringup::TrackingPoint;
using lightweight_fusion_bringup::TrackingVelocity;

TEST(PathTracking, SparseAndDensePathsHaveSameLookahead)
{
  PathTracking a, b;
  a.setPlan({{0, 0}, {10, 0}});
  std::vector<TrackingPoint> dense;
  for (int i = 0; i <= 100; ++i) {dense.push_back({i/10., 0});}
  b.setPlan(dense);
  for (int step = 0; step <= 100; ++step) {
    const double x = step / 10.0;
    const auto p = a.target(x, .1), q = b.target(x, .1);
    EXPECT_NEAR(p.x, std::min(10., x+.65), 1e-9);
    EXPECT_NEAR(p.x, q.x, 1e-9);
    EXPECT_NEAR(p.y, q.y, 1e-9);
  }
}

TEST(PathTracking, SharpCornerIsCapturedBeforeTurning)
{
  PathTracking tracker;
  tracker.setPlan({{0, 0}, {2, 0}, {2, 2, M_PI/2}});
  const auto target = tracker.target(1.6, 0);
  EXPECT_NEAR(target.x, 2., 1e-9);
  EXPECT_NEAR(target.y, 0., 1e-9);
  const auto approaching = tracker.command({1.6, 0, 0}, {}, .45);
  EXPECT_LT(approaching.x, .4);
  EXPECT_DOUBLE_EQ(approaching.yaw, 0);
  const auto braking = tracker.command({1.91, 0, 0}, {.1, 0, 0}, .45);
  EXPECT_DOUBLE_EQ(braking.x, 0);
  EXPECT_DOUBLE_EQ(braking.yaw, 0);
  const auto turning = tracker.command({1.93, 0, 0}, {}, .45);
  EXPECT_GT(turning.yaw, 0);
}

TEST(PathTracking, ProgressDoesNotJumpToLaterCrossing)
{
  PathTracking tracker;
  tracker.setPlan({{-2, 0}, {2, 0}, {2, 2}, {0, 2}, {0, -2}});
  tracker.target(-2, 0);
  tracker.target(-.3, 0);
  const auto p = tracker.target(.01, .05);
  EXPECT_GT(p.x, .5);
  EXPECT_NEAR(p.y, 0., 1e-9);
}

TEST(PathTracking, WaitsForResidualTranslationAndRotation)
{
  PathTracking tracker;
  tracker.setPlan({{0, 0}, {0, 5, M_PI/2}});
  auto v = tracker.command({0, 0, 0}, {.3, 0, 0}, .45);
  EXPECT_DOUBLE_EQ(v.x, 0);
  EXPECT_NEAR(v.yaw, 0., 1e-12);
  v = tracker.command({0, 0, 0}, {}, .45);
  EXPECT_GT(v.yaw, .1);
  v = tracker.command({0, 0, M_PI/2}, {0, 0, .15}, .45);
  EXPECT_DOUBLE_EQ(v.x, 0);
  EXPECT_NEAR(v.yaw, 0., 1e-12);
  v = tracker.command({0, 0, M_PI/2}, {}, .45);
  EXPECT_GT(v.x, .4);
  EXPECT_NEAR(v.yaw, 0., 1e-12);
}

TEST(PathTracking, AlignmentCompletesBeforeStartingStraightLeg)
{
  PathTracking tracker;
  tracker.setPlan({{0, 0}, {5, 0}});
  for (double error : {1.1, 1.0, .8, .5, .10, .06}) {
    auto v = tracker.command({0, 0, -error}, {}, .45);
    EXPECT_DOUBLE_EQ(v.x, 0);
    EXPECT_GT(v.yaw, 0);
  }
  EXPECT_GT(tracker.command({0, 0, -.02}, {}, .45).x, 0);
  EXPECT_GT(tracker.command({0, 0, -.24}, {}, .45).x, 0);
}

TEST(PathTracking, TightPassageRefinesEntryHeadingWithoutStrafing)
{
  PathTracking tracker;tracker.setPlan({{0,0},{3,0}});
  auto corridor=[](const TrackingPoint & p,const TrackingPoint &) {return std::abs(p.yaw)<.012;};
  auto v=tracker.command({0,0,-.02},{},.45,.067,{},corridor);
  EXPECT_GT(v.yaw,0);EXPECT_EQ(v.x,0);EXPECT_EQ(v.y,0);
  EXPECT_NEAR(tracker.remainingTurn(),.02,1e-9);
  v=tracker.command({0,0,-.01},{},.45,.067,{},corridor);
  EXPECT_GT(v.x,0);EXPECT_EQ(v.yaw,0);EXPECT_EQ(v.y,0);
  EXPECT_TRUE(std::isnan(tracker.remainingTurn()));
}

TEST(PathTracking, BlockedExactHeadingDoesNotSpinAimlessly)
{
  PathTracking tracker;tracker.setPlan({{0,0},{3,0}});
  auto blocked=[](const TrackingPoint &,const TrackingPoint &) {return false;};
  EXPECT_THROW(tracker.command({0,0,0},{},.45,.067,{},blocked),std::runtime_error);
}

TEST(PathTracking, GoalLatchSurvivesNoiseAndReplanButNotNewGoal)
{
  PathTracking tracker;
  tracker.setPlan({{0, 0}, {2, 0, M_PI/2}});
  EXPECT_GT(tracker.command({1.81, 0, 0}, {}, .45).yaw, 0);
  tracker.setPlan({{1.81, 0}, {2, 0, M_PI/2}});
  const auto v = tracker.command({1.795, 0, 0}, {}, .45);
  EXPECT_DOUBLE_EQ(v.x, 0);
  EXPECT_GT(v.yaw, 0);
  tracker.setPlan({{1.795, 0}, {4, 0, 0}});
  EXPECT_GT(tracker.command({1.795, 0, 0}, {}, .45).x, 0);
}

TEST(PathTracking, InvalidGeometryAndLimitsFailClosed)
{
  PathTracking tracker;
  EXPECT_TRUE(tracker.valid());
  tracker.heading_exit = 0.;
  EXPECT_FALSE(tracker.valid());
  EXPECT_THROW(tracker.setPlan({{0, 0}, {NAN, 2}}), std::invalid_argument);
}

TEST(PathTracking, FinalHeadingConvergesWithMeasuredYawDeadZone)
{
  // Recorded IMU calibration showed essentially no rotation at 5 deg/s,
  // but rotation at 10 deg/s. Do not rely on arbitrarily small angular output.
  for (double initial : {-.7, .7}) {
    PathTracking tracker;
    tracker.setPlan({{0, 0, 0}});
    double yaw = initial, smooth = 0, actual = 0;
    for (int step = 0; step < 2000; ++step) {
      const auto desired = tracker.command({0, 0, yaw}, {0, 0, actual}, .45);
      smooth += std::clamp(desired.yaw-smooth, -.008, .008);
      const double motor = std::abs(smooth) < .12 ? 0 : .5*smooth;
      actual += .08*(motor-actual);
      yaw = PathTracking::angle(yaw+actual*.01);
    }
    EXPECT_LT(std::abs(yaw), tracker.yaw_tolerance);
    EXPECT_NEAR(actual, 0., 1e-6);
  }
}

TEST(PathTracking, OpenSpaceLastDegreesClearLoadedMotorDeadZone)
{
  // Reproduce a loaded motor that no longer responds to the old 0.18 floor.
  // Include the real smoother limits and actuator lag, for both turn phases
  // and both directions. No obstacles are involved in this regression.
  for (bool final_heading : {false,true}) {
    for (double initial : {-.13,.13}) {
      for (double response : {.6,1.0}) {
        PathTracking tracker;
        tracker.setPlan(final_heading ? std::vector<TrackingPoint>{{0,0,0}} :
          std::vector<TrackingPoint>{{0,0,0},{3,0,0}});
        double yaw=initial,smooth=0,actual=0;
        TrackingVelocity desired;
        bool completed=false;
        int reversals=0,previous_sign=0;
        for (int tick=0;tick<1000;++tick) {
          if (tick%7==0) {
            desired=tracker.command({0,0,yaw},{0,0,smooth},.45,.07,{0,0,actual});
            const int sign=desired.yaw>0 ? 1 : (desired.yaw<0 ? -1 : 0);
            if (sign && previous_sign && sign!=previous_sign) {++reversals;}
            if (sign) {previous_sign=sign;}
            const double tolerance=final_heading ? tracker.yaw_tolerance : tracker.heading_exit;
            if (std::abs(yaw)<=tolerance && std::abs(actual)<.01 &&
              std::abs(smooth)<.01 && (final_heading || desired.x>0)) {
              completed=true;break;
            }
          }
          const double limit=std::abs(desired.yaw)>std::abs(smooth) ? .008 : .009;
          smooth+=std::clamp(desired.yaw-smooth,-limit,limit);
          const double motor=std::abs(smooth)<.22 ? 0 : response*smooth;
          actual+=(motor-actual)*(1-std::exp(-.01/.12));
          yaw=PathTracking::angle(yaw+actual*.01);
        }
        EXPECT_TRUE(completed) << final_heading << " " << initial << " " << response << " " << yaw;
        EXPECT_LE(reversals,2);
      }
    }
  }
}

TEST(PathTracking, TurnBrakesBeforeTargetAndSettlesBeforeCorrecting)
{
  PathTracking tracker;tracker.setPlan({{0,0,0},{3,0,0}});
  auto v=tracker.command({0,0,-.10},{0,0,.27},.45,.07,{0,0,.27});
  EXPECT_DOUBLE_EQ(v.yaw,0);
  EXPECT_DOUBLE_EQ(v.x,0);
  // Even if the instantaneous stopping estimate shrinks, finish this stop.
  v=tracker.command({0,0,-.07},{0,0,.10},.45,.07,{0,0,.05});
  EXPECT_DOUBLE_EQ(v.yaw,0);
  v=tracker.command({0,0,-.04},{},.45,.07,{});
  EXPECT_GE(v.yaw,tracker.preferred_min_angular);
  // An overshoot must brake before asking the still-moving vehicle to reverse.
  v=tracker.command({0,0,.04},{0,0,.20},.45,.07,{0,0,.20});
  EXPECT_DOUBLE_EQ(v.yaw,0);
  PathTracking starting;starting.setPlan({{0,0,0},{3,0,0}});
  v=starting.command({0,0,-.10},{0,0,.10},.45,.07,{0,0,-.005});
  EXPECT_GE(v.yaw,starting.preferred_min_angular);  // Small IMU noise is not a reversal.
}

TEST(PathTracking, SteadyTranslationIsNotInterruptedByOdometryYawNoise)
{
  PathTracking tracker;
  tracker.setPlan({{0, 0}, {4, 0}});
  tracker.command({.5, 0, .01}, {}, .45);
  for (double measured_yaw_rate : {0.03, -0.04, 0.01, -0.025}) {
    const auto v = tracker.command({.5, 0, .01}, {.4, 0, measured_yaw_rate}, .45);
    EXPECT_GT(v.x, .4);
    EXPECT_DOUBLE_EQ(v.y, 0.);
  }
}

TEST(PathTracking, BehindGoalKeepsTurnDirectionAcrossPiSeamAndReplans)
{
  for (double initial_noise : {-.002, .002}) {
    PathTracking tracker;
    const std::vector<TrackingPoint> path{{0, 0}, {-2, 0, M_PI}};
    tracker.setPlan(path);
    const double first = tracker.command({0, 0, initial_noise}, {}, .45).yaw;
    ASSERT_NE(first, 0.);
    for (int i = 0; i < 40; ++i) {
      tracker.setPlan(path);
      const auto v = tracker.command({0, 0, (i%2 ? -1 : 1)*initial_noise}, {}, .45);
      EXPECT_GT(first*v.yaw, 0.);
      EXPECT_DOUBLE_EQ(v.x, 0.);
    }
    // After turning halfway, the wrap seam no longer influences feedback.
    const double half_turn = first > 0 ? M_PI/2 : -M_PI/2;
    EXPECT_GT(first*tracker.command({0, 0, half_turn}, {}, .45).yaw, 0.);
    const auto aligned = tracker.command({0, 0, first > 0 ? M_PI : -M_PI}, {}, .45);
    EXPECT_GT(aligned.x, 0.);
    EXPECT_NEAR(aligned.yaw, 0., 1e-12);
  }
}

TEST(PathTracking, FinalHeadingBehindAlsoKeepsDirectionAcrossPiSeam)
{
  PathTracking tracker;
  tracker.setPlan({{0, 0, M_PI}});
  const double first = tracker.command({0, 0, .002}, {}, .45).yaw;
  for (double yaw : {-.003, .002, -.001}) {
    EXPECT_GT(first*tracker.command({0, 0, yaw}, {}, .45).yaw, 0.);
  }
}

TEST(PathTracking, TranslationFloorPreservesDirectionAndSpeedLimits)
{
  PathTracking tracker;
  tracker.setPlan({{0, 0}, {2, 0}, {2, 2, M_PI/2}});
  const auto v = tracker.command({1.875, 0., 0.}, {}, .45);
  EXPECT_NEAR(std::hypot(v.x, v.y), .10, 1e-9);
  EXPECT_LT(std::abs(v.y), .01);
  EXPECT_DOUBLE_EQ(v.y, 0.);
  const auto limited = tracker.command({1.875, 0., 0.}, {}, .06);
  EXPECT_LE(std::hypot(limited.x, limited.y), .06000001);
}

TEST(PathTracking, OnceTurningSmallLinearEstimationNoiseDoesNotInterruptRotation)
{
  PathTracking tracker;
  tracker.setPlan({{0,0}, {-2,0,M_PI}});
  EXPECT_NE(tracker.command({0,0,0}, {}, .45).yaw, 0.);
  for (double heading : {.1,.2,.3}) {
    const auto v=tracker.command({0,0,heading}, {.025,.025,.3}, .45);
    EXPECT_GT(v.yaw, 0.);
    EXPECT_DOUBLE_EQ(v.x, 0.);
  }
}

TEST(PathTracking, CentimetreJitterDoesNotProduceAlternatingStrafe)
{
  PathTracking tracker;
  tracker.setPlan({{0,0}, {4,0}});
  tracker.command({0,0,0},{},.45);
  for (int i=0; i<100; ++i) {
    const auto v=tracker.command({.01*i, .02*std::sin(i), 0}, {.4,0,0}, .45);
    EXPECT_GT(v.x, .4);
    EXPECT_NEAR(v.y, 0., 1e-12);
    EXPECT_DOUBLE_EQ(v.yaw, 0.);
  }
}

TEST(PathTracking, OnlyFinalApproachAllowsSlowLateralAdjustment)
{
  PathTracking tracker;tracker.setPlan({{0,0},{4,0}});
  auto far=tracker.command({1,.3,0},{},.45);
  EXPECT_DOUBLE_EQ(far.y,0.);
  tracker.command({2.8,.2,0},{},.45); // advance within the bounded projection search
  auto near=tracker.command({3.75,.2,0},{},.45);
  EXPECT_LT(near.y,0.);EXPECT_GT(near.x,0.);
  EXPECT_NEAR(std::hypot(near.x,near.y),.1,1e-9);
  EXPECT_DOUBLE_EQ(near.yaw,0.);
  auto side=tracker.command({4,.3,0},{},.45);
  EXPECT_NEAR(side.x,0.,1e-9);EXPECT_NEAR(side.y,-.1,1e-9);
  // A real excursion leaves adjustment mode; it is not latched indefinitely.
  auto escaped=tracker.command({3.,.3,0},{},.45);
  EXPECT_DOUBLE_EQ(escaped.y,0.);
}

TEST(PathTracking, NearbyGoalAcrossLoopDoesNotEnableEarlyStrafe)
{
  PathTracking tracker;
  tracker.setPlan({{0,0},{2,0},{2,2},{0,2},{0,.3}});
  auto v=tracker.command({0,0,0},{},.45);
  EXPECT_GT(v.x,0.);EXPECT_DOUBLE_EQ(v.y,0.);EXPECT_DOUBLE_EQ(v.yaw,0.);
}

TEST(PathTracking, FinalAdjustmentRespectsCornerAndEndsWithFinalHeading)
{
  PathTracking tracker;
  tracker.setPlan({{0,0},{2,0},{2,.3,M_PI/2}});
  auto before=tracker.command({1.8,0,0},{},.45);
  EXPECT_DOUBLE_EQ(before.y,0.); // do not strafe diagonally across the last corner
  tracker.command({1.90,0,0},{},.45);
  auto final=tracker.command({2,.15,0},{},.45);
  EXPECT_DOUBLE_EQ(final.x,0.);EXPECT_DOUBLE_EQ(final.y,0.);EXPECT_GT(final.yaw,0.);
}

TEST(PathTracking, StraightLegNeverSteersOrStrafesForSmallDrift)
{
  PathTracking tracker;tracker.setPlan({{0,0},{6,0}});
  ASSERT_GT(tracker.command({0,0,0},{},.45).x,0.);
  for (double x : {1.,2.,3.,4.,5.}) {
    const auto v=tracker.command({x,.12,.03},{.4,0,0},.45);
    EXPECT_GT(v.x,0.);EXPECT_DOUBLE_EQ(v.y,0.);EXPECT_DOUBLE_EQ(v.yaw,0.);
  }
  const auto limited=tracker.command({5,.12,.03},{},.06);
  EXPECT_LE(limited.x,.06);
}

TEST(PathTracking, IdenticalReplansDoNotInterruptLockedLeg)
{
  PathTracking tracker;const std::vector<TrackingPoint> path{{0,0},{6,0}};
  tracker.setPlan(path);tracker.command({0,0,0},{},.45);
  for (int i=1;i<40;++i) {
    tracker.setPlan(path);
    const auto v=tracker.command({.1*i,.10,.04},{.4,0,0},.45);
    EXPECT_GT(v.x,0.);EXPECT_DOUBLE_EQ(v.yaw,0.);EXPECT_DOUBLE_EQ(v.y,0.);
  }
}

TEST(PathTracking, CollinearPlannerSamplesBecomeOneLongLeg)
{
  PathTracking tracker;std::vector<TrackingPoint> path;
  for(int i=0;i<=120;++i) {path.push_back({.05*i,0.0,0});}
  path.front().y=path.back().y=0.;tracker.setPlan(path);
  for(int i=0;i<100;++i) {
    auto v=tracker.command({.05*i,0,0},{},.45);
    EXPECT_GT(v.x,0.);EXPECT_DOUBLE_EQ(v.yaw,0.);EXPECT_DOUBLE_EQ(v.y,0.);
  }
}

TEST(PathTracking, NextLegUsesActualEndpointPositionAfterStopping)
{
  PathTracking tracker;tracker.setPlan({{0,0},{3,0},{3,3,M_PI/2}});
  tracker.command({0,0,0},{},.45);
  tracker.command({2.,.10,0},{.4,0,0},.45);
  auto v=tracker.command({2.89,.10,0},{.15,0,0},.45);
  EXPECT_DOUBLE_EQ(v.x,0.);EXPECT_DOUBLE_EQ(v.yaw,0.);
  v=tracker.command({2.91,.10,0},{},.45);
  EXPECT_DOUBLE_EQ(v.x,0.);EXPECT_GT(v.yaw,0.);
  const double heading=std::atan2(2.90,.09);
  v=tracker.command({2.91,.10,heading},{0,0,.10},.45);
  EXPECT_DOUBLE_EQ(v.x,0.);EXPECT_DOUBLE_EQ(v.yaw,0.);
  v=tracker.command({2.91,.10,heading},{},.45);
  EXPECT_GT(v.x,0.);EXPECT_DOUBLE_EQ(v.y,0.);EXPECT_DOUBLE_EQ(v.yaw,0.);
}

TEST(PathTracking, ExcessiveDriftRequestsReplanInsteadOfDrivingBlindly)
{
  PathTracking tracker;tracker.setPlan({{0,0},{6,0}});
  tracker.command({0,0,0},{},.45);
  EXPECT_THROW(tracker.command({2.,.30,0},{},.45),std::runtime_error);
}

TEST(PathTracking, ClearanceGuardRunsDuringMotionBeforeAbsoluteDriftLimit)
{
  PathTracking tracker;tracker.setPlan({{0,0},{6,0}});
  bool corridor_free=true;int checks=0;
  auto guard=[&](const TrackingPoint &,const TrackingPoint &) {++checks;return corridor_free;};
  ASSERT_GT(tracker.command({0,0,0},{},.8,.067,{},guard).x,0.);
  const int entry_checks=checks;
  const auto moving=tracker.command({1,.02,.005},{.5,0,0},.8,.067,{},guard);
  EXPECT_GT(moving.x,0.);EXPECT_EQ(moving.y,0.);EXPECT_EQ(moving.yaw,0.);
  EXPECT_GT(checks,entry_checks);
  // The body is only 4 cm off the reference, well below the 25 cm global
  // limit, but its forward ray is about to consume a narrow corridor's room.
  corridor_free=false;
  const auto stop=tracker.command({2,.04,.06},{.5,0,0},.8,.067,{},guard);
  EXPECT_EQ(stop.x,0.);EXPECT_EQ(stop.y,0.);EXPECT_EQ(stop.yaw,0.);
  EXPECT_EQ(tracker.clearanceRealignments(),1u);
  const auto braking=tracker.command({2,.04,.06},{.1,0,0},.8,.067,{},guard);
  EXPECT_EQ(braking.x,0.);EXPECT_EQ(braking.yaw,0.);
  const auto turn=tracker.command({2,.04,.06},{},.8,.067,{},guard);
  EXPECT_EQ(turn.x,0.);EXPECT_EQ(turn.y,0.);EXPECT_LT(turn.yaw,0.);
  // The same goal/reference resumes after a stopped correction, without
  // exhausting the navigation action's finite replanning retry count.
  corridor_free=true;
  const double heading=std::atan2(-.04,4.);
  const auto resumed=tracker.command({2,.04,heading},{},.8,.067,{},guard);
  EXPECT_GT(resumed.x,0.);EXPECT_EQ(resumed.y,0.);EXPECT_EQ(resumed.yaw,0.);
}

TEST(PathTracking, LongLegMeasuresDriftAgainstIntendedBearing)
{
  PathTracking tracker;tracker.setPlan({{0,0},{20,0}});
  EXPECT_GT(tracker.command({0,0,.02},{},.45).x,0.);
  // A small starting angle error must not accumulate without a bound.
  EXPECT_THROW(tracker.command({18.,.30,.02},{},.45),std::runtime_error);
}

TEST(PathTracking, FinalStrafeWaitsForRotationAndFinalRotationWaitsForTranslation)
{
  PathTracking tracker;tracker.setPlan({{0,0},{0,.32,M_PI/2}});
  auto v=tracker.command({0,0,0},{0,0,.2},.45);
  EXPECT_DOUBLE_EQ(v.x,0.);EXPECT_DOUBLE_EQ(v.y,0.);EXPECT_DOUBLE_EQ(v.yaw,0.);
  v=tracker.command({0,0,0},{},.45);
  EXPECT_GT(v.y,0.);EXPECT_DOUBLE_EQ(v.yaw,0.);
  v=tracker.command({0,.20,0},{0,.1,0},.45);
  EXPECT_DOUBLE_EQ(v.x,0.);EXPECT_DOUBLE_EQ(v.y,0.);EXPECT_DOUBLE_EQ(v.yaw,0.);
  v=tracker.command({0,.20,0},{},.45);
  EXPECT_GT(v.yaw,0.);EXPECT_DOUBLE_EQ(v.x,0.);EXPECT_DOUBLE_EQ(v.y,0.);
}

TEST(PathTracking, DelayedClosedLoopUsesOneStraightLegThenGoalAdjustment)
{
  PathTracking tracker;tracker.xy_tolerance=.12;
  const std::vector<TrackingPoint> path{{0,0},{6,0}};tracker.setPlan(path);
  TrackingPoint pose{0,.08,.08};TrackingVelocity actual{},smooth{},desired{};
  int straight_turns=0;double max_deviation=0;
  for(int tick=0;tick<4000;++tick) {
    const double time=tick*.01;
    if(tick%7==0) {
      tracker.setPlan(path);
      auto noisy=pose;noisy.y+=.002*std::sin(5*time);noisy.yaw+=.002*std::sin(3*time);
      TrackingVelocity residual=smooth;
      if(std::abs(actual.yaw)>std::abs(residual.yaw)) {residual.yaw=actual.yaw;}
      desired=tracker.command(noisy,residual,.45,.07,actual);
      EXPECT_FALSE(std::hypot(desired.x,desired.y)>0 && std::abs(desired.yaw)>0);
      if(pose.x>.5 && pose.x<5.4) {
        EXPECT_DOUBLE_EQ(desired.y,0.);straight_turns+=std::abs(desired.yaw)>1e-9;
      }
    }
    auto ramp=[](double a,double b,double step){return a+std::clamp(b-a,-step,step);};
    smooth.x=ramp(smooth.x,desired.x,.0065);smooth.y=ramp(smooth.y,desired.y,.002);
    smooth.yaw=ramp(smooth.yaw,desired.yaw,.008);
    actual.x+=(smooth.x-actual.x)*.08;actual.y+=(smooth.y-actual.y)*.08;
    actual.yaw+=(smooth.yaw-actual.yaw)*.08;
    pose.x+=(std::cos(pose.yaw)*actual.x-std::sin(pose.yaw)*actual.y)*.01;
    pose.y+=(std::sin(pose.yaw)*actual.x+std::cos(pose.yaw)*actual.y)*.01;
    pose.yaw=PathTracking::angle(pose.yaw+actual.yaw*.01);
    max_deviation=std::max(max_deviation,std::abs(pose.y));
  }
  EXPECT_EQ(straight_turns,0);EXPECT_LT(max_deviation,.25);
  EXPECT_LT(std::hypot(pose.x-6,pose.y),.14);EXPECT_LT(std::abs(pose.yaw),tracker.yaw_tolerance);
}

TEST(PathTracking, RealBendsAndReversalsAreNotFittedAway)
{
  PathTracking tracker;
  tracker.setPlan({{0,0},{1,.02},{2,0}});
  ASSERT_EQ(tracker.legEnds().size(),2u);
  EXPECT_EQ(tracker.legEnds()[0],1u);
  tracker.setPlan({{0,0},{2,0},{0,0}});
  ASSERT_EQ(tracker.legEnds().size(),2u);
  EXPECT_EQ(tracker.legEnds()[0],1u);
  tracker.setPlan({{0,0},{1,0},{2,0},{2,0}});
  ASSERT_EQ(tracker.legEnds().size(),1u);
  EXPECT_EQ(tracker.legEnds()[0],3u);
}

namespace {
const auto side_free=[](const TrackingPoint &,const TrackingPoint &) {return true;};
TrackingVelocity sideCommand(PathTracking & tracker,const TrackingPoint & pose,
  const TrackingVelocity & velocity={})
{
  return tracker.command(pose,velocity,.8,.02,velocity,{},side_free);
}
}

TEST(PathTracking, NearbySideGoalUsesBodyFrameAndStrictRange)
{
  using lightweight_fusion_bringup::nearbySideGoal;
  EXPECT_TRUE(nearbySideGoal(0,2,0));
  EXPECT_TRUE(nearbySideGoal(.3,-1.8,0));
  EXPECT_TRUE(nearbySideGoal(-1.5,.2,M_PI/2));
  EXPECT_TRUE(nearbySideGoal(.2,1.5,M_PI));
  EXPECT_FALSE(nearbySideGoal(0,2.001,0));
  EXPECT_FALSE(nearbySideGoal(.301,1,0));
  EXPECT_FALSE(nearbySideGoal(1,0,0));
  EXPECT_FALSE(nearbySideGoal(-1,0,0));
  EXPECT_FALSE(nearbySideGoal(.2,.1,0));
}

TEST(PathTracking, SideStepKeepsHeadingThenStopsBeforeGoalArrowTurn)
{
  for (double sign:{-1.,1.}) {
    PathTracking t;t.xy_tolerance=.12;
    t.setPlan({{0,0,0},{.1,sign*.75,0},{.2,sign*1.5,M_PI/2}});
    auto v=sideCommand(t,{0,0,0});
    EXPECT_GT(sign*v.y,0);EXPECT_GT(v.x,0);EXPECT_EQ(v.yaw,0);
    v=sideCommand(t,{0,0,0},{0,0,.1});
    EXPECT_EQ(v.x,0);EXPECT_EQ(v.y,0);EXPECT_EQ(v.yaw,0);
    v=sideCommand(t,{.2,sign*1.45,0},{0,sign*.1,0});
    EXPECT_EQ(v.x,0);EXPECT_EQ(v.y,0);EXPECT_EQ(v.yaw,0);
    v=sideCommand(t,{.2,sign*1.45,0});
    EXPECT_EQ(v.x,0);EXPECT_EQ(v.y,0);EXPECT_GT(v.yaw,0);
  }
}

TEST(PathTracking, SideStepDoesNotIgnoreDetourOrForwardAttitudes)
{
  PathTracking t;
  t.setPlan({{0,0,0},{.5,0,0},{.5,1,0},{0,1,0}});
  auto v=sideCommand(t,{0,0,0});EXPECT_GT(v.x,0);EXPECT_EQ(v.y,0);
  t.setPlan({{0,0,M_PI/2},{0,.5,M_PI/2},{0,1,M_PI/2}});
  v=sideCommand(t,{0,0,0});EXPECT_EQ(v.x,0);EXPECT_EQ(v.y,0);EXPECT_GT(v.yaw,0);
  t.setPlan({{0,0,0},{0,3,0}});
  v=sideCommand(t,{0,0,0});EXPECT_EQ(v.y,0);EXPECT_GT(v.yaw,0);
}

TEST(PathTracking, SideStepStopsWhenCorridorBecomesBlocked)
{
  PathTracking t;t.setPlan({{0,0,0},{0,1,0}});
  EXPECT_GT(sideCommand(t,{0,0,0}).y,0);
  EXPECT_THROW(t.command({0,.2,0},{0,.1,0},.8,.02,{}, {},
    [](const TrackingPoint &,const TrackingPoint &) {return false;}),std::runtime_error);
  EXPECT_THROW(sideCommand(t,{0,.2,.2}),std::runtime_error);
}

TEST(PathTracking, SideStepClosesWithActuatorLagAndPerAxisDeadzone)
{
  for (double heading:{0.,M_PI/2,M_PI}) {
    PathTracking t;t.xy_tolerance=.12;
    const double c=std::cos(heading),s=std::sin(heading);
    TrackingPoint goal{c*.25-s*1.,s*.25+c*1.,heading+M_PI/2};
    t.setPlan({{0,0,heading},goal});
    TrackingPoint pose{0,0,heading};TrackingVelocity actual;
    bool translated=false,turned=false,settled=false;
    for (int i=0;i<3500;++i) {
      const auto v=sideCommand(t,pose,actual);
      EXPECT_FALSE(std::hypot(v.x,v.y)>1e-9 && std::abs(v.yaw)>1e-9);
      if (std::hypot(goal.x-pose.x,goal.y-pose.y)>.12) {EXPECT_EQ(v.yaw,0);}
      translated=translated || std::abs(v.y)>.05;
      turned=turned || std::abs(v.yaw)>.05;
      auto motor=[](double speed) {return std::abs(speed)<.04 ? 0.:speed;};
      actual.x+=(motor(v.x)-actual.x)*(1-std::exp(-.02/.12));
      actual.y+=(motor(v.y)-actual.y)*(1-std::exp(-.02/.12));
      actual.yaw+=(v.yaw-actual.yaw)*(1-std::exp(-.02/.12));
      pose.x+=(std::cos(pose.yaw)*actual.x-std::sin(pose.yaw)*actual.y)*.02;
      pose.y+=(std::sin(pose.yaw)*actual.x+std::cos(pose.yaw)*actual.y)*.02;
      pose.yaw+=actual.yaw*.02;
      if (std::hypot(goal.x-pose.x,goal.y-pose.y)<.12 &&
        std::abs(PathTracking::angle(goal.yaw-pose.yaw))<t.yaw_tolerance &&
        std::hypot(actual.x,actual.y)<.005 && std::abs(actual.yaw)<.005) {settled=true;break;}
    }
    EXPECT_TRUE(translated);EXPECT_TRUE(turned);EXPECT_TRUE(settled);
  }
}

TEST(PathTracking, SideCruiseIsFasterButApproachAndForwardCorrectionRemainBounded)
{
  PathTracking t;t.max_lateral=.30;t.xy_tolerance=.12;
  t.setPlan({{0,0,0},{.2,1.5,0}});
  const auto cruise=sideCommand(t,{0,0,0});
  EXPECT_GT(cruise.y,.28);EXPECT_LE(cruise.y,.30);EXPECT_LE(std::abs(cruise.x),.10);
  const auto approach=sideCommand(t,{.2,1.24,0});
  EXPECT_LT(approach.y,.13);EXPECT_GE(approach.y,.10);EXPECT_EQ(approach.yaw,0);
  const auto micro=sideCommand(t,{0,1.48,0});
  EXPECT_LE(micro.x,.10);EXPECT_EQ(micro.yaw,0);
  const auto limited=t.command({0,0,0},{},.15,.02,{}, {},side_free);
  EXPECT_LE(std::hypot(limited.x,limited.y),.15+1e-9);
}

TEST(PathTracking, FasterTurnsSettleInsideTightToleranceWithLagAndMotorDeadzone)
{
  for (double initial:{-.13,.13,-1.57,1.57,-3.10,3.10}) {
    for (double response:{.6,1.,1.3}) {
      PathTracking t;t.max_angular=.60;t.turn_gain=1.8;t.yaw_tolerance=.025;
      t.setPlan({{0,0,0}});
      double yaw=initial,smooth=0,actual=0,peak=0;bool settled=false;
      TrackingVelocity desired;
      for (int tick=0;tick<3000;++tick) {
        if (tick%7==0) {
          desired=t.command({0,0,yaw},{0,0,smooth},.8,.07,{0,0,actual});
          peak=std::max(peak,std::abs(desired.yaw));
          if (std::abs(yaw)<=.025 && std::abs(actual)<.01 && std::abs(smooth)<.01) {
            settled=true;break;
          }
        }
        const double limit=std::abs(desired.yaw)>std::abs(smooth) ? .008 : .009;
        smooth+=std::clamp(desired.yaw-smooth,-limit,limit);
        const double motor=std::abs(smooth)<.22 ? 0 : response*smooth;
        actual+=(motor-actual)*(1-std::exp(-.01/.12));
        yaw=PathTracking::angle(yaw+actual*.01);
      }
      EXPECT_TRUE(settled)<<initial<<" "<<response<<" "<<yaw;
      if (std::abs(initial)>1) {EXPECT_DOUBLE_EQ(peak,.60);}
    }
  }
}

TEST(PathTracking, HeadingCorrectionTracksOriginalEdgeWithoutStrafing)
{
  PathTracking t;t.heading_correction=true;
  t.setPlan({{0,0,0},{4,0,0}});
  EXPECT_GT(t.command({0,0,0},{},.8).x,0);
  const auto left=t.command({1,.04,.02},{.4,0,0},.8);
  EXPECT_GT(left.x,0);EXPECT_EQ(left.y,0);EXPECT_LT(left.yaw,0);
  EXPECT_NEAR(t.crossTrackError(),.04,1e-9);EXPECT_TRUE(t.trackingForward());
  const auto right=t.command({1.1,-.04,-.02},{.4,0,0},.8);
  EXPECT_GT(right.x,0);EXPECT_EQ(right.y,0);EXPECT_GT(right.yaw,0);
  EXPECT_LE(std::abs(right.yaw),t.max_tracking_yaw);
}

TEST(PathTracking, HeadingCorrectionDoesNotCaptureCornerByEndpointPlaneAlone)
{
  PathTracking t;t.heading_correction=true;t.segment_capture=.04;
  t.setPlan({{0,0,0},{2,0,M_PI/2},{2,2,M_PI/2}});
  t.command({0,0,0},{},.8);
  auto v=t.command({1.975,.05,0},{.1,0,0},.8);
  EXPECT_TRUE(t.trackingForward());EXPECT_EQ(v.y,0);EXPECT_LT(v.yaw,0);
  v=t.command({1.98,.01,0},{.1,0,0},.8);
  EXPECT_FALSE(t.trackingForward());EXPECT_EQ(v.x,0);EXPECT_EQ(v.yaw,0);
  v=t.command({1.98,.01,0},{},.8);
  EXPECT_EQ(v.x,0);EXPECT_GT(v.yaw,0);
}

TEST(PathTracking, HeadingCorrectionKeepsReferenceAfterStopAndReacquisition)
{
  PathTracking t;t.heading_correction=true;
  t.setPlan({{0,0,0},{5,0,0}});
  t.command({0,0,0},{},.8);
  const double heading=std::atan2(-.04,.6);
  // A time gap reacquires the edge from an off-line stopped pose. It must
  // not replace the reference by a new long diagonal through that pose.
  const auto v=t.command({1,.04,heading},{},.8,.7);
  EXPECT_GT(v.x,0);EXPECT_EQ(v.y,0);EXPECT_TRUE(t.trackingForward());
  EXPECT_NEAR(t.crossTrackError(),.04,1e-9);
}

TEST(PathTracking, ClosedLoopStraightTracksWithSlipHeadingBiasAndLag)
{
  for (double sign:{-1.,1.}) {
    PathTracking t;t.heading_correction=true;t.max_linear=.8;t.xy_tolerance=.12;
    t.max_segment_drift=.12;t.segment_capture=.04;
    t.setPlan({{0,0,0},{5,0,0}});
    TrackingPoint pose{0,0,0};TrackingVelocity smooth,actual,desired;
    double maximum_cross=0;bool injected=false,corrected=false,complete=false;
    for (int tick=0;tick<2500;++tick) {
      if (tick%7==0) {
        desired=t.command(pose,smooth,.8,.07,actual);
        if (pose.x<4.5) {
          EXPECT_EQ(desired.y,0);
          corrected=corrected || (desired.x>0 && std::abs(desired.yaw)>.01);
          maximum_cross=std::max(maximum_cross,std::abs(pose.y));
        }
      }
      smooth.x+=std::clamp(desired.x-smooth.x,-.0075,.0065);
      smooth.y+=std::clamp(desired.y-smooth.y,-.002,.002);
      smooth.yaw+=std::clamp(desired.yaw-smooth.yaw,-.009,.008);
      actual.x+=(smooth.x-actual.x)*(1-std::exp(-.01/.15));
      actual.y+=(smooth.y-actual.y)*(1-std::exp(-.01/.15));
      actual.yaw+=(smooth.yaw-actual.yaw)*(1-std::exp(-.01/.15));
      pose.x+=(std::cos(pose.yaw)*actual.x-std::sin(pose.yaw)*actual.y)*.01;
      pose.y+=(std::sin(pose.yaw)*actual.x+std::cos(pose.yaw)*actual.y+sign*.008*actual.x)*.01;
      pose.yaw+=(actual.yaw+sign*.012*actual.x)*.01;
      if (!injected && pose.x>1.) {pose.y+=sign*.015;pose.yaw+=sign*.025;injected=true;}
      if (std::hypot(pose.x-5,pose.y)<.12 && std::abs(pose.yaw)<t.yaw_tolerance &&
        std::hypot(actual.x,actual.y)<.02 && std::abs(actual.yaw)<.02) {complete=true;break;}
    }
    EXPECT_TRUE(injected);EXPECT_TRUE(corrected);EXPECT_TRUE(complete);
    EXPECT_LT(maximum_cross,.05);
  }
}

TEST(PathTracking, HeadingCorrectionDoesNotLookBeyondShortCornerEdge)
{
  PathTracking t;t.heading_correction=true;t.segment_capture=.04;
  t.setPlan({{0,0,0},{.10,0,M_PI/2},{.10,2,M_PI/2}});
  bool checked=false;
  const auto v=t.command({0,0,0},{},.8,.067,{},
    [&](const TrackingPoint &,const TrackingPoint & aim) {
      checked=true;EXPECT_NEAR(aim.x,.10,1e-9);EXPECT_NEAR(aim.y,0,1e-9);return true;
    });
  EXPECT_TRUE(checked);EXPECT_GT(v.x,0);EXPECT_EQ(v.y,0);EXPECT_EQ(v.yaw,0);
}

TEST(PathTracking, StoppedAlignmentBrakesBeforeMixingLateralAndYaw)
{
  PathTracking t;t.stopped_alignment=true;t.max_segment_drift=.12;t.segment_capture=.04;
  t.setPlan({{0,0,0},{3,0,0}});
  auto v=t.command({.5,.06,.15},{.4,0,0},.8);
  EXPECT_EQ(v.x,0);EXPECT_EQ(v.y,0);EXPECT_EQ(v.yaw,0);EXPECT_FALSE(t.aligningReference());
  v=t.command({.5,.06,.15},{},.8);
  EXPECT_TRUE(t.aligningReference());EXPECT_EQ(v.x,0);EXPECT_LT(v.y,0);EXPECT_LT(v.yaw,0);
  EXPECT_NEAR(t.remainingAlignmentLateral(),-.06/std::cos(.15),1e-9);
}

TEST(PathTracking, StoppedAlignmentRequiresPositionHeadingAndSettlingBeforeCruise)
{
  PathTracking t;t.stopped_alignment=true;t.max_segment_drift=.12;t.segment_capture=.04;
  t.setPlan({{0,0,0},{3,0,0}});
  auto v=t.command({0,.05,0},{},.8);
  EXPECT_LT(v.y,0);EXPECT_EQ(v.x,0);EXPECT_EQ(v.yaw,0);
  for (int i=0;i<4;++i) {
    v=t.command({0,.01,.005},{0,-.03,.03},.8);
    EXPECT_EQ(v.x,0);
  }
  for (int i=0;i<5;++i) {v=t.command({0,.01,.005},{},.8);}
  EXPECT_GT(v.x,0);EXPECT_EQ(v.y,0);EXPECT_EQ(v.yaw,0);
  v=t.command({1,.025,.02},{.6,0,0},.8);
  EXPECT_GT(v.x,0);EXPECT_EQ(v.y,0);EXPECT_EQ(v.yaw,0);
  v=t.command({1.1,.045,.02},{.6,0,0},.8);
  EXPECT_EQ(v.x,0);EXPECT_EQ(v.y,0);EXPECT_EQ(v.yaw,0);
  v=t.command({1.2,.045,.02},{.3,0,0},.8);
  EXPECT_EQ(v.x,0);EXPECT_EQ(v.y,0);EXPECT_EQ(v.yaw,0);
  v=t.command({1.25,.045,.02},{},.8);
  EXPECT_EQ(v.x,0);EXPECT_LT(v.y,0);EXPECT_LT(v.yaw,0);
}

TEST(PathTracking, StoppedAlignmentReturnsToOriginalLineWithDeadzoneAndLag)
{
  for (double sign:{-1.,1.}) {
    for (double response:{.6,1.}) {
      PathTracking t;t.stopped_alignment=true;t.max_segment_drift=.12;t.segment_capture=.04;
      t.setPlan({{0,0,0},{4,0,0}});
      TrackingPoint pose{.5,sign*.065,sign*.14};
      TrackingVelocity smooth,actual,desired;bool mixed=false,cruise=false;
      for (int tick=0;tick<2000;++tick) {
        if (tick%7==0) {
          desired=t.command(pose,smooth,.8,.07,actual);
          if (desired.x>0) {
            EXPECT_LE(std::abs(pose.y),t.alignment_position_tolerance);
            EXPECT_LE(std::abs(pose.yaw),t.alignment_heading_tolerance);
            EXPECT_EQ(desired.y,0);EXPECT_EQ(desired.yaw,0);cruise=true;break;
          }
          EXPECT_EQ(desired.x,0);
          mixed=mixed || (std::abs(desired.y)>.05 && std::abs(desired.yaw)>.1);
        }
        smooth.y+=std::clamp(desired.y-smooth.y,-.002,.002);
        smooth.yaw+=std::clamp(desired.yaw-smooth.yaw,-.009,.008);
        const double motor_y=std::abs(smooth.y)<.04 ? 0 : response*smooth.y;
        const double motor_w=std::abs(smooth.yaw)<.22 ? 0 : response*smooth.yaw;
        actual.y+=(motor_y-actual.y)*(1-std::exp(-.01/.12));
        actual.yaw+=(motor_w-actual.yaw)*(1-std::exp(-.01/.12));
        pose.x-=std::sin(pose.yaw)*actual.y*.01;
        pose.y+=std::cos(pose.yaw)*actual.y*.01;
        pose.yaw+=actual.yaw*.01;
      }
      EXPECT_TRUE(mixed);EXPECT_TRUE(cruise)<<sign<<" "<<response<<" "<<pose.y<<" "<<pose.yaw;
    }
  }
}

TEST(PathTracking, StoppedAlignmentBlocksUnsafeStraightRayAfterSettling)
{
  PathTracking t;t.stopped_alignment=true;t.setPlan({{0,0,0},{3,0,0}});
  EXPECT_THROW(t.command({0,0,0},{},.8,.067,{},
    [](const TrackingPoint &,const TrackingPoint &){return false;}),std::runtime_error);
  t.heading_correction=true;EXPECT_FALSE(t.valid());
}

TEST(PathTracking, TolerantAlignmentKeepsReferenceWithoutChasingSmallErrors)
{
  PathTracking t;t.stopped_alignment=true;t.max_segment_drift=.12;t.segment_capture=.08;
  t.alignment_position_tolerance=.04;t.alignment_heading_tolerance=.035;
  t.realign_cross_threshold=.08;t.realign_heading_threshold=.087;
  ASSERT_TRUE(t.valid());
  const std::vector<TrackingPoint> plan{{0,0,0},{4,0,0}};t.setPlan(plan);
  TrackingVelocity v;
  for (int i=0;i<5;++i) {v=t.command({.2,.03,.025},{},.8);}
  EXPECT_GT(v.x,0);EXPECT_EQ(v.y,0);EXPECT_EQ(v.yaw,0);
  // Cross the acquisition tolerance repeatedly, while still inside the
  // wider driving band; neither jitter nor identical replans restart it.
  for (double cross:{.035,.045,.039,.055,.04,.065}) {
    t.setPlan(plan);v=t.command({1,cross,.04},{.5,0,0},.8);
    EXPECT_GT(v.x,0);EXPECT_EQ(v.y,0);EXPECT_EQ(v.yaw,0);
  }
  EXPECT_EQ(t.clearanceRealignments(),0u);
  v=t.command({1.2,.081,.04},{.5,0,0},.8);
  EXPECT_EQ(v.x,0);EXPECT_EQ(v.y,0);EXPECT_EQ(v.yaw,0);
  EXPECT_EQ(t.clearanceRealignments(),1u);
  v=t.command({1.3,.085,.04},{},.8);
  EXPECT_EQ(v.x,0);EXPECT_LT(v.y,0);EXPECT_LT(v.yaw,0);
  // It must reacquire the original line, rather than moving the line to the
  // new stopped pose. A small residual offset is intentionally accepted.
  for (int i=0;i<5;++i) {v=t.command({1.3,.03,.025},{},.8);}
  EXPECT_GT(v.x,0);EXPECT_EQ(v.y,0);EXPECT_EQ(v.yaw,0);
  EXPECT_THROW(t.command({2,.13,0},{},.8),std::runtime_error);
}

TEST(PathTracking, TolerantAlignmentStillHonorsBlockedForwardCorridor)
{
  PathTracking t;t.stopped_alignment=true;t.max_segment_drift=.12;
  t.alignment_position_tolerance=.04;t.alignment_heading_tolerance=.035;
  t.realign_cross_threshold=.08;t.realign_heading_threshold=.087;
  t.setPlan({{0,0,0},{4,0,0}});
  const auto blocked=[](const TrackingPoint &,const TrackingPoint &){return false;};
  auto v=t.command({.2,.03,.025},{},.8,.067,{},blocked);
  EXPECT_EQ(v.x,0);EXPECT_LT(v.y,0);EXPECT_EQ(v.yaw,0);
  // Even inside half the normal acceptance band, try centering before
  // abandoning the reference if the forward clearance is slightly short.
  v=t.command({.2,.0155,.0093},{},.8,.067,{},blocked);
  EXPECT_EQ(v.x,0);EXPECT_LT(v.y,0);EXPECT_EQ(v.yaw,0);
  // If a physically checked corridor clears, use the normal tolerance again;
  // do not continue chasing an absolute line/angle after clearance returns.
  for(int i=0;i<5;++i) {
    v=t.command({.2,.025,.02},{},.8,.067,{},
      [](const TrackingPoint &,const TrackingPoint &){return true;});
  }
  EXPECT_GT(v.x,0);EXPECT_EQ(v.y,0);EXPECT_EQ(v.yaw,0);
  v=t.command({.4,.025,.02},{.5,0,0},.8,.067,{},blocked);
  EXPECT_EQ(v.x,0);EXPECT_EQ(v.y,0);EXPECT_EQ(v.yaw,0);
  EXPECT_THROW(t.command({.5,0,0},{},.8,.067,{},blocked),std::runtime_error);
}
