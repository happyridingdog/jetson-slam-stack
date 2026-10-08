#include "footprint_filter.hpp"
#include <gtest/gtest.h>
TEST(FootprintFilter, SensorExtrinsicAndBoundary) {
  FootprintFilter filter;filter.validate();
  EXPECT_TRUE(filter.contains(0,0));EXPECT_TRUE(filter.contains(-.50,0));
  EXPECT_TRUE(filter.contains(.09,.225));EXPECT_TRUE(filter.contains(-.51,-.225));
  EXPECT_FALSE(filter.contains(.10,0));EXPECT_FALSE(filter.contains(-.52,0));
  EXPECT_FALSE(filter.contains(0,.226));EXPECT_FALSE(filter.contains(0,-.226));
}
TEST(FootprintFilter, RotatedSensor) {
  FootprintFilter filter;filter.base_from_sensor={0,0,1.5707963267948966};filter.validate();
  EXPECT_TRUE(filter.contains(0,.29));EXPECT_FALSE(filter.contains(.29,0));
}
TEST(FootprintFilter, ConcavePolygon) {
  FootprintFilter filter;filter.base_from_sensor={0,0,0};
  filter.polygon={0,0,2,0,2,1,1,1,1,2,0,2};filter.validate();
  EXPECT_TRUE(filter.contains(.5,1.5));EXPECT_FALSE(filter.contains(1.5,1.5));
}
TEST(FootprintFilter, InvalidConfiguration) {
  FootprintFilter filter;filter.polygon={0,0,1};EXPECT_THROW(filter.validate(),std::invalid_argument);
  filter.polygon={0,0,1,0,1,1};filter.base_from_sensor={0,0};EXPECT_THROW(filter.validate(),std::invalid_argument);
}
TEST(FootprintFilter, OutsideVoxelSamplesCanHaveInsideCentroid) {
  FootprintFilter filter;filter.validate();
  // Same 0.10 m sensor voxel, outside opposite sides of the front-left corner.
  EXPECT_FALSE(filter.contains(.095,.205));
  EXPECT_FALSE(filter.contains(.075,.235));
  EXPECT_TRUE(filter.contains((.095+.075)/2.,(.205+.235)/2.));
}
