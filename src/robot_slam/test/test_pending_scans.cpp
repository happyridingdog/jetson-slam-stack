#include <gtest/gtest.h>
#include <deque>
#include "pending_scans.h"

TEST(PendingScans, WaitingForImuCannotConsumeWrongScanAfterOverflow)
{
    std::deque<int> clouds{1}; std::deque<double> stamps{.1};
    bool selected=true; int measurement=clouds.front();
    clouds.push_back(2); stamps.push_back(.2);
    EXPECT_EQ(robot::slam::trim_pending_scans(clouds,stamps,1,selected),1u);
    // syncData must reselect before integrating IMU and removing the front.
    ASSERT_FALSE(selected);
    if (!selected) {measurement=clouds.front();selected=true;}
    EXPECT_EQ(measurement,2); EXPECT_EQ(stamps.front(),.2);
    clouds.pop_front(); stamps.pop_front();
    EXPECT_TRUE(clouds.empty()); EXPECT_TRUE(stamps.empty());
}

TEST(PendingScans, DefaultQueueRetainsHistoryAndSelectionUntilOverflow)
{
    std::deque<int> clouds{1,2,3,4,5}; std::deque<double> stamps{.1,.2,.3,.4,.5};
    bool selected=true;
    EXPECT_EQ(robot::slam::trim_pending_scans(clouds,stamps,5,selected),0u);
    EXPECT_TRUE(selected); EXPECT_EQ(clouds.front(),1);
    clouds.push_back(6); stamps.push_back(.6);
    EXPECT_EQ(robot::slam::trim_pending_scans(clouds,stamps,5,selected),1u);
    EXPECT_FALSE(selected); EXPECT_EQ(clouds.front(),2); EXPECT_EQ(stamps.front(),.2);
}
