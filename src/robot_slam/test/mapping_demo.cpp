/**
 * @file mapping_demo.cpp
 * @brief
 * @author Liuzhao Li (liliuzhao@jushenzhiren.com)
 * @version 1.0
 * @date 2025-07-31
 * @copyright Copyright (C) 2025 具身智人(北京)科技有限公司
 */

#include "mapping_alg.h"
#include <thread>
static std::atomic<bool> flag{ false };
void                     SigHandle(int sig)
{
    if (!flag.exchange(true, std::memory_order_relaxed))
    {
        rclcpp::shutdown();
    }
}
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::signal(SIGINT, SigHandle);
    std::signal(SIGTERM, SigHandle);
    auto         node = std::make_shared<robot::slam::MappingAlg>();
    std::thread mapping_worker([&]() {
        rclcpp::Rate rate(1000);
        while (rclcpp::ok() && !flag.load(std::memory_order_relaxed))
        {
            node->run();
            rate.sleep();
        }
    });

    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
    if (mapping_worker.joinable())
        mapping_worker.join();
    node->finish();
    return 0;
}
