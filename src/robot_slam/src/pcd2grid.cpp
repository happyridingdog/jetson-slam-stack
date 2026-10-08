
#include "pcd2grid.h"
#include <filesystem>

namespace robot::slam
{
    Pcd2Grid::Pcd2Grid(const Pcd2GridOptions &options) : options_(options)
    {
    }

    void Pcd2Grid::run(const CloudPtr &pcd_cloud, const std::string &file_name)
    {
        CloudPtr cloud_after_pass_through = CloudPtr(new PointCloudType());
        CloudPtr cloud_after_radius = CloudPtr(new PointCloudType());
        nav_msgs::msg::OccupancyGrid map_topic_msg;

        PassThroughFilter(pcd_cloud, cloud_after_pass_through);
        RadiusOutlierFilter(cloud_after_pass_through, cloud_after_radius);
        SetMapTopicMsg(cloud_after_radius, map_topic_msg);
        SavePGMAndYAML(map_topic_msg, file_name);
    }
    void Pcd2Grid::PassThroughFilter(const CloudPtr &pcd_cloud, CloudPtr &cloud_after_pass_through)
    {
        pcl::PassThrough<PointType> passthrough;
        passthrough.setInputCloud(pcd_cloud);
        passthrough.setFilterFieldName("z");
        passthrough.setFilterLimits(options_.thre_z_min, options_.thre_z_max);
        // passthrough.setFilterLimitsNegative(flag_in);
        passthrough.setNegative(bool(options_.flag_pass_through));
        passthrough.filter(*cloud_after_pass_through);
        // pcl::io::savePCDFile<PointType>(options_.file_name + "_filter.pcd",
        //                                 *cloud_after_pass_through);
        // std::cout << "Point cloud size after passthrough filter: "
        //           << cloud_after_pass_through->points.size() << std::endl;
    }

    void Pcd2Grid::RadiusOutlierFilter(const CloudPtr &pcd_cloud, CloudPtr &cloud_after_radius)
    {
        pcl::RadiusOutlierRemoval<PointType> radiusoutlier;
        radiusoutlier.setInputCloud(pcd_cloud);
        radiusoutlier.setRadiusSearch(options_.thre_radius);
        radiusoutlier.setMinNeighborsInRadius(options_.thres_point_count);
        radiusoutlier.filter(*cloud_after_radius);
        // pcl::io::savePCDFile<PointType>(options_.file_name + "_radius_filter.pcd",
        //                                 *cloud_after_radius);
        // std::cout << "Point cloud size after radius filter: "
        //           << cloud_after_radius->points.size() << std::endl;
    }

    void Pcd2Grid::SetMapTopicMsg(const CloudPtr cloud, nav_msgs::msg::OccupancyGrid &msg)
    {
        msg.header.stamp = rclcpp::Clock().now();
        msg.header.frame_id = "map";
        msg.info.map_load_time = rclcpp::Clock().now();
        msg.info.resolution = options_.map_resolution;

        double x_min, x_max, y_min, y_max;
        double z_max_grey_rate = 0.05;
        double z_min_grey_rate = 0.95;
        double k_line =
            (z_max_grey_rate - z_min_grey_rate) / (options_.thre_z_max - options_.thre_z_min);
        double b_line =
            (options_.thre_z_max * z_min_grey_rate - options_.thre_z_min * z_max_grey_rate) /
            (options_.thre_z_max - options_.thre_z_min);

        if (cloud->points.empty())
        {
            RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "PCD is empty!");
            return;
        }

        for (size_t i = 0; i < cloud->points.size() - 1; i++)
        {
            if (i == 0)
            {
                x_min = x_max = cloud->points[i].x;
                y_min = y_max = cloud->points[i].y;
            }

            double x = cloud->points[i].x;
            double y = cloud->points[i].y;

            if (x < x_min)
                x_min = x;
            if (x > x_max)
                x_max = x;

            if (y < y_min)
                y_min = y;
            if (y > y_max)
                y_max = y;
        }

        msg.info.origin.position.x = x_min;
        // msg.info.origin.position.y = y_min;
        int h = int((y_max - y_min) / options_.map_resolution);
        msg.info.origin.position.y = y_max - h * options_.map_resolution;
        msg.info.origin.position.z = 0.0;
        msg.info.origin.orientation.w = 1.0;

        msg.info.width = int((x_max - x_min) / options_.map_resolution);
        msg.info.height = int((y_max - y_min) / options_.map_resolution);
        msg.data.resize(msg.info.width * msg.info.height);
        msg.data.assign(msg.info.width * msg.info.height, 0);

        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Data size: %ld", msg.data.size());

        for (size_t iter = 0; iter < cloud->points.size(); iter++)
        {
            int i = int((cloud->points[iter].x - x_min) / options_.map_resolution);
            if (i < 0 || i >= msg.info.width)
                continue;

            int j = int((cloud->points[iter].y - y_min) / options_.map_resolution);
            if (j < 0 || j >= msg.info.height - 1)
                continue;

            msg.data[i + j * msg.info.width] = 100;
        }
    }

    void Pcd2Grid::SavePGMAndYAML(const nav_msgs::msg::OccupancyGrid &msg, const std::string &name)
    {
        int width = msg.info.width;
        int height = msg.info.height;

        // Save a binary PGM without pulling OpenCV into the deployment image.
        const std::string pgm_file = name + ".pgm";
        const std::string pgm_tmp = pgm_file + ".tmp";
        std::ofstream pgm_output(pgm_tmp, std::ios::binary);
        pgm_output << "P5\n" << width << " " << height << "\n255\n";
        for (int y = height - 1; y >= 0; --y)
        {
            for (int x = 0; x < width; ++x)
            {
                const int8_t data = msg.data[x + y * width];
                const unsigned char pixel = data < 0 ? 205 :
                    static_cast<unsigned char>(255 - data * 255 / 100);
                pgm_output.write(reinterpret_cast<const char *>(&pixel), 1);
            }
        }
        pgm_output.close();
        std::error_code rename_ec;
        std::filesystem::rename(pgm_tmp, pgm_file, rename_ec);
        if (rename_ec)
            RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "Could not install PGM atomically: %s", rename_ec.message().c_str());
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Saved PGM file: %s", pgm_file.c_str());

        // Save YAML
        const std::string yaml_file = name + ".yaml";
        const std::string yaml_tmp = yaml_file + ".tmp";
        std::ofstream yaml_output(yaml_tmp);
        yaml_output << "image: " << options_.file_name << ".pgm" << std::endl;
        yaml_output << "resolution: " << msg.info.resolution << std::endl;
        yaml_output << "origin: [" << msg.info.origin.position.x << ", "
                    << msg.info.origin.position.y << ", "
                    << msg.info.origin.position.z << "]" << std::endl;
        yaml_output << "negate: 0" << std::endl;
        yaml_output << "occupied_thresh: 0.65" << std::endl;
        yaml_output << "free_thresh: 0.196" << std::endl;
        yaml_output.close();
        rename_ec.clear();
        std::filesystem::rename(yaml_tmp, yaml_file, rename_ec);
        if (rename_ec)
            RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "Could not install YAML atomically: %s", rename_ec.message().c_str());

        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Saved YAML file: %s", yaml_file.c_str());
    }
}
