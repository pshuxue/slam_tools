#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h> //体素滤波相关

#include <iostream>

template <typename PointCloudPtrType>
void show_point_cloud(PointCloudPtrType cloud, std::string display_name)
{
    pcl::visualization::CloudViewer viewer(display_name);
    viewer.showCloud(cloud);
    while (!viewer.wasStopped())
    {
    }
}

int main()
{
    //点云读取
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
    reader.read("/home/pengshuxue/Public/slam_tools/pcl/DBscan_cluster/resources/table_scene_lms400.pcd", *cloud);
    show_point_cloud(cloud, "orgin cloud");

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filter(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.1, 0.1, 0.1); //滤波的分辨率
    sor.filter(*cloud_filter);

    show_point_cloud(cloud_filter, "cloud after filter");
}