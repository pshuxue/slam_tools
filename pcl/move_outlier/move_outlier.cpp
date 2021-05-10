#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>

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
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);          //设置在进行统计时考虑查询点邻近点数
    sor.setStddevMulThresh(1); //设置判断是否为离群点的倒值,x = 1.0, t = mean+/-(stddev*x) 越小滤除的越狠
    sor.filter(*cloud_filter);
    show_point_cloud(cloud_filter, "cloud after filter");
}