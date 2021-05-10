#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

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

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filter(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_Indices(new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02); //平面的阈值
    seg.setInputCloud(cloud);
    seg.segment(*inliers_Indices, *coefficients);

    for (auto &p : cloud->points)
    {
        pcl::PointXYZRGB point(100, 100, 100);
        point.x = p.x;
        point.y = p.y;
        point.z = p.z;
        cloud_filter->points.push_back(point);
    }

    for (auto &index : inliers_Indices->indices)
    {
        pcl::PointXYZRGB point(255, 0, 0);
        point.x = cloud->points[index].x;
        point.y = cloud->points[index].y;
        point.z = cloud->points[index].z;
        cloud_filter->points.push_back(point);
    }
    show_point_cloud(cloud_filter, "plane ");
}