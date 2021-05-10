#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <time.h>
#include <iostream>

#include "DBSCAN_kdtree.h"

template <typename PointCloudPtrType>
void show_point_cloud(PointCloudPtrType cloud, std::string display_name)
{
    pcl::visualization::CloudViewer viewer(display_name);
    viewer.showCloud(cloud);
    while (!viewer.wasStopped())
    {
    }
}

int main(int argc, char **argv)
{
    //点云读取
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
    reader.read("/home/pengshuxue/Public/slam_tools/pcl/DBscan_cluster/resources/table_scene_lms400.pcd", *cloud);

    //建立变量保存最后提取结果
    std::vector<pcl::PointIndices> cluster_indices;

    //建立DBSCAN聚类器
    DBSCANKdtreeCluster<pcl::PointXYZ> ec;
    ec.setCorePointMinPts(10);    //搜索半径内点数少于这个那就认为这个点是噪声
    ec.setClusterTolerance(0.02); //搜索半径
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(2500000);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices); //结果

    //将点云分类保存
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_clustered(new pcl::PointCloud<pcl::PointXYZI>);
    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++, j++)
    {
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
        {
            pcl::PointXYZI tmp;
            tmp.x = cloud->points[*pit].x;
            tmp.y = cloud->points[*pit].y;
            tmp.z = cloud->points[*pit].z;
            tmp.intensity = j % 8;
            cloud_clustered->points.push_back(tmp);
        }
    }

    //可视化
    cloud_clustered->width = cloud_clustered->points.size();
    cloud_clustered->height = 1;
    show_point_cloud(cloud_clustered, "colored clusters of point cloud");

    return 0;
}