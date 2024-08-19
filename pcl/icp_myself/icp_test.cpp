#include <Eigen/Eigen>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/gicp.h>

class IcpSolver
{
private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_, src_cloud_;

    const int max_iter_number = 20;
    const int src_ignore_point = 2;

    const float angle_th = 0.1;
    const float trans_th = 0.002;

public:
    IcpSolver() {}
    ~IcpSolver() {}
    void SetTargetCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud) { target_cloud_ = target_cloud; }

    void SetSrcCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud) { src_cloud_ = src_cloud; }

    Eigen::Matrix4f GetT_Target_Src()
    {
        Eigen::Matrix4f T_tar_src_res = Eigen::Matrix4f::Identity();
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(target_cloud_);

        int iter_number = 0;
        pcl::PointCloud<pcl::PointXYZ>::Ptr src_iter_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (int i = 0; i < src_cloud_->size(); i += src_ignore_point)
        {
            src_iter_cloud->push_back(src_cloud_->points[i]);
        }

        while (++iter_number <= max_iter_number)
        {
            std::vector<pcl::PointXYZ> tar_points;
            for (int i = 0; i < src_iter_cloud->size(); i++)
            {
                pcl::PointXYZ &searchPoint = src_iter_cloud->points[i];
                std::vector<int> pointIdxKNNSearch(1);
                std::vector<float> pointKNNSquaredDistance(1);
                kdtree.nearestKSearch(searchPoint, 1, pointIdxKNNSearch, pointKNNSquaredDistance);
                pcl::PointXYZ target_point = target_cloud_->points[pointIdxKNNSearch[0]];
                tar_points.push_back(target_point);
            }

            Eigen::Matrix4f T_tar_src = ComputeTrans(src_iter_cloud, tar_points);
            for (auto &pl : src_iter_cloud->points)
            {
                Eigen::Map<Eigen::Vector3f> p(pl.data);
                p = T_tar_src.topLeftCorner(3, 3) * p + T_tar_src.topRightCorner(3, 1);
            }

            T_tar_src_res = T_tar_src * T_tar_src_res;

            Eigen::Matrix3f R_tar_src = T_tar_src.topLeftCorner(3, 3);
            Eigen::AngleAxisf vec(R_tar_src);

            float angle = vec.angle() * 180.0 / M_PI;
            float trans = T_tar_src.topRightCorner(3, 1).norm();
            std::cout << "iter " << iter_number << ", angle error = " << angle << ", trans error = " << trans << std::endl;

            if (trans < trans_th && angle < angle_th)
                break;
        }

        return T_tar_src_res;
    }


    //这个函数肯定可以用，根据两个对应的点云求解Rt！！！！
    Eigen::Matrix4f ComputeTrans(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, const std::vector<pcl::PointXYZ> &tar_points)
    {
        if (src_cloud->size() != tar_points.size())
            throw std::runtime_error("src_cloud size != tar_points.size");

        Eigen::Matrix3d R_tar_src;
        Eigen::Vector3d t_tar_src;

        Eigen::Vector3f p_sum(0, 0, 0);
        for (auto &p : src_cloud->points)
        {
            Eigen::Map<Eigen::Vector3f> pp(p.data);
            p_sum += pp;
        }
        Eigen::Vector3f mean_p_src = p_sum / src_cloud->points.size();

        p_sum = Eigen::Vector3f(0, 0, 0);
        for (const auto &p : tar_points)
        {
            Eigen::Map<const Eigen::Vector3f> pp(p.data);
            p_sum += pp;
        }
        Eigen::Vector3f mean_p_tar = p_sum / tar_points.size();

        Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
        for (int i = 0; i < tar_points.size(); i++)
        {
            Eigen::Vector3d q_tar_1(tar_points[i].x - mean_p_tar.x(), tar_points[i].y - mean_p_tar.y(), tar_points[i].z - mean_p_tar.z());
            Eigen::Vector3d q_src_1(src_cloud->points[i].x - mean_p_src.x(), src_cloud->points[i].y - mean_p_src.y(),
                                    src_cloud->points[i].z - mean_p_src.z());
            W = W + q_tar_1 * q_src_1.transpose();
        }

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullV | Eigen::ComputeFullU);
        R_tar_src = svd.matrixU() * svd.matrixV().transpose();
        t_tar_src = mean_p_tar.cast<double>() - R_tar_src * mean_p_src.cast<double>();

        Eigen::Matrix4f T_res = Eigen::Matrix4f::Identity();
        T_res.topLeftCorner(3, 3) = R_tar_src.cast<float>();
        T_res.topRightCorner(3, 1) = t_tar_src.cast<float>();
        return T_res;
    }
};

int main(int argc, char **argv)
{
    IcpSolver solver;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tar(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::io::loadPLYFile("/home/psx/workspace/slam_tools/pcl/icp_myself/1.ply", *cloud_src);
    pcl::io::loadPLYFile("/home/psx/workspace/slam_tools/pcl/icp_myself/2.ply", *cloud_tar);

    solver.SetSrcCloud(cloud_src);
    solver.SetTargetCloud(cloud_tar);
    Eigen::Matrix4f T1 = solver.GetT_Target_Src();

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputCloud(cloud_src);
    icp.setInputTarget(cloud_tar);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    Eigen::Matrix4f T2 = icp.getFinalTransformation();

    Eigen::Matrix4f dT = T1.inverse() * T2;
    Eigen::Matrix3f dR = dT.topLeftCorner(3, 3);
    Eigen::AngleAxisf vec(dR);

    std::cout << vec.angle() * 180 / M_PI << " " << dT.topRightCorner(3, 1).norm() << std::endl;
    std::cout << T1 << std::endl;
    std::cout << T2 << std::endl;

    return 0;
}