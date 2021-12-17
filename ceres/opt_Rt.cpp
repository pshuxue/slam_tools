#include <iostream>
#include <ceres/ceres.h>
#include <Rt_optimizer.h>
#include <opencv2/highgui.hpp>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <chrono>
using namespace std;
using namespace chrono;
using namespace cv;

const int num_points = 100;
vector<Eigen::Vector3d> points; //世界坐标
vector<cv::Point3d> pts_3d;
vector<Eigen::Vector2d> pixs; //像素坐标
vector<cv::Point2d> pts_2d;
vector<Eigen::Vector2d> normalized_pts; //归一化坐标

//位姿的真实值
Eigen::Quaterniond std_qwc;
Eigen::Vector3d std_twc;
Eigen::Quaterniond std_qcw;
Eigen::Vector3d std_tcw;

//位姿的估计值
cv::Mat rvec, tvec;

Mat K;
Eigen::Matrix3d K_eigen;

void GetTestData();               //获取测试数据
void TestOpencvPNP();             //测试opencv的pnp算法
void OptimizeRtByCeresAutoDiff(); //使用ceres对opencv的结果进行优化
void OptimizeRtByCeresNumericDiff();

int main(int, char **)
{
    K = (Mat_<double>(3, 3) << 400, 0, 300, 0, 401, 302, 0, 0, 1);
    K_eigen << 400, 0, 300, 0, 401, 302, 0, 0, 1;

    GetTestData();

    TestOpencvPNP();

    OptimizeRtByCeresAutoDiff();
    OptimizeRtByCeresNumericDiff();
}

void GetTestData()
{
    Eigen::AngleAxisd vec(1.1, Eigen::Vector3d(13, 2, 10).normalized());
    std_qwc = Eigen::Quaterniond(vec);
    std_twc = Eigen::Vector3d(-1, -1, 0);
    std_qcw = std_qwc.inverse();
    std_tcw = -1 * std_qcw.toRotationMatrix() * std_twc;

    points.resize(num_points);
    pixs.resize(num_points);
    for (int i = 0; i < num_points; ++i)
    {
        pixs[i] = (Eigen::Vector2d::Random() + Eigen::Vector2d(1, 1)) * 600;
        Eigen::Vector3d p_normalized = K_eigen.inverse() * Eigen::Vector3d(pixs[i].x(), pixs[i].y(), 1);
        double depth = (rand() % 10000) / 1000.0 + 1;
        Eigen::Vector3d p_c = p_normalized * depth;
        // pixs[i] += Eigen::Vector2d::Random() * 5; //add noise
        points[i] = std_qwc.toRotationMatrix() * p_c + std_twc + Eigen::Vector3d::Random() * 0.1;
    }

    for (auto &p : points)
    {
        pts_3d.push_back(cv::Point3d(p.x(), p.y(), p.z()));
    }
    for (auto &p : pixs)
    {
        pts_2d.push_back(cv::Point2d(p.x(), p.y()));
    }
    cout << "真实旋转四元数qwc:  " << std_qwc.w() << " " << std_qwc.vec().transpose() << endl;
    cout << "真实旋转四元数qcw:  " << std_qcw.w() << " " << std_qcw.vec().transpose() << endl;
    cout << "真实平移twc:  " << std_twc.transpose() << endl;
    cout << "真实平移tcw:  " << std_tcw.transpose() << endl
         << endl;
}

//cv中的pnp函数，求解的是欧拉角和平移
void TestOpencvPNP()
{
    //设置内参和畸变系数
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
    // solvePnP(pts_3d, pts_2d, K, distCoeffs, rvec, tvec, false, CV_EPNP);

    cv::Mat inliners;
    solvePnPRansac(pts_3d, pts_2d, K, distCoeffs, rvec, tvec, false, 100, 10, 0.99, inliners); //inliners是int型Mat，是索引表

    Eigen::Quaterniond quaternion;
    quaternion = Eigen::AngleAxisd(rvec.at<double>(0, 0), Eigen::Vector3d::UnitZ()) *
                 Eigen::AngleAxisd(rvec.at<double>(1, 0), Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(rvec.at<double>(2, 0), Eigen::Vector3d::UnitX());

    cout << "opencv CV_EPNP计算四元数qcw:  " << quaternion.w() << " " << quaternion.vec().transpose() << endl;
    cout << "opencv CV_EPNP计算四元数tcw:  " << tvec.at<double>(0, 0) << " " << tvec.at<double>(1, 0) << " " << tvec.at<double>(2, 0) << endl
         << endl;
}

//ceres中的优化函数，优化量是李代数和平移
void OptimizeRtByCeresAutoDiff()
{
    auto start = system_clock::now();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t(0, 0, 1);

    RtOptimizer optimizer;
    optimizer.SetInitialVal(R, t);
    for (int i = 0; i < pts_2d.size(); ++i)
    {
        Eigen::Vector3d xyz(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z);
        Eigen::Vector2d uv(pts_2d[i].x, pts_2d[i].y);
        optimizer.AddResidualItemAutoDiff(uv, xyz, K_eigen);
    }
    optimizer.Solve();

    Eigen::Matrix4d Tcw = optimizer.GetTcw();
    Eigen::Quaterniond qcw = Eigen::Quaterniond(Eigen::Matrix3d(Tcw.topLeftCorner(3, 3))); //再转四元数
    cout << "自动求导，优化后 qcw " << qcw.w() << " " << qcw.vec().transpose() << endl;
    Eigen::Vector3d tcw = Tcw.topRightCorner(3, 1);
    cout << "自动求导，优化后 tcw=" << tcw.transpose() << endl;
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "花费了"
         << double(duration.count()) * microseconds::period::num / microseconds::period::den
         << "秒" << endl;
}

//ceres中的优化函数，优化量是李代数和平移
void OptimizeRtByCeresNumericDiff()
{
    auto start = system_clock::now();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t(0, 0, 1);
    RtOptimizer optimizer;
    optimizer.SetInitialVal(R, t);
    for (int i = 0; i < pts_2d.size(); ++i)
    {
        Eigen::Vector3d xyz(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z);
        Eigen::Vector2d uv(pts_2d[i].x, pts_2d[i].y);
        optimizer.AddResidualItemNumericDiff(uv, xyz, K_eigen);
    }
    optimizer.Solve();

    Eigen::Matrix4d Tcw = optimizer.GetTcw();
    Eigen::Quaterniond qcw = Eigen::Quaterniond(Eigen::Matrix3d(Tcw.topLeftCorner(3, 3))); //再转四元数
    cout << "手动求导，优化后 qcw " << qcw.w() << " " << qcw.vec().transpose() << endl;
    Eigen::Vector3d tcw = Tcw.topRightCorner(3, 1);
    cout << "手动求导，优化后 tcw=" << tcw.transpose() << endl;
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "花费了"
         << double(duration.count()) * microseconds::period::num / microseconds::period::den
         << "秒" << endl;
}