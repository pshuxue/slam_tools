#include <iostream>
#include <eigen3/Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/eigen.hpp>
#include <chrono>


struct PnPCeres
{
    PnPCeres(Eigen::Vector2d uv, Eigen::Vector3d xyz,Eigen::Matrix3d K) : _uv(uv), _xyz(xyz),_K(K) {}
    // 残差的计算
    template <typename T>
    bool operator()(
        const T *const camera, // 位姿参数，有6维
        T *residual) const     // 残差
    {
        T p[3];
        T point[3];
        point[0] = T(_xyz.x());
        point[1] = T(_xyz.y());
        point[2] = T(_xyz.z());
        ceres::AngleAxisRotatePoint(camera, point, p); //计算RP
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        T xp = p[0] / p[2];
        T yp = p[1] / p[2]; //xp,yp是归一化坐标，深度为p[2]
        T u_ = xp * _K(0, 0) + _K(0, 2);
        T v_ = yp * _K(1, 1) + _K(1, 2);
        residual[0] = T(_uv.x()) - u_;
        residual[1] = T(_uv.y()) - v_;
        return true;
    }
    static ceres::CostFunction *Create(const Eigen::Vector2d uv, const Eigen::Vector3d xyz,const Eigen::Matrix3d K)
    {
        return (new ceres::AutoDiffCostFunction<PnPCeres, 2, 6>(
            new PnPCeres(uv, xyz,K)));
    }
    const Eigen::Vector2d _uv;
    const Eigen::Vector3d _xyz;
    const Eigen::Matrix3d _K;
};

class RtOptimizer
{
private:
    ceres::Problem problem;
    double camera[6];
public:
    RtOptimizer(/* args */);
    ~RtOptimizer();

    void SetInitialVal(const Eigen::Matrix3d &Rcw,const Eigen::Vector3d &tcw);
    void SetInitialVal(const Eigen::Quaterniond &qcw,const Eigen::Vector3d &tcw);
    void SetInitialVal(const Eigen::Matrix4d &Tcw);

    void AddResidualItem(const Eigen::Vector2d &uv, const Eigen::Vector3d &xyz,const Eigen::Matrix3d &K);
    void Solve();
    Eigen::Matrix4d GetTcw();
};

