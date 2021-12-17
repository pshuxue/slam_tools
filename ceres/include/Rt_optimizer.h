#include <iostream>
#include <eigen3/Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/eigen.hpp>
#include <chrono>

struct PnPAutoDiffCeres
{
    PnPAutoDiffCeres(Eigen::Vector2d uv, Eigen::Vector3d xyz, Eigen::Matrix3d K) : _uv(uv), _xyz(xyz), _K(K) {}
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
    static ceres::CostFunction *Create(const Eigen::Vector2d uv, const Eigen::Vector3d xyz, const Eigen::Matrix3d K)
    {
        return (new ceres::AutoDiffCostFunction<PnPAutoDiffCeres, 2, 6>(
            new PnPAutoDiffCeres(uv, xyz, K)));
    }
    const Eigen::Vector2d _uv;
    const Eigen::Vector3d _xyz;
    const Eigen::Matrix3d _K;
};

class PnPNumericDiffCeres : public ceres::SizedCostFunction<2, 6>
{
public:
    PnPNumericDiffCeres(Eigen::Vector2d uv, Eigen::Vector3d xyz, Eigen::Matrix3d K) : _uv(uv), _xyz(xyz), _K(K) {}
    virtual ~PnPNumericDiffCeres() {}
    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const
    {
        const Eigen::Vector3d vec(parameters[0][0], parameters[0][1], parameters[0][2]);
        const Eigen::Vector3d t(parameters[0][3], parameters[0][4], parameters[0][5]);
        const Eigen::AngleAxisd axis_v(vec.norm(), vec.normalized());
        const Eigen::Vector3d pc = axis_v.toRotationMatrix() * _xyz + t;

        const Eigen::Vector3d pc_n(pc.x() / pc.z(), pc.y() / pc.z(), 1);
        const Eigen::Vector3d uv_1 = _K * pc_n;
        residuals[0] = uv_1.x() - _uv.x();
        residuals[1] = uv_1.y() - _uv.y();

        Eigen::Matrix<double, 2, 3> duv_dpc;
        duv_dpc << _K(0, 0) / pc.z(), 0, -_K(0, 0) * pc.x() / pc.z() / pc.z(),
            0, _K(1, 1) / pc.z(), -_K(1, 1) * pc.y() / pc.z() / pc.z();
        duv_dpc = -duv_dpc;

        Eigen::Matrix3d dpc_dpw;
        dpc_dpw << 0, -pc.z(), pc.y(),
            pc.z(), 0, -pc.x(),
            -pc.y(), pc.x(), 0;

        Eigen::Matrix<double, 2, 3> duv_dpw = duv_dpc * dpc_dpw;

        if (jacobians != NULL && jacobians[0] != NULL)
        {
            jacobians[0][0] = duv_dpw(0, 0);
            jacobians[0][1] = duv_dpw(0, 1);
            jacobians[0][2] = duv_dpw(0, 2);
            jacobians[0][3] = duv_dpw(1, 0);
            jacobians[0][4] = duv_dpw(1, 1);
            jacobians[0][5] = duv_dpw(1, 2);
        }
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector2d uv, const Eigen::Vector3d xyz, const Eigen::Matrix3d K)
    {
        ceres::CostFunction *cost_function = new ceres::NumericDiffCostFunction<PnPNumericDiffCeres, ceres::FORWARD, 2, 6>(new PnPNumericDiffCeres(uv, xyz, K));
        return cost_function;
    }

private:
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

    void SetInitialVal(const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw);
    void SetInitialVal(const Eigen::Quaterniond &qcw, const Eigen::Vector3d &tcw);
    void SetInitialVal(const Eigen::Matrix4d &Tcw);

    void AddResidualItemAutoDiff(const Eigen::Vector2d &uv, const Eigen::Vector3d &xyz, const Eigen::Matrix3d &K);
    void AddResidualItemNumericDiff(const Eigen::Vector2d &uv, const Eigen::Vector3d &xyz, const Eigen::Matrix3d &K);
    void Solve();
    Eigen::Matrix4d GetTcw();
};
