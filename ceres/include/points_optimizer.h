//使用opencv进行pnp估计，然后再使用ceres优化
#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <chrono>

namespace lvm
{
  struct OptPointsCeres
  {
    OptPointsCeres(Eigen::Vector2d uv, Eigen::Matrix4d Tcw, Eigen::Matrix3d K) : _uv(uv), _Tcw(Tcw), _K(K) {}
    // 残差的计算
    template <typename T>
    bool operator()(
        const T *const point, // 3D点参数，有3维
        T *residual) const    // 2维残差
    {
      Eigen::Matrix<double, 3, 4> Tcw_34 = _Tcw.topLeftCorner(3, 4);
      Eigen::Matrix<double, 3, 4> K_Tcw = _K * Tcw_34;

      T x = T(K_Tcw(0, 0)) * point[0] + T(K_Tcw(0, 1)) * point[1] + T(K_Tcw(0, 2)) * point[2] + T(K_Tcw(0, 3));
      T y = T(K_Tcw(1, 0)) * point[0] + T(K_Tcw(1, 1)) * point[1] + T(K_Tcw(1, 2)) * point[2] + T(K_Tcw(1, 3));
      T z = T(K_Tcw(2, 0)) * point[0] + T(K_Tcw(2, 1)) * point[1] + T(K_Tcw(2, 2)) * point[2] + T(K_Tcw(2, 3));

      residual[0] = T(x / z) - T(_uv.x());
      residual[1] = T(y / z) - T(_uv.y());
      return true;
    }
    static ceres::CostFunction *Create(const Eigen::Vector2d uv, const Eigen::Matrix4d Tcw, const Eigen::Matrix3d K)
    {
      return (new ceres::AutoDiffCostFunction<OptPointsCeres, 2, 3>(
          new OptPointsCeres(uv, Tcw, K)));
    }
    const Eigen::Vector2d _uv;
    const Eigen::Matrix4d _Tcw;
    const Eigen::Matrix3d _K;
  };

  class PointsOptimizer
  {
  private:
    double point[3];
    ceres::Problem problem;

  public:
    PointsOptimizer(/* args */);
    ~PointsOptimizer();

    void SetInitialVal(Eigen::Vector3d pw);
    void AddResidualItemAutoDiff(Eigen::Vector2d uv, Eigen::Matrix4d Tcw, Eigen::Matrix3d K);
    void Solve();
    Eigen::Vector3d GetPoint();
  };

}