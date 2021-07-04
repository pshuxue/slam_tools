#include "points_optimizer.h"

namespace lvm
{
  PointsOptimizer::PointsOptimizer(/* args */)
  {
  }

  PointsOptimizer::~PointsOptimizer()
  {
  }

  void PointsOptimizer::SetInitialVal(Eigen::Vector3d pw)
  {
    point[0] = pw.x();
    point[1] = pw.y();
    point[2] = pw.z();
  }

  void PointsOptimizer::AddResidualItem(Eigen::Vector2d uv, Eigen::Matrix4d Tcw, Eigen::Matrix3d K)
  {
    ceres::CostFunction *cost_function = OptPointsCeres::Create(uv, Tcw, K);
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);

    problem.AddResidualBlock(cost_function, loss_function, point);
  }

  void PointsOptimizer::Solve()
  {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
  }

  Eigen::Vector3d PointsOptimizer::GetPoint()
  {
    Eigen::Vector3d pw(point[0], point[1], point[2]);
    return pw;
  }
}