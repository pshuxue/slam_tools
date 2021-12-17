#include "Rt_optimizer.h"

RtOptimizer::RtOptimizer(/* args */)
{
}

RtOptimizer::~RtOptimizer()
{
}

void RtOptimizer::SetInitialVal(const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw)
{
  Eigen::AngleAxisd vec(Rcw);
  camera[0] = vec.angle() * vec.axis().x();
  camera[1] = vec.angle() * vec.axis().y();
  camera[2] = vec.angle() * vec.axis().z();
  camera[3] = tcw.x();
  camera[4] = tcw.y();
  camera[5] = tcw.z();
}

void RtOptimizer::SetInitialVal(const Eigen::Quaterniond &qcw, const Eigen::Vector3d &tcw)
{
  Eigen::AngleAxisd vec(qcw);
  camera[0] = vec.angle() * vec.axis().x();
  camera[1] = vec.angle() * vec.axis().y();
  camera[2] = vec.angle() * vec.axis().z();
  camera[3] = tcw.x();
  camera[4] = tcw.y();
  camera[5] = tcw.z();
}

void RtOptimizer::SetInitialVal(const Eigen::Matrix4d &Tcw)
{
  Eigen::AngleAxisd vec(Eigen::Matrix3d(Tcw.topLeftCorner(3, 3)));
  camera[0] = vec.angle() * vec.axis().x();
  camera[1] = vec.angle() * vec.axis().y();
  camera[2] = vec.angle() * vec.axis().z();
  camera[3] = Tcw(0, 3);
  camera[4] = Tcw(1, 3);
  camera[5] = Tcw(2, 3);
}

void RtOptimizer::AddResidualItemAutoDiff(const Eigen::Vector2d &uv, const Eigen::Vector3d &xyz, const Eigen::Matrix3d &K)
{
  ceres::CostFunction *cost_function = PnPAutoDiffCeres::Create(uv, xyz, K);
  ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
  problem.AddResidualBlock(cost_function, loss_function, camera);
}

void RtOptimizer::AddResidualItemNumericDiff(const Eigen::Vector2d &uv, const Eigen::Vector3d &xyz, const Eigen::Matrix3d &K)
{
  ceres::CostFunction *cost_function = PnPNumericDiffCeres::Create(uv, xyz, K);
  ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);

  problem.AddResidualBlock(cost_function, loss_function, camera);
}

void RtOptimizer::Solve()
{
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
}

Eigen::Matrix4d RtOptimizer::GetTcw()
{
  Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();

  Eigen::Vector3d v(camera[0], camera[1], camera[2]);
  Eigen::AngleAxisd vec(v.norm(), v.normalized());
  Tcw.topLeftCorner(3, 3) = vec.toRotationMatrix();
  Tcw(0, 3) = camera[3];
  Tcw(1, 3) = camera[4];
  Tcw(2, 3) = camera[5];
  return Tcw;
}