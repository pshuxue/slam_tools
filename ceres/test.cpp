//逆深度的优化，在inverse_depth.cpp 上改的，这里主要优化的是imu，主要解决多目只优化一个位姿的问题

#include <iostream>
#include <Eigen/Core>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/gradient_checker.h>
#include <ceres/gradient_problem_solver.h>
#include "so3.h"

using namespace std;

Eigen::Matrix4d TransFromArr(double *arr)
{
  Eigen::Vector3d t(arr[0], arr[1], arr[2]);
  Eigen::Quaterniond q(arr[6], arr[3], arr[4], arr[5]);
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.topLeftCorner(3, 3) = q.toRotationMatrix();
  T.topRightCorner(3, 1) = t;
  return T;
}

void PrintT(Eigen::Matrix4d &T)
{
  Eigen::Matrix3d R = T.topLeftCorner(3, 3);
  Eigen::Quaterniond q(R);
  std::cout << T(0, 3) << " " << T(1, 3) << " " << T(2, 3) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
}

void TransFromArr(double *arr, Eigen::Quaterniond &q, Eigen::Vector3d &t)
{
  t = Eigen::Vector3d(arr[0], arr[1], arr[2]);
  q = Eigen::Quaterniond(arr[6], arr[3], arr[4], arr[5]);
}

class InverseDistanceInSingleRigFactor : public ceres::SizedCostFunction<3, 2, 1> // imupose imupose uv p
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  InverseDistanceInSingleRigFactor(const Eigen::Vector3d &meas_direction, const Eigen::Matrix4d &Tba, const Eigen::Matrix4d &Tbc); //  Tij
  void SetinformationMatrix(const Eigen::Matrix<double, 3, 3> &info);
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;
  Eigen::Matrix3d Rca; //外参
  Eigen::Vector3d tca;
  Eigen::Vector3d meas_direction_;
  Eigen::Matrix<double, 3, 3> sqrt_info_;
};

InverseDistanceInSingleRigFactor::InverseDistanceInSingleRigFactor(const Eigen::Vector3d &meas_direction, const Eigen::Matrix4d &Tba, const Eigen::Matrix4d &Tbc)
{
  Rca = Tbc.topLeftCorner(3, 3).transpose() * Tba.topLeftCorner(3, 3);
  tca = Tbc.topLeftCorner(3, 3).transpose() * (Tba.topRightCorner(3, 1) - Tbc.topRightCorner(3, 1));
  meas_direction_ = meas_direction;
  sqrt_info_ = Eigen::Matrix<double, 3, 3>::Identity();
  // std::cout << "Rca" << Rca << std::endl;
  // std::cout << "tca" << (Tba.topRightCorner(3, 1) - Tbc.topRightCorner(3, 1)) << std::endl;
}

void InverseDistanceInSingleRigFactor::SetinformationMatrix(const Eigen::Matrix<double, 3, 3> &info)
{
  sqrt_info_ = info;
}

bool InverseDistanceInSingleRigFactor::Evaluate(double const *const *parameters, double *residuals,
                                                double **jacobians) const
{
  const double u = parameters[0][0];
  const double u2 = u * u;
  const double v = parameters[0][1];
  const double v2 = v * v;
  const double inverse = parameters[1][0];

  const double n = 2 / (u * u + v * v + 1);
  const Eigen::Vector3d pa(n * u, n * v, n - 1);
  // std::cout << "sspa " << pa.transpose() << std::endl;
  const Eigen::Vector3d pc = Rca * pa + inverse * tca;
  const Eigen::Vector3d pc_norm = pc.normalized();
  // std::cout << "pc_norm " << pc.transpose() << " " << inverse << std::endl;

  Eigen::Map<Eigen::Matrix<double, 3, 1>> residual(residuals);
  residual = pc_norm - meas_direction_;
  // std::cout << "residual " << residual.transpose() << std::endl;
  const double norm = pc.norm();
  const double norm_3 = norm * norm * norm;

  const Eigen::Matrix3d dr_dpc = -1 / norm_3 * pc * pc.transpose() + 1 / norm * Eigen::Matrix3d::Identity();

  if (jacobians)
  {
    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> dr_duv(jacobians[0]);
      Eigen::Matrix<double, 3, 2> dpc_duv;
      dpc_duv.setZero();
      const Eigen::Matrix3d dpc_dpa = Rca;
      Eigen::Matrix<double, 3, 2> dpa_duv;
      const double N = u * u + v * v + 1;
      const double NN = N * N;
      dpa_duv(0, 0) = (2 * v2 - 2 * u2 + 2) / NN;
      dpa_duv(0, 1) = -4 * u * v / NN;
      dpa_duv(1, 0) = dpa_duv(0, 1);
      dpa_duv(1, 1) = (2 * u2 - 2 * v2 + 2) / NN;
      dpa_duv(2, 0) = -4 * u / NN;
      dpa_duv(2, 1) = -4 * v / NN;
      dpc_duv = dpc_dpa * dpa_duv;
      dr_duv = dr_dpc * dpc_duv;
      dr_duv = sqrt_info_ * dr_duv;
    }
    if (jacobians[1])
    {
      Eigen::Map<Eigen::Vector3d> dr_ds(jacobians[1]);
      Eigen::Vector3d dpc_ds;
      dpc_ds = tca;
      dr_ds = dr_dpc * dpc_ds;
      // std::cout << dr_ds << std::endl;
      dr_ds = sqrt_info_ * dr_ds;
    }
  }
  return true;
}

void Test3();

int main()
{
  Test3();
  return 0;
}

void Test3()
{
  std::cout << "optimizer uvp" << std::endl;

  Eigen::Matrix4d Tbc = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d Tba = Eigen::Matrix4d::Identity();
  Tbc.topRightCorner(3, 1) = Eigen::Vector3d(1, 1, 0);

  double uv_arr[2] = {0.3, 0.7};
  double inverse_dis_arr[1] = {1.0 / 2};

  ceres::Problem problem;
  problem.AddParameterBlock(uv_arr, 2);
  problem.AddParameterBlock(inverse_dis_arr, 1);
  problem.SetParameterBlockConstant(inverse_dis_arr);

  Eigen::Vector3d meas(-0.707, 0.707, 0);
  InverseDistanceInSingleRigFactor *inverse_distance_factor = new InverseDistanceInSingleRigFactor(meas, Tba, Tbc);
  Eigen::Matrix<double, 3, 3> sqrt_info = Eigen::Matrix<double, 3, 3>::Identity();
  inverse_distance_factor->SetinformationMatrix(sqrt_info);

  problem.AddResidualBlock(
      inverse_distance_factor,
      nullptr,
      uv_arr, inverse_dis_arr);

  // std::vector<const ceres::LocalParameterization *> local_parameterizations;
  // local_parameterizations.push_back(nullptr);
  // local_parameterizations.push_back(nullptr);

  // ceres::NumericDiffOptions numeric_diff_options;

  // std::vector<double *> parameter_blocks;
  // parameter_blocks.push_back(uv_arr);
  // parameter_blocks.push_back(inverse_dis_arr);

  // ceres::GradientChecker::ProbeResults results;
  // ceres::GradientChecker checker(inverse_distance_factor, &local_parameterizations, numeric_diff_options);
  // checker.Probe(parameter_blocks.data(), 1e-5, &results);

  // if (!checker.Probe(parameter_blocks.data(), 1e-5, &results))
  // {
  //   std::cout << "An error has occurred:\n"
  //             << results.error_log << std::endl;
  // }

  ceres::Solver::Options ops;
  ceres::Solver::Summary summary;
  ops.minimizer_progress_to_stdout = true;
  ceres::Solve(ops, &problem, &summary);

  std::cout << "uvp " << uv_arr[0] << " " << uv_arr[1] << " " << inverse_dis_arr[0] << std::endl;
}
