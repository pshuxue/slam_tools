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
class InverseDistanceInSingleRigFactor : public ceres::SizedCostFunction<3, 7> // imupose imupose uv p
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  InverseDistanceInSingleRigFactor(const Eigen::Vector3d &pc, const Eigen::Vector3d &pa); //  Tij
  void SetinformationMatrix(const Eigen::Matrix<double, 3, 3> &info);
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  Eigen::Vector3d pc_, pa_;
  Eigen::Matrix<double, 3, 3> sqrt_info_;
};

InverseDistanceInSingleRigFactor::InverseDistanceInSingleRigFactor(const Eigen::Vector3d &pc, const Eigen::Vector3d &pa)
{
  pc_ = pc;
  pa_ = pa;
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
  const Eigen::Vector3d tca(parameters[0][0], parameters[0][1], parameters[0][2]);
  const Eigen::Quaterniond qca(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  const Eigen::Matrix3d Rca = qca.toRotationMatrix();

  const Eigen::Vector3d pc = Rca * pa_ + tca;

  Eigen::Map<Eigen::Matrix<double, 3, 1>> residual(residuals);
  residual = pc - pc_;
  if (jacobians)
  {
    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J(jacobians[0]);
      J.setZero();
      J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();  // dpc_dtwa
      J.block<3, 3>(0, 3) = -Sophus::SO3::hat(Rca * pa_); // dpc_dRwa
      J = sqrt_info_ * J;
    }
  }
  return true;
}

class PoseLocalParameterization : public ceres::LocalParameterization
{
  virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 7; };
  virtual int LocalSize() const { return 6; };
};

Eigen::Quaterniond deltaQ(const Eigen::Vector3d &theta);

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
  Eigen::Map<const Eigen::Vector3d> _p(x);
  Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

  Eigen::Map<const Eigen::Vector3d> dp(delta);

  Eigen::Quaterniond dq = deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

  Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
  Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

  p = _p + dp;
  q = (dq * _q).normalized();

  return true;
}
bool PPlus(const double *x, const double *delta, double *x_plus_delta)
{
  Eigen::Map<const Eigen::Vector3d> _p(x);
  Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

  Eigen::Map<const Eigen::Vector3d> dp(delta);

  Eigen::Quaterniond dq = deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));
  Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
  Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

  p = _p + dp;
  // std::cout<<"q "<<_q.coeffs()<<std::endl;
  q = (dq * _q).normalized();

  return true;
}

bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
  j.topRows<6>().setIdentity();
  j.bottomRows<1>().setZero();

  return true;
}

Eigen::Quaterniond deltaQ(const Eigen::Vector3d &theta)
{
  Eigen::Quaterniond dq;
  Eigen::Vector3d half_theta = theta;
  half_theta /= 2.0;
  dq.w() = 1.0;
  dq.x() = half_theta.x();
  dq.y() = half_theta.y();
  dq.z() = half_theta.z();
  return dq;
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


  double Tca[7] = {0.1, 0.1, 0.1, 0.8, 0.6, 0, 0};

  {
    double parameters[7] = {0.1, 0.1, 0.1, 0.8, 0.6, 0, 0};
    Eigen::Matrix<double, 3, 1> residual_init;
    {
      const Eigen::Vector3d tca(parameters[0], parameters[1], parameters[2]);
      const Eigen::Quaterniond qca(parameters[6], parameters[3], parameters[4], parameters[5]);
      const Eigen::Matrix3d Rca = qca.toRotationMatrix();
      const Eigen::Vector3d pc = Rca * Eigen::Vector3d(1, 1, 1) + tca;
      residual_init = pc - Eigen::Vector3d(1, 1, 1);
      // std::cout<<"init "<<residual_init<<std::endl;
    }

    double x = 0;
    double y = 0;
    double z = 1e-5;
    double dT[6] = {0, 0, 0, x, y, z};
    PPlus(Tca, dT, parameters);
    for(int i = 0; i < 7; i++)
    std::cout<<parameters[i]<< std::endl;
    Eigen::Matrix<double, 3, 1> residual_i;
    {
      const Eigen::Vector3d tca(parameters[0], parameters[1], parameters[2]);
      const Eigen::Quaterniond qca(parameters[6], parameters[3], parameters[4], parameters[5]);
      const Eigen::Matrix3d Rca = qca.toRotationMatrix();
      const Eigen::Vector3d pc = Rca * Eigen::Vector3d(1, 1, 1) + tca;
      residual_i = pc - Eigen::Vector3d(1, 1, 1);
      // std::cout<<"init "<<residual_i<<std::endl;
    }

    Eigen::Matrix<double, 3, 1> residual = (residual_i- residual_init) / z;
    std::cout << "J " << residual << std::endl;
  }

  ceres::LocalParameterization *parameterization =
      new PoseLocalParameterization;
  ceres::Problem problem;
  problem.AddParameterBlock(Tca, 7, parameterization);

  Eigen::Vector3d meas(-0.707, 0.707, 0);
  InverseDistanceInSingleRigFactor *inverse_distance_factor = new InverseDistanceInSingleRigFactor(Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(1, 1, 1));
  Eigen::Matrix<double, 3, 3> sqrt_info = Eigen::Matrix<double, 3, 3>::Identity();
  inverse_distance_factor->SetinformationMatrix(sqrt_info);

  problem.AddResidualBlock(
      inverse_distance_factor,
      nullptr,
      Tca);

  std::vector<const ceres::LocalParameterization *> local_parameterizations;
  local_parameterizations.push_back(parameterization);

  ceres::NumericDiffOptions numeric_diff_options;

  std::vector<double *> parameter_blocks;
  parameter_blocks.push_back(Tca);

  ceres::GradientChecker::ProbeResults results;
  ceres::GradientChecker checker(inverse_distance_factor, &local_parameterizations, numeric_diff_options);
  checker.Probe(parameter_blocks.data(), 1e-5, &results);

  if (!checker.Probe(parameter_blocks.data(), 1e-5, &results))
  {
    std::cout << "An error has occurred:\n"
              << results.error_log << std::endl;
  }

  ceres::Solver::Options ops;
  ceres::Solver::Summary summary;
  ops.minimizer_progress_to_stdout = true;
  ceres::Solve(ops, &problem, &summary);
}
