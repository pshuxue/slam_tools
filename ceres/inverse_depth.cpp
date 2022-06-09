//逆深度的优化，同时优化两个图像的位姿 立体投影的位置以及深度
#include <iostream>
#include <Eigen/Core>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "so3.h"

using namespace std;

class InverseDistanceFactor : public ceres::SizedCostFunction<3, 7, 7, 2, 1>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  InverseDistanceFactor(const Eigen::Vector3d &meas_direction); //  Tij
  void SetinformationMatrix(const Eigen::Matrix<double, 3, 3> &info);
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  Eigen::Vector3d meas_direction_;
  Eigen::Matrix<double, 3, 3> sqrt_info_;
};

InverseDistanceFactor::InverseDistanceFactor(const Eigen::Vector3d &meas_direction)
{
  meas_direction_ = meas_direction;
}

void InverseDistanceFactor::SetinformationMatrix(const Eigen::Matrix<double, 3, 3> &info)
{
  sqrt_info_ = info;
}

bool InverseDistanceFactor::Evaluate(double const *const *parameters, double *residuals,
                                     double **jacobians) const
{
  const Eigen::Vector3d twa(parameters[0][0], parameters[0][1], parameters[0][2]);
  const Eigen::Quaterniond qwa(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  const Eigen::Matrix3d Rwa = qwa.toRotationMatrix();
  const Eigen::Matrix3d Raw = Rwa.transpose();
  // std::cout << "Rwa " << Rwa << std::endl;

  const Eigen::Vector3d twc(parameters[1][0], parameters[1][1], parameters[1][2]);
  const Eigen::Quaterniond qwc(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
  const Eigen::Matrix3d Rwc = qwc.toRotationMatrix();
  const Eigen::Matrix3d Rcw = Rwc.transpose();
  // std::cout << "Rwc " << Rwc << std::endl;

  const double u = parameters[2][0];
  const double u2 = u * u;
  const double v = parameters[2][1];
  const double v2 = v * v;
  const double inverse = parameters[3][0];

  const double n = 2 / (u * u + v * v + 1);
  const Eigen::Vector3d pa(n * u, n * v, n - 1);
  // std::cout << "pa " << pa.transpose() << std::endl;
  const Eigen::Vector3d pc = Rcw * Rwa * pa + inverse * Rcw * (twa - twc);
  const Eigen::Vector3d pc_norm = pc.normalized();
  // std::cout << "pc " << pc.transpose() << std::endl;
  // std::cout << "pc_norm " << pc_norm.transpose() << " " << inverse << std::endl;

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
      Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> dr_dTwa(jacobians[0]);
      Eigen::Matrix<double, 3, 7> dpc_dTwa;
      dpc_dTwa.setZero();

      dpc_dTwa.block<3, 3>(0, 0) = inverse * Rcw;                     // dpc_dtwa
      dpc_dTwa.block<3, 3>(0, 3) = -Rcw * Sophus::SO3::hat(Rwa * pa); // dpc_dRwa
      dr_dTwa = dr_dpc * dpc_dTwa;
      dr_dTwa = sqrt_info_ * dr_dTwa;
    }

    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> dr_dTwc(jacobians[1]);
      Eigen::Matrix<double, 3, 7> dpc_dTwc;
      dpc_dTwc.setZero();

      dpc_dTwc.block<3, 3>(0, 0) = -inverse * Rcw;                                           // dpc_dtwc
      dpc_dTwc.block<3, 3>(0, 3) = Rcw * Sophus::SO3::hat(Rwa * pa + inverse * (twa - twc)); // dpc_dRwc
      dr_dTwc = dr_dpc * dpc_dTwc;
      dr_dTwc = sqrt_info_ * dr_dTwc;
    }

    if (jacobians[2])
    {
      Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> dr_duv(jacobians[2]);
      Eigen::Matrix<double, 3, 2> dpc_duv;
      dpc_duv.setZero();
      const Eigen::Matrix3d dpc_dpa = Rcw * Rwa;
      Eigen::Matrix<double, 3, 2> dpa_duv;
      const double N = u * u + v * v + 1;
      const double NN = N * N;
      dpa_duv(0, 0) = (2 * v2 - 2 * u2 + 2) / NN;
      dpa_duv(0, 1) = -4 * u * v / NN;
      dpa_duv(1, 0) = dpa_duv(0, 1);
      dpa_duv(1, 1) = (2 * u2 - 2 * v2 + 1) / NN;
      dpa_duv(2, 0) = -4 * u / NN;
      dpa_duv(2, 1) = -4 * v / NN;
      dpc_duv = dpc_dpa * dpa_duv;
      dr_duv = dr_dpc * dpc_duv;
      dr_duv = sqrt_info_ * dr_duv;
    }
    if (jacobians[3])
    {
      Eigen::Map<Eigen::Vector3d> dr_ds(jacobians[3]);
      Eigen::Vector3d dpc_ds;
      dpc_ds = Rcw * (twa - twc);
      dr_ds = dr_dpc * dpc_ds;
      // std::cout << dr_ds << std::endl;
      dr_ds = sqrt_info_ * dr_ds;
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
  q = (_q * dq).normalized();

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

void Test1();
void Test2();
void Test3();

int main()
{
  Test1();
  Test2();
  Test3();
  return 0;
}

void Test1()
{
  std::cout << "optimizer Twa" << std::endl;

  double Twa_arr[7] = {0.1, 0.2, 0.1, 0, 0, 0.8, 0.6}; // txtytzqxqyqzqw
  Eigen::Matrix3d Rwa = Eigen::Matrix3d::Identity();
  Eigen::Vector3d twa(0.7, 0, 0);

  constexpr int number_point = 20;
  constexpr int number_image = 100;

  double uv_arr[number_point][2];
  double inverse_dis_arr[number_point][1];
  double Twc_arr[number_image][7];
  std::vector<Eigen::Vector3d> map_points;
  std::vector<Eigen::Matrix3d> Rwcs;
  std::vector<Eigen::Vector3d> twcs;

  ceres::Problem problem;
  ceres::LocalParameterization *parameterization =
      new PoseLocalParameterization;
  problem.AddParameterBlock(Twa_arr, 7, parameterization);

  for (int i = 0; i < number_point; ++i)
  {
    Eigen::Vector3d map_point = Eigen::Vector3d::Random() * 5;
    map_point.z() += 8;
    map_point.x() += 5;
    map_point.y() += 5;
    map_points.push_back(map_point);

    Eigen::Vector3d pa = Rwa.transpose() * (map_point - twa);
    inverse_dis_arr[i][0] = 1 / pa.norm();
    pa = pa.normalized();
    uv_arr[i][0] = pa.x() / (pa.z() + 1);
    uv_arr[i][1] = pa.y() / (pa.z() + 1);
  }

  for (int i = 0; i < number_point; ++i)
  {
    problem.AddParameterBlock(uv_arr[i], 2);
    problem.SetParameterBlockConstant(uv_arr[i]);
    problem.AddParameterBlock(inverse_dis_arr[i], 1);
    problem.SetParameterBlockConstant(inverse_dis_arr[i]);
  }

  for (int i = 0; i < number_image; ++i)
  {
    Eigen::AngleAxisd vec(0.1, Eigen::Vector3d::Random());
    Eigen::Matrix3d Rwc = vec.toRotationMatrix();
    Eigen::Vector3d twc = Eigen::Vector3d::Random() * 3;
    Rwcs.emplace_back(Rwc);
    twcs.push_back(twc);
    Eigen::Quaterniond qwc(Rwc);
    Twc_arr[i][0] = twc.x();
    Twc_arr[i][1] = twc.y();
    Twc_arr[i][2] = twc.z();
    Twc_arr[i][3] = qwc.x();
    Twc_arr[i][4] = qwc.y();
    Twc_arr[i][5] = qwc.z();
    Twc_arr[i][6] = qwc.w();
  }
  for (int i = 0; i < number_image; ++i)
  {
    problem.AddParameterBlock(Twc_arr[i], 7, parameterization);
    problem.SetParameterBlockConstant(Twc_arr[i]);
  }

  for (int i = 0; i < number_point; ++i)
  {
    for (int j = 0; j < number_image; ++j)
    {
      Eigen::Matrix3d Rwc = Rwcs[j];
      Eigen::Vector3d twc = twcs[j];
      Eigen::Vector3d pc = Rwc.transpose() * (map_points[i] - twc);

      Eigen::Vector3d meas = pc.normalized();
      InverseDistanceFactor *inverse_distance_factor = new InverseDistanceFactor(meas);
      Eigen::Matrix<double, 3, 3> sqrt_info = Eigen::Matrix<double, 3, 3>::Identity();
      inverse_distance_factor->SetinformationMatrix(sqrt_info);

      problem.AddResidualBlock(
          inverse_distance_factor,
          nullptr,
          Twa_arr, Twc_arr[j], uv_arr[i], inverse_dis_arr[i]);
    }
  }

  ceres::Solver::Options ops;
  ceres::Solver::Summary summary;
  ops.minimizer_progress_to_stdout = true;
  ceres::Solve(ops, &problem, &summary);

  std::cout << "Twa ";
  for (int i = 0; i < 7; ++i)
  {
    std::cout << Twa_arr[i] << " ";
  }
  std::cout << endl;
}

void Test2()
{
  std::cout << "optimizer Twc" << std::endl;

  double Twc_arr[7] = {0.1, 0.2, -0.4, 0.8, 0, 0, 0.6}; // txtytzqxqyqzqw
  Eigen::Matrix3d Rwc = Eigen::Matrix3d::Identity();
  Eigen::Vector3d twc(0.1, 2, -0.1);

  constexpr int number_point_image = 200;

  double uv_arr[number_point_image][2];
  double inverse_dis_arr[number_point_image][1];
  double Twa_arr[number_point_image][7];
  std::vector<Eigen::Vector3d> map_points;
  std::vector<Eigen::Matrix3d> Rwas;
  std::vector<Eigen::Vector3d> twas;

  ceres::Problem problem;
  ceres::LocalParameterization *parameterization =
      new PoseLocalParameterization;
  problem.AddParameterBlock(Twc_arr, 7, parameterization);

  for (int i = 0; i < number_point_image; ++i)
  {
    Eigen::Vector3d map_point = Eigen::Vector3d::Random() * 5;
    map_point.z() += 8;
    map_point.x() += 5;
    map_point.y() += 5;
    map_points.push_back(map_point);

    Eigen::AngleAxisd vec(0.15, Eigen::Vector3d::Random());
    Eigen::Matrix3d Rwa = vec.toRotationMatrix();
    Eigen::Vector3d twa = Eigen::Vector3d::Random() * 3;
    Rwas.push_back(Rwa);
    twas.push_back(twa);

    Eigen::Vector3d pa = Rwa.transpose() * (map_point - twa);
    inverse_dis_arr[i][0] = 1 / pa.norm();
    pa = pa.normalized();
    uv_arr[i][0] = pa.x() / (pa.z() + 1);
    uv_arr[i][1] = pa.y() / (pa.z() + 1);

    Eigen::Quaterniond qwa(Rwa);
    Twa_arr[i][0] = twa.x();
    Twa_arr[i][1] = twa.y();
    Twa_arr[i][2] = twa.z();
    Twa_arr[i][3] = qwa.x();
    Twa_arr[i][4] = qwa.y();
    Twa_arr[i][5] = qwa.z();
    Twa_arr[i][6] = qwa.w();

    problem.AddParameterBlock(uv_arr[i], 2);
    problem.SetParameterBlockConstant(uv_arr[i]);
    problem.AddParameterBlock(inverse_dis_arr[i], 1);
    problem.SetParameterBlockConstant(inverse_dis_arr[i]);
    problem.AddParameterBlock(Twa_arr[i], 7, parameterization);
    problem.SetParameterBlockConstant(Twa_arr[i]);

    Eigen::Vector3d pc = Rwc.transpose() * (map_point - twc);

    InverseDistanceFactor *inverse_distance_factor = new InverseDistanceFactor(pc.normalized());
    Eigen::Matrix<double, 3, 3> sqrt_info = Eigen::Matrix<double, 3, 3>::Identity();
    inverse_distance_factor->SetinformationMatrix(sqrt_info);

    problem.AddResidualBlock(
        inverse_distance_factor,
        nullptr,
        Twa_arr[i], Twc_arr, uv_arr[i], inverse_dis_arr[i]);
    // std::cout << i << std::endl;
  }

  ceres::Solver::Options ops;
  ceres::Solver::Summary summary;
  ops.minimizer_progress_to_stdout = true;
  ceres::Solve(ops, &problem, &summary);

  std::cout << "Twc ";
  for (int i = 0; i < 7; ++i)
  {
    std::cout << Twc_arr[i] << " ";
  }
  std::cout << endl;
}

void Test3()
{
  std::cout << "optimizer uvp" << std::endl;

  double Twa_arr[7] = {0.7, 0, 0, 0, 0, 0, 1}; // txtytzqxqyqzqw
  Eigen::Matrix3d Rwa = Eigen::Matrix3d::Identity();
  Eigen::Vector3d twa(0.7, 0, 0);

  constexpr int number_image = 100;

  double uv_arr[2] = {0.5, -0.4};
  double inverse_dis_arr[1] = {0.8};

  double Twc_arr[number_image][7];

  Eigen::Vector3d map_point = Eigen::Vector3d::Random() * 5;
  map_point.z() += 8;
  map_point.x() += 5;
  map_point.y() += 5;

  Eigen::Vector3d pa = Rwa.transpose() * (map_point - twa);
  inverse_dis_arr[0] = 1 / pa.norm();
  pa = pa.normalized();
  uv_arr[0] = pa.x() / (pa.z() + 1);
  uv_arr[1] = pa.y() / (pa.z() + 1);
  std::cout << "uvp init " << uv_arr[0] << " " << uv_arr[1] << " " << inverse_dis_arr[0] << std::endl;
  inverse_dis_arr[0] += 0.3;
  uv_arr[0] += 0.12;
  uv_arr[1] -= 0.12;
  std::cout << "uvp err  " << uv_arr[0] << " " << uv_arr[1] << " " << inverse_dis_arr[0] << std::endl;

  ceres::Problem problem;
  ceres::LocalParameterization *parameterization =
      new PoseLocalParameterization;
  problem.AddParameterBlock(Twa_arr, 7, parameterization);
  problem.SetParameterBlockConstant(Twa_arr);
  problem.AddParameterBlock(uv_arr, 2);
  problem.AddParameterBlock(inverse_dis_arr, 1);

  for (int i = 0; i < number_image; ++i)
  {
    Eigen::AngleAxisd vec(0.1, Eigen::Vector3d::Random());
    Eigen::Matrix3d Rwc = vec.toRotationMatrix();
    Eigen::Vector3d twc = Eigen::Vector3d::Random() * 3;
    Eigen::Quaterniond qwc(Rwc);
    Twc_arr[i][0] = twc.x();
    Twc_arr[i][1] = twc.y();
    Twc_arr[i][2] = twc.z();
    Twc_arr[i][3] = qwc.x();
    Twc_arr[i][4] = qwc.y();
    Twc_arr[i][5] = qwc.z();
    Twc_arr[i][6] = qwc.w();
    problem.AddParameterBlock(Twc_arr[i], 7, parameterization);
    problem.SetParameterBlockConstant(Twc_arr[i]);

    Eigen::Vector3d pc = Rwc.transpose() * (map_point - twc);
    Eigen::Vector3d meas = pc.normalized();
    InverseDistanceFactor *inverse_distance_factor = new InverseDistanceFactor(meas);
    Eigen::Matrix<double, 3, 3> sqrt_info = Eigen::Matrix<double, 3, 3>::Identity();
    inverse_distance_factor->SetinformationMatrix(sqrt_info);

    problem.AddResidualBlock(
        inverse_distance_factor,
        nullptr,
        Twa_arr, Twc_arr[i], uv_arr, inverse_dis_arr);
  }

  ceres::Solver::Options ops;
  ceres::Solver::Summary summary;
  ops.minimizer_progress_to_stdout = true;
  ceres::Solve(ops, &problem, &summary);

  std::cout << "uvp " << uv_arr[0] << " " << uv_arr[1] << " " << inverse_dis_arr[0] << std::endl;
}
