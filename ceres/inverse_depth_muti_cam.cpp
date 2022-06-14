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
#include "iomanip"

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

class RigInverseDistanceFactor : public ceres::SizedCostFunction<3, 7, 7, 2, 1> // imupose imupose uv p
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RigInverseDistanceFactor(const Eigen::Vector3d &meas_direction, const Eigen::Matrix4d &Tia, const Eigen::Matrix4d &Tic); //  Tij
  void SetinformationMatrix(const Eigen::Matrix<double, 3, 3> &info);
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  Eigen::Vector3d meas_direction_;
  Eigen::Matrix3d Ria; //与3d点绑定的图像对应imu的外参
  Eigen::Vector3d tia;
  Eigen::Matrix3d Ric; //没有与3d点绑定的图像对应imu的外参
  Eigen::Vector3d tic;

  Eigen::Matrix<double, 3, 3> sqrt_info_;
};

RigInverseDistanceFactor::RigInverseDistanceFactor(const Eigen::Vector3d &meas_direction, const Eigen::Matrix4d &Tia, const Eigen::Matrix4d &Tic)
{
  Ria = Tia.topLeftCorner(3, 3);
  tia = Tia.topRightCorner(3, 1);
  Ric = Tic.topLeftCorner(3, 3);
  tic = Tic.topRightCorner(3, 1);
  meas_direction_ = meas_direction;
  sqrt_info_ = Eigen::Matrix<double, 3, 3>::Identity();
}

void RigInverseDistanceFactor::SetinformationMatrix(const Eigen::Matrix<double, 3, 3> &info)
{
  sqrt_info_ = info;
}

bool RigInverseDistanceFactor::Evaluate(double const *const *parameters, double *residuals,
                                        double **jacobians) const
{
  // x表示Ic  y表示Ia
  const Eigen::Vector3d twy(parameters[0][0], parameters[0][1], parameters[0][2]);
  const Eigen::Quaterniond qwy(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  const Eigen::Matrix3d Rwy = qwy.toRotationMatrix();
  const Eigen::Matrix3d Rwa = Rwy * Ria;
  const Eigen::Vector3d twa = twy + Rwy * tia;

  const Eigen::Vector3d twx(parameters[1][0], parameters[1][1], parameters[1][2]);
  const Eigen::Quaterniond qwx(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
  const Eigen::Matrix3d Rwx = qwx.toRotationMatrix();
  const Eigen::Matrix3d Rxw = Rwx.transpose();
  const Eigen::Matrix3d Rwc = Rwx * Ric;
  const Eigen::Matrix3d Rcw = Rwc.transpose();
  const Eigen::Vector3d twc = twx + Rwx * tic;
  const Eigen::Matrix3d Rcy = Rcw * Rwy;
  const Eigen::Matrix3d Rca = Rcw * Rwa;
  const Eigen::Vector3d tca = Rcw * (twa - twc);
  // std::cout << "Rwc " << Rwc << std::endl;

  const double u = parameters[2][0];
  const double u2 = u * u;
  const double v = parameters[2][1];
  const double v2 = v * v;
  const double inverse = parameters[3][0];

  const double n = 2 / (u * u + v * v + 1);
  const Eigen::Vector3d pa(n * u, n * v, n - 1);
  // std::cout << "sspa " << pa.transpose() << std::endl;
  const Eigen::Vector3d pc = Rca * pa + inverse * tca;
  const Eigen::Vector3d pc_norm = pc.normalized();
  // std::cout<<Rwx<<std::endl;
  // std::cout << "twc sss  " << twa.transpose() << std::endl;
  // std::cout << "pc_norm " << pc_norm.transpose() << " " << inverse << std::endl;

  Eigen::Map<Eigen::Matrix<double, 3, 1>> residual(residuals);
  residual = pc_norm - meas_direction_;
  // std::cout << setprecision(15) << "error " << residual.transpose() << std::endl;
  // throw std::runtime_error("a");
  // std::cout << "residual " << residual.transpose() << std::endl;
  const double norm = pc.norm();
  const double norm_3 = norm * norm * norm;

  const Eigen::Matrix3d dr_dpc = -1 / norm_3 * pc * pc.transpose() + 1 / norm * Eigen::Matrix3d::Identity();

  if (jacobians)
  {
    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> dr_dTwy(jacobians[0]);
      Eigen::Matrix<double, 3, 7> dpc_dTwy;
      dpc_dTwy.setZero();

      dpc_dTwy.block<3, 3>(0, 0) = inverse * Rcw;                                                             // dpc_dtwa
      dpc_dTwy.block<3, 3>(0, 3) = -Rcy * Sophus::SO3::hat(Ria * pa) - Rcy * Sophus::SO3::hat(inverse * tia); // dpc_dRwa
      // std::cout << "err " << Rcw * Sophus::SO3::hat(inverse * Rwy * tia) << std::endl;
      // std::cout<<Rwa << std::endl;
      dr_dTwy = dr_dpc * dpc_dTwy;
      dr_dTwy = sqrt_info_ * dr_dTwy;
    }

    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> dr_dTwx(jacobians[1]);
      Eigen::Matrix<double, 3, 7> dpc_dTwx;
      dpc_dTwx.setZero();

      dpc_dTwx.block<3, 3>(0, 0) = -inverse * Rcw;                                                                               // dpc_dtwc
      dpc_dTwx.block<3, 3>(0, 3) = Ric.transpose() * Sophus::SO3::hat(Rxw * Rwa * pa + inverse * Rxw * (twy + Rwy * tia - twx)); // dpc_dRwc
      dr_dTwx = dr_dpc * dpc_dTwx;
      dr_dTwx = sqrt_info_ * dr_dTwx;
    }

    if (jacobians[2])
    {
      Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> dr_duv(jacobians[2]);
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
    if (jacobians[3])
    {
      Eigen::Map<Eigen::Vector3d> dr_ds(jacobians[3]);
      Eigen::Vector3d dpc_ds;
      dpc_ds = tca;
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

bool PPlus(const double *x, const double *delta, double *x_plus_delta)
{
  Eigen::Map<const Eigen::Vector3d> _p(x);
  Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

  Eigen::Map<const Eigen::Vector3d> dp(delta);

  Eigen::Quaterniond dq = deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

  Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
  Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

  p = _p + dp;
  q = (dq*_q ).normalized();

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
  // Test2();
  // Test3();
  return 0;
}
//-1.16815 -0.782205  0.256189  -1.16897 -0.782336  0.255315
//-1.16912 -0.781456  0.255153 -1.16897 -0.782336  0.255315
//-1.16901 -0.783182  0.255282 -1.16897 -0.782336  0.255315
void Test1()
{
  std::cout << "optimizer Twa" << std::endl;
  double d = 0;
  // double d = 1e-5;
  double dt[6] = {0, 0, 0, 0, 0, d};
  double Twa_arr[7] = {0.7, 0.2, 0.1, 0, 0, 0.8, 0.6}; // txtytzqxqyqzqw
  Eigen::Matrix3d Rwa = Eigen::Matrix3d::Identity();
  Eigen::Vector3d twa(0, 0, 0);

  Eigen::Matrix4d Tia = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d Tic = Eigen::Matrix4d::Identity();
  Eigen::AngleAxisd vec(0.4, Eigen::Vector3d::Random().normalized());
  Tia.topLeftCorner(3, 3) = vec.toRotationMatrix();
  Tia.topRightCorner(3, 1) = Eigen::Vector3d::Random();
  Eigen::Matrix4d Tai = Tia.inverse();
  std::cout << "Tia" << Tia << std::endl;
  PrintT(Tai);

  constexpr int number_point = 1;
  constexpr int number_image = 1;

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
    double TT[7];
    TT[0] = twc.x();
    TT[1] = twc.y();
    TT[2] = twc.z();
    TT[3] = qwc.x();
    TT[4] = qwc.y();
    TT[5] = qwc.z();
    TT[6] = qwc.w();

    Twc_arr[i][0] = twc.x();
    Twc_arr[i][1] = twc.y();
    Twc_arr[i][2] = twc.z();
    Twc_arr[i][3] = qwc.x();
    Twc_arr[i][4] = qwc.y();
    Twc_arr[i][5] = qwc.z();
    Twc_arr[i][6] = qwc.w();
    PPlus(TT, dt, Twc_arr[i]);
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
      RigInverseDistanceFactor *inverse_distance_factor = new RigInverseDistanceFactor(meas, Tia, Tic);
      Eigen::Matrix<double, 3, 3> sqrt_info = Eigen::Matrix<double, 3, 3>::Identity();
      inverse_distance_factor->SetinformationMatrix(sqrt_info);

      problem.AddResidualBlock(
          inverse_distance_factor,
          nullptr,
          Twa_arr, Twc_arr[j], uv_arr[i], inverse_dis_arr[i]);

      std::vector<const ceres::LocalParameterization *> local_parameterizations;
      local_parameterizations.push_back(parameterization);
      local_parameterizations.push_back(parameterization);
      local_parameterizations.push_back(nullptr);
      local_parameterizations.push_back(nullptr);

      ceres::NumericDiffOptions numeric_diff_options;

      std::vector<double *> parameter_blocks;
      parameter_blocks.push_back(Twa_arr);
      parameter_blocks.push_back(Twc_arr[j]);
      parameter_blocks.push_back(uv_arr[i]);
      parameter_blocks.push_back(inverse_dis_arr[i]);

      ceres::GradientChecker::ProbeResults results;
      ceres::GradientChecker checker(inverse_distance_factor, &local_parameterizations, numeric_diff_options);
      checker.Probe(parameter_blocks.data(), 1e-5, &results);

      if (!checker.Probe(parameter_blocks.data(), 1e-5, &results))
      {
        std::cout << "An error has occurred:\n"
                  << results.error_log << std::endl;
      }
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

  Eigen::Matrix4d Tia = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d Tic = Eigen::Matrix4d::Identity();
  Eigen::AngleAxisd vec(0.3, Eigen::Vector3d::Random().normalized());
  Tic.topLeftCorner(3, 3) = vec.toRotationMatrix();
  Tic.topRightCorner(3, 1) = Eigen::Vector3d::Random();
  Eigen::Matrix4d Tci = Tic.inverse();
  std::cout << "Tic" << Tic << std::endl;
  PrintT(Tci);
  double TwIc_arr[7] = {0.1, 0.2, -0.4, 0, 0.8, 0, 0.6}; // txtytzqxqyqzqw
  Eigen::Matrix3d Rwc = Eigen::Matrix3d::Identity();
  Eigen::Vector3d twc(0, 0, 0);

  constexpr int number_point_image = 20000;

  double uv_arr[number_point_image][2];
  double inverse_dis_arr[number_point_image][1];
  double TwIa_arr[number_point_image][7];
  std::vector<Eigen::Vector3d> map_points;
  std::vector<Eigen::Matrix3d> Rwas;
  std::vector<Eigen::Vector3d> twas;

  ceres::Problem problem;
  ceres::LocalParameterization *parameterization =
      new PoseLocalParameterization;
  problem.AddParameterBlock(TwIc_arr, 7, parameterization);

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
    TwIa_arr[i][0] = twa.x();
    TwIa_arr[i][1] = twa.y();
    TwIa_arr[i][2] = twa.z();
    TwIa_arr[i][3] = qwa.x();
    TwIa_arr[i][4] = qwa.y();
    TwIa_arr[i][5] = qwa.z();
    TwIa_arr[i][6] = qwa.w();

    problem.AddParameterBlock(uv_arr[i], 2);
    problem.SetParameterBlockConstant(uv_arr[i]);
    problem.AddParameterBlock(inverse_dis_arr[i], 1);
    problem.SetParameterBlockConstant(inverse_dis_arr[i]);
    problem.AddParameterBlock(TwIa_arr[i], 7, parameterization);
    problem.SetParameterBlockConstant(TwIa_arr[i]);

    Eigen::Vector3d pc = Rwc.transpose() * (map_point - twc);

    RigInverseDistanceFactor *inverse_distance_factor = new RigInverseDistanceFactor(pc.normalized(), Eigen::Matrix4d::Identity(), Tic);
    Eigen::Matrix<double, 3, 3> sqrt_info = Eigen::Matrix<double, 3, 3>::Identity();
    inverse_distance_factor->SetinformationMatrix(sqrt_info);

    problem.AddResidualBlock(
        inverse_distance_factor,
        nullptr,
        TwIa_arr[i], TwIc_arr, uv_arr[i], inverse_dis_arr[i]);
    // std::cout << i << std::endl;
  }

  ceres::Solver::Options ops;
  ceres::Solver::Summary summary;
  ops.minimizer_progress_to_stdout = true;
  ceres::Solve(ops, &problem, &summary);

  std::cout << "Twc ";
  for (int i = 0; i < 7; ++i)
  {
    std::cout << TwIc_arr[i] << " ";
  }
  std::cout << endl;
}

void Test3()
{
  std::cout << "optimizer uvp" << std::endl;

  double TwIa_arr[7] = {0.7, 0, 0, 0, 0, 0, 1}; // txtytzqxqyqzqw
  Eigen::Matrix3d Rwa = Eigen::Matrix3d::Identity();
  Eigen::Vector3d twa(0.7, 0, 0);

  constexpr int number_image = 100;

  double uv_arr[2] = {0.5, -0.4};
  double inverse_dis_arr[1] = {0.8};

  double TwIc_arr[number_image][7];

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
  problem.AddParameterBlock(TwIa_arr, 7, parameterization);
  problem.SetParameterBlockConstant(TwIa_arr);
  problem.AddParameterBlock(uv_arr, 2);
  problem.AddParameterBlock(inverse_dis_arr, 1);

  for (int i = 0; i < number_image; ++i)
  {
    Eigen::AngleAxisd vec(0.1, Eigen::Vector3d::Random());
    Eigen::Matrix3d Rwc = vec.toRotationMatrix();
    Eigen::Vector3d twc = Eigen::Vector3d::Random() * 3;
    Eigen::Quaterniond qwc(Rwc);
    TwIc_arr[i][0] = twc.x();
    TwIc_arr[i][1] = twc.y();
    TwIc_arr[i][2] = twc.z();
    TwIc_arr[i][3] = qwc.x();
    TwIc_arr[i][4] = qwc.y();
    TwIc_arr[i][5] = qwc.z();
    TwIc_arr[i][6] = qwc.w();
    problem.AddParameterBlock(TwIc_arr[i], 7, parameterization);
    problem.SetParameterBlockConstant(TwIc_arr[i]);

    Eigen::Vector3d pc = Rwc.transpose() * (map_point - twc);
    Eigen::Vector3d meas = pc.normalized();
    RigInverseDistanceFactor *inverse_distance_factor = new RigInverseDistanceFactor(meas, Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity());
    Eigen::Matrix<double, 3, 3> sqrt_info = Eigen::Matrix<double, 3, 3>::Identity();
    inverse_distance_factor->SetinformationMatrix(sqrt_info);

    problem.AddResidualBlock(
        inverse_distance_factor,
        nullptr,
        TwIa_arr, TwIc_arr[i], uv_arr, inverse_dis_arr);
  }

  ceres::Solver::Options ops;
  ceres::Solver::Summary summary;
  ops.minimizer_progress_to_stdout = true;
  ceres::Solve(ops, &problem, &summary);

  std::cout << "uvp " << uv_arr[0] << " " << uv_arr[1] << " " << inverse_dis_arr[0] << std::endl;
}
