#include <iostream>
#include <iomanip>
#include <ceres/ceres.h>
#include "so3.h"
#include <fstream>
#include <sstream>
#include "ceres/gradient_checker.h"

using namespace std;

class LaserOdometryFactor : public ceres::SizedCostFunction<6, 7, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LaserOdometryFactor(const Eigen::Matrix4d &meas_T);
    void SetInformationMatrix(const Eigen::Matrix<double, 6, 6> &info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    Eigen::Vector3d meas_t_;
    Eigen::Quaterniond meas_q_;
    Eigen::Matrix3d meas_R_;
    Eigen::Matrix<double, 6, 6> sqrt_info_;
};

LaserOdometryFactor::LaserOdometryFactor(const Eigen::Matrix4d &meas_T)
{
    meas_R_ = meas_T.topLeftCorner(3, 3);
    meas_q_ = Eigen::Quaterniond(meas_R_);
    meas_t_ = meas_T.topRightCorner(3, 1);
}

void LaserOdometryFactor::SetInformationMatrix(const Eigen::Matrix<double, 6, 6> &info)
{
    sqrt_info_ = info;
    // sqrt_info_ = Eigen::LLT<Eigen::Matrix<double,6,6>>(info).matrixL().transpose();
}

bool LaserOdometryFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Vector3d twl_i(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond qwl_i(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
    Eigen::Matrix3d Rwl_i = qwl_i.toRotationMatrix();
    Eigen::Matrix3d Rlw_i = Rwl_i.transpose();
    // std::cout << "twl_i: " << twl_i.transpose() << std::endl;
    // std::cout << "Rwl_i: " << Rwl_i << std::endl;

    Eigen::Vector3d twl_j(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond qwl_j(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
    Eigen::Matrix3d Rwl_j = qwl_j.toRotationMatrix();
    // std::cout << "twl_j: " << twl_j.transpose() << std::endl;
    // std::cout << "Rwl_j: " << Rwl_j << std::endl;

    Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
    residual.head(3) = Rlw_i * (twl_j - twl_i) - meas_t_; // t
    Eigen::Matrix3d rR = meas_R_.transpose() * Rlw_i * Rwl_j;
    residual.tail(3) = Sophus::SO3(rR).log(); // R
    residual = sqrt_info_ * residual;
    // std::cout << "laser residual: " << residual.transpose() << std::endl;

    if (jacobians)
    {
        Eigen::Matrix3d Jrinv = Sophus::SO3::JacobianRInv(residual.tail(3));
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_twl_i(jacobians[0]);
            jacobian_twl_i.setZero();

            jacobian_twl_i.block<3, 3>(0, 0) = -Rlw_i;                                    // dt_dt
            jacobian_twl_i.block<3, 3>(0, 3) = Sophus::SO3::hat(Rlw_i * (twl_j - twl_i)); // dt_dR
            jacobian_twl_i.block<3, 3>(3, 3) = -Jrinv * Rwl_j.transpose() * Rwl_i;        // dR_dR

            jacobian_twl_i = sqrt_info_ * jacobian_twl_i;
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_twl_j(jacobians[1]);
            jacobian_twl_j.setZero();

            jacobian_twl_j.block<3, 3>(0, 0) = Rlw_i; // dt_dt
            jacobian_twl_j.block<3, 3>(3, 3) = Jrinv; // dR_dR

            jacobian_twl_j = sqrt_info_ * jacobian_twl_j;
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

struct IMU
{
    double t;
    Eigen::Matrix4d Twc;
    double arr[7];
    bool flag = false;
    IMU()
    {
        flag = false;
        t = 0;
        Twc = Eigen::Matrix4d::Identity();
    }
};

void fun(vector<IMU> &imus, int i, int j);
int main(int argc, char **argv)
{
    //该程序默认首尾几帧pose相同,由于一般pose会比较多,几千帧,先每十个采样一个做posegraph
    //然后再fix住这些采样点,优化一下采样点之间的pose,得到最终的pose
    string input = "../traj_loop2.txt";
    string output = "output.txt";
    int bias = 10;
    ifstream fin(input);
    vector<IMU> imus;
    ceres::Problem problem;
    int ign = 0;
    while (!fin.eof())
    {
        string str;
        getline(fin, str);
        IMU imu;
        if (str.length() == 0)
            continue;
        stringstream ss(str);
        Eigen::Quaterniond qwc;
        Eigen::Vector3d twc;
        ss >> imu.t >> twc.x() >> twc.y() >> twc.z() >> qwc.w() >> qwc.x() >> qwc.y() >> qwc.z();
        imu.Twc = Eigen::Matrix4d::Identity();
        imu.Twc.topLeftCorner(3, 3) = qwc.toRotationMatrix();
        imu.Twc.topRightCorner(3, 1) = twc;
        imu.arr[0] = twc.x();
        imu.arr[1] = twc.y();
        imu.arr[2] = twc.z();
        imu.arr[3] = qwc.x();
        imu.arr[4] = qwc.y();
        imu.arr[5] = qwc.z();
        imu.arr[6] = qwc.w();
        if (ign++ % 10 == 0)
        {
            imu.flag = true;
        }
        imus.push_back(imu);
    }

    imus[imus.size() - 1].flag = true;
    ceres::LocalParameterization *parameterization =
        new PoseLocalParameterization;
    for (auto &imu : imus)
    {
        if (imu.flag)
        {
            problem.AddParameterBlock(imu.arr, 7, parameterization);
        }
    }

    int num = 0;
    for (int i = 0; i < imus.size(); ++i)
    {
        if (!imus[i].flag)
            continue;
        for (int step = 1; step <= bias; ++step)
        {
            if (!imus[i + step].flag || i + step >= imus.size())
                continue;

            cout << "edge: " << i << " " << step + i << endl;
            Eigen::Matrix4d Tij = imus[i].Twc.inverse() * imus[i + step].Twc;
            LaserOdometryFactor *laser_odom_factor = new LaserOdometryFactor(Tij);
            Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::Matrix<double, 6, 6>::Identity();
            laser_odom_factor->SetInformationMatrix(sqrt_info);
            // cout << Tij << endl;
            problem.AddResidualBlock(
                laser_odom_factor,
                nullptr,
                imus[i].arr, imus[i + step].arr);
            break;
        }
    }

    int begin = -1, end = -1;

    for (int step = 0; step < 11; ++step)
    {
        if (imus[0 + step].flag)
        {
            begin = 0 + step;
            break;
        }
    }
    for (int step = 0; step < 11; ++step)
    {
        if (imus[imus.size() - 1 - step].flag)
        {
            end = imus.size() - 1 - step;
            break;
        }
    }
    cout << "edge: " << begin << " " << end << endl;

    Eigen::Matrix4d Tij = Eigen::Matrix4d::Identity();
    LaserOdometryFactor *laser_odom_factor = new LaserOdometryFactor(Tij);
    Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::Matrix<double, 6, 6>::Identity();
    laser_odom_factor->SetInformationMatrix(sqrt_info);
    // // cout << Tij << endl;
    problem.AddResidualBlock(
        laser_odom_factor,
        nullptr,
        imus[begin].arr, imus[end].arr);

    problem.SetParameterBlockConstant(imus[0].arr);

    ceres::Solver::Options ops;
    ceres::Solver::Summary summary;
    ops.minimizer_progress_to_stdout = true;

    ceres::Solve(ops, &problem, &summary);

    // sleep(2);

    begin = 0;
    for (int i = 0; i < imus.size(); ++i)
    {
        if (imus[i].flag)
            begin = i;
        break;
    }

    for (int i = 0; i < imus.size(); ++i)
    {
        if (imus[i].flag)
        {
            fun(imus, begin, i);
            begin = i;
        }
    }
    ofstream fout(output);
    for (auto &imu : imus)
    {
        // if (imu.flag)
        fout << setprecision(12) << imu.t << " " << imu.arr[0] << " " << imu.arr[1] << " " << imu.arr[2] << " " << imu.arr[6] << " " << imu.arr[3] << " " << imu.arr[4] << " " << imu.arr[5] << endl;
    }

    //==================================================
    return 0;
}

void fun(vector<IMU> &imus, int i, int j)
{
    ceres::Problem problem1;

    ceres::LocalParameterization *parameterization =
        new PoseLocalParameterization;

    for (int ii = i; ii <= j; ++ii)
    {
        problem1.AddParameterBlock(imus[ii].arr, 7, parameterization);
    }

    problem1.SetParameterBlockConstant(imus[i].arr);
    problem1.SetParameterBlockConstant(imus[j].arr);

    for (int idx = i; idx <= j - 1; ++idx)
    {
        // cout << i << " " << idx << " " << j << endl;
        Eigen::Matrix4d Tij = imus[idx].Twc.inverse() * imus[idx + 1].Twc;
        LaserOdometryFactor *laser_odom_factor = new LaserOdometryFactor(Tij);
        Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::Matrix<double, 6, 6>::Identity();
        laser_odom_factor->SetInformationMatrix(sqrt_info);
        // cout << Tij << endl;
        problem1.AddResidualBlock(
            laser_odom_factor,
            nullptr,
            imus[idx].arr, imus[idx + 1].arr);

        std::vector<const ceres::LocalParameterization *> local_parameterizations;
        local_parameterizations.push_back(parameterization);
        local_parameterizations.push_back(parameterization);

        ceres::NumericDiffOptions numeric_diff_options;

        std::vector<double *> parameter_blocks;
        parameter_blocks.push_back(imus[idx].arr);
        parameter_blocks.push_back(imus[idx + 1].arr);

        ceres::GradientChecker::ProbeResults results;
        ceres::GradientChecker checker(laser_odom_factor, &local_parameterizations, numeric_diff_options);
        checker.Probe(parameter_blocks.data(), 1e-5, &results);

        if (!checker.Probe(parameter_blocks.data(), 1e-5, &results))
        {
            std::cout << "An error has occurred:\n"
                      << results.error_log << std::endl;
        }
    }
    ceres::Solver::Options ops;
    ceres::Solver::Summary summary;
    // ops.minimizer_progress_to_stdout = true;
    ceres::Solve(ops, &problem1, &summary);

    // std::cout << "Hello, world!\n";
    // std::cout << summary.BriefReport() << "\n";
}
