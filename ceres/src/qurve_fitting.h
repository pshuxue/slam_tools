#pragma once
#include <ceres/ceres.h>
#include <vector>
using namespace std;

//y=a*x*x+b*x+c
//根据xy的值拟合a b c

double a = 2, b = 3, c = 4;

struct FitCeres
{
    FitCeres(double x, double y) : x_(x), y_(y) {}

    // 残差的计算
    template <typename T>
    bool operator()(
        const T *const abc, // 未知参数，有3维
        T *residual) const  // 残差
    {
        T aa = abc[0];
        T bb = abc[1];
        T cc = abc[2];
        residual[0] = x_ * x_ * aa + x_ * bb + cc - y_;
        return true;
    }
    static ceres::CostFunction *Create(const double x, const double y)
    {
        return (new ceres::AutoDiffCostFunction<FitCeres, 1, 3>(
            new FitCeres(x, y)));
    }

    // virtual bool Evaluate(double const *parameters,
    //                       double *residuals,
    //                       double **jacobians) const
    // {
    //     residuals[0] = parameters[0];
    //     residuals[0] = parameters[1];
    //     residuals[0] = parameters[2];

    //     // 计算Jacobian
    //     if (jacobians != NULL && jacobians[0] != NULL)
    //     {
    //         jacobians[0][0] = -1;
    //     }
    //     return true;
    // }

    const double x_;
    const double y_;
};

void QurveFitting()
{
    vector<double> xs;
    vector<double> ys;
    for (int i = 0; i < 100; ++i)
    {
        double x = 50 * ((rand() % 1000) / 1000.0);
        double y = a * x * x + b * x + c + 0.5 * ((rand() % 1000) / 1000.0);
        xs.push_back(x);
        ys.push_back(y);
        // cout << "data : " << x << " " << y << endl;
    }
    double abc[3] = {0, 0, 0};

    ceres::Problem problem;
    for (int i = 0; i < xs.size(); ++i)
    {
        ceres::CostFunction *cost_function =
            FitCeres::Create(xs[i], ys[i]);
        problem.AddResidualBlock(cost_function,
                                 NULL /* squared loss */,
                                 abc);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << "优化后： " << abc[0] << " " << abc[1] << " " << abc[2] << endl;
}