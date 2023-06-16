#include "Eigen/Eigen"
#include "ceres/ceres.h"
#include <iostream>
#include <opencv2/opencv.hpp>

typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> Vector2fs;

class Order3BezierlineFactor : public ceres::SizedCostFunction<2, 2, 2, 2, 2, 1>  // P0 P1 P2 P3 t
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Order3BezierlineFactor(const Eigen::Vector2d& point) { point_ = point; }
  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const;

  Eigen::Vector2d point_;
};

bool Order3BezierlineFactor::Evaluate(double const* const* parameters, double* residuals,
                                      double** jacobians) const {
  Eigen::Map<const Eigen::Vector2d> P0(parameters[0]);
  Eigen::Map<const Eigen::Vector2d> P1(parameters[1]);
  Eigen::Map<const Eigen::Vector2d> P2(parameters[2]);
  Eigen::Map<const Eigen::Vector2d> P3(parameters[3]);
  const double t = parameters[4][0];

  Eigen::Map<Eigen::Vector2d> residual(residuals);
  const double r0 = (1 - t) * (1 - t) * (1 - t);
  const double r1 = 3 * t * (1 - t) * (1 - t);
  const double r2 = 3 * t * t * (1 - t);
  const double r3 = t * t * t;
  residual = r0 * P0 + r1 * P1 + r2 * P2 + r3 * P3 - point_;

  if (jacobians) {
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 2>> dr_P0(jacobians[0]);
      dr_P0.setZero();
      dr_P0(0, 0) = r0;
      dr_P0(1, 1) = r0;
    }
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> dr_P1(jacobians[1]);
      dr_P1.setZero();
      dr_P1(0, 0) = r1;
      dr_P1(1, 1) = r1;
    }
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> dr_P2(jacobians[2]);
      dr_P2.setZero();
      dr_P2(0, 0) = r2;
      dr_P2(1, 1) = r2;
    }
    if (jacobians[3]) {
      Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> dr_P3(jacobians[3]);
      dr_P3.setZero();
      dr_P3(0, 0) = r3;
      dr_P3(1, 1) = r3;
    }
    if (jacobians[4]) {
      Eigen::Map<Eigen::Vector2d> dr_t(jacobians[4]);
      dr_t = -3 * (1 - t) * (1 - t) * P0 + (3 + 9 * t * t - 12 * t) * P1 +
             (6 * t - 9 * t * t) * P2 + 3 * t * t * P3;
    }
  }
  return true;
}

class Order3BezierlineManager {
 private:
  double p0[2];
  double p1[2];
  double p2[2];
  double p3[2];
  int knn_number = 5;

  Eigen::Vector2f FindEndPoint(const Vector2fs& points, cv::flann::Index& kdtree,
                               const Eigen::Vector2f& begin_point) {
    Eigen::Vector2f result;

    std::queue<int> que;
    std::vector<uchar> visited(points.size(), 0);

    cv::Mat p_search(1, 2, CV_32FC1);
    p_search.at<float>(0, 0) = begin_point.x();
    p_search.at<float>(0, 1) = begin_point.y();
    cv::Mat indices, dists;
    kdtree.knnSearch(p_search, indices, dists, knn_number);
    for (int i = 0; i < knn_number; ++i) {
      int nei_idx = indices.at<int>(0, i);
      if (visited[nei_idx] == 1) continue;
      visited[nei_idx] = 1;
      que.push(nei_idx);
    }

    while (!que.empty()) {
      int idx = que.front();
      que.pop();

      result = points[idx];

      cv::Mat p_search(1, 2, CV_32FC1);
      p_search.at<float>(0, 0) = points[idx].x();
      p_search.at<float>(0, 1) = points[idx].y();
      cv::Mat indices, dists;
      kdtree.knnSearch(p_search, indices, dists, knn_number);

      for (int i = 1; i < knn_number; ++i) {
        int nei_idx = indices.at<int>(0, i);
        if (visited[nei_idx] == 1) continue;
        visited[nei_idx] = 1;
        que.push(nei_idx);
      }
    }
    return result;
  }

  bool FindFourEndPoints(const Vector2fs& points, Eigen::Vector2f& p0, Eigen::Vector2f& p1,
                         Eigen::Vector2f& p2, Eigen::Vector2f& p3) {
    cv::Mat dataset(points.size(), 2, CV_32FC1);
    for (int i = 0; i < points.size(); i++) {
      dataset.at<float>(i, 0) = points[i].x();
      dataset.at<float>(i, 1) = points[i].y();
    }
    cv::flann::Index kdtree(dataset, cv::flann::KDTreeIndexParams());  // 创建KDTree

    p0 = FindEndPoint(points, kdtree, points[points.size() / 2]);
    p3 = FindEndPoint(points, kdtree, p0);

    p1 = 1.0 / 3.0 * p0 + 2.0 / 3.0 * p3;
    p2 = 2.0 / 3.0 * p0 + 1.0 / 3.0 * p3;
    return true;
  }

 public:
  Order3BezierlineManager() {}

  bool AddPoints(const Vector2fs& points) {
    if (points.size() <= knn_number * 2) return false;
    Eigen::Vector2f p0_eigen, p1_eigen, p2_eigen, p3_eigen;
    FindFourEndPoints(points, p0_eigen, p1_eigen, p2_eigen, p3_eigen);

    p0[0] = p0_eigen.x();
    p0[1] = p0_eigen.y();
    p1[0] = p1_eigen.x();
    p1[1] = p1_eigen.y();
    p2[0] = p2_eigen.x();
    p2[1] = p2_eigen.y();
    p3[0] = p3_eigen.x();
    p3[1] = p3_eigen.y();

    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
    std::vector<double> ts(points.size(), 0.5);
    for (auto& t : ts) {
      problem.AddParameterBlock(&t, 1);
      problem.SetParameterLowerBound(&t, 0, 0);
      problem.SetParameterUpperBound(&t, 0, 1);
    }
    problem.AddParameterBlock(p0, 2);
    problem.AddParameterBlock(p1, 2);
    problem.AddParameterBlock(p2, 2);
    problem.AddParameterBlock(p3, 2);
    problem.SetParameterBlockConstant(p0);
    problem.SetParameterBlockConstant(p3);

    for (int i = 0; i < points.size(); ++i) {
      ceres::CostFunction* cost_function = new Order3BezierlineFactor(points[i].cast<double>());
      problem.AddResidualBlock(cost_function, loss_function, p0, p1, p2, p3, &ts[i]);
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    std::cout << "p0 = (" << p0[0] << ", " << p0[1] << ")" << std::endl;
    std::cout << "p1 = (" << p1[0] << ", " << p1[1] << ")" << std::endl;
    std::cout << "p2 = (" << p2[0] << ", " << p2[1] << ")" << std::endl;
    std::cout << "p3 = (" << p3[0] << ", " << p3[1] << ")" << std::endl;
    return true;
  }

  Vector2fs GetSamplePoints(int numPoints) {
    Vector2fs result;
    double step = 1.0 / double(numPoints);
    for (double t = 0; t < 1; t += step) {
      const double r0 = (1 - t) * (1 - t) * (1 - t);
      const double r1 = 3 * t * (1 - t) * (1 - t);
      const double r2 = 3 * t * t * (1 - t);
      const double r3 = t * t * t;
      Eigen::Map<const Eigen::Vector2d> P0(p0);
      Eigen::Map<const Eigen::Vector2d> P1(p1);
      Eigen::Map<const Eigen::Vector2d> P2(p2);
      Eigen::Map<const Eigen::Vector2d> P3(p3);
      Eigen::Vector2f p = (r0 * P0 + r1 * P1 + r2 * P2 + r3 * P3).cast<float>();
      result.push_back(p);
    }
    return result;
  }

  ~Order3BezierlineManager() {}
};
