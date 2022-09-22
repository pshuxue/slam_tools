#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <ctime>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <random>
#include <unordered_map>
#include <vector>
// g++ -std=c++11 11.cpp -I/usr/include/eigen3 -I/usr/local/include/opencv4
// -lopencv_core

using namespace Eigen;
using namespace std;

/// @brief laplace 平滑
///
/// @param connectityMap 连接关系
/// @param vWeight 先验权重
/// @param vPointType 顶点类型，0: 不优化, 1:优化
/// @param vDepth 深度先验
/// @param vDepthRet 求解后深度
/// @return
bool laplaceSmoothing(const cv::Mat &connectityMap,
                      const Eigen::VectorXf &vWeight,
                      const Eigen::VectorXi &vPointType,
                      const Eigen::VectorXf &vDepth,
                      Eigen::VectorXf &vDepthRet)
{

  using SparseType = SparseMatrix<float, Eigen::RowMajor>;
  using Key = long;
  using LtLType = std::unordered_map<Key, float>;

  auto make_key = [](int r, int c) -> long
  { return r * 40000L + c; };
  auto row_from_key = [](Key key) -> int
  {
    return static_cast<int>(key / 40000L);
  };
  auto col_from_key = [](Key key) -> int
  {
    return static_cast<int>(key % 40000L);
  };
  auto accumulate = [](LtLType &ltl, Key key, float v) -> void
  {
    if (ltl.find(key) == ltl.end())
    {
      ltl.emplace(key, 0);
    }
    ltl[key] += v;
  };

  SparseType L;
  SparseType LtL;
  SparseType W;
  SparseType A;
  SparseType B;

  auto t0 = std::chrono::steady_clock::now();

  // L 矩阵 (Laplace)
  std::vector<Triplet<float>> LTriplets;
  // L^t*L 矩阵
  std::vector<Triplet<float>> LtLTriplets;

  // L^t*L 构建用
  std::unordered_map<Key, float> LtLMaps;
  //
  std::vector<Triplet<float>> ATriplets;

  std::vector<Triplet<float>> BTriplets;

  int rows = connectityMap.rows;
  int cols = connectityMap.cols;
  int size = rows * cols;

  int mCnt = 0;

  // 构建LtL
  for (int i = 0; i < size; i++)
  {
    int row = i / cols;
    int col = i % cols;
    int row4[4] = {row, row, row - 1, row + 1};
    int col4[4] = {col - 1, col + 1, col, col};

    std::vector<int> conn_idx;
    for (int j = 0; j < 4; j++)
    {
      if (row4[j] >= 0 && row4[j] < rows && col4[j] >= 0 && col4[j] < cols)
      {
        int idx = row4[j] * cols + col4[j];
        if (connectityMap.at<cv::Vec4b>(row, col)[j] == 1)
        {
          conn_idx.push_back(idx);
        }
      }
    }

    int degree = conn_idx.size();

    // 如果顶点为孤立点
    if (degree == 0)
    {
      std::cout << "Error: Found isolated node..." << std::endl;
      throw std::runtime_error("Error: Found isolated node...");
    }

    for (int idx : conn_idx)
    {
      accumulate(LtLMaps, make_key(i, idx), -degree);
      accumulate(LtLMaps, make_key(idx, i), -degree);

      for (int idx2 : conn_idx)
      {
        accumulate(LtLMaps, make_key(idx, idx2), 1);
      }
      LTriplets.emplace_back(i, idx, -1);
    }

    if (vPointType[i] != 0)
    { // 不是slam点
      mCnt++;
    }
    accumulate(LtLMaps, make_key(i, i), degree * degree);
    LTriplets.emplace_back(i, i, conn_idx.size());
  }

  for (auto &elem : LtLMaps)
  {
    LtLTriplets.emplace_back(row_from_key(elem.first), col_from_key(elem.first),
                             elem.second);
  }

  L.resize(size, size);
  L.setFromTriplets(LTriplets.begin(), LTriplets.end());

  LtL.resize(size, size);
  LtL.setFromTriplets(LtLTriplets.begin(), LtLTriplets.end());
  auto t1 = std::chrono::steady_clock::now();

  if (0)
  {
    MatrixXf lMat = MatrixXf(L);
    MatrixXf ltlMat = MatrixXf(lMat.transpose() * lMat);
    float max_diff = (MatrixXf(LtL) - ltlMat).cwiseAbs().maxCoeff();

    std::cout << lMat << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << ltlMat << std::endl;

    std::cout << "==========================" << std::endl;
    std::cout << "max_diff = " << max_diff << std::endl;

    if (max_diff > 1e-5)
    {
      throw new std::runtime_error("wrong LtL");
    }

    std::cout << "==========================" << std::endl;
  }

  // 构建A和B, Xm和Xn
  Eigen::VectorXf vXm(mCnt);
  Eigen::VectorXf vXn(size - mCnt);
  std::unordered_map<int, int> reorderMap;
  std::unordered_map<int, int> iReorderMap;
  int mIdx = 0;
  int nIdx = 0;
  for (int i = 0; i < vPointType.rows(); i++)
  {
    if (vPointType[i] == 0)
    {
      reorderMap.emplace(i, nIdx);
      vXn[nIdx] = vDepth[i];
      nIdx++;
    }
    else
    {
      reorderMap.emplace(i, mIdx);
      iReorderMap.emplace(mIdx, i);
      vXm[mIdx] = vWeight[i] * vDepth[i];
      mIdx++;
    }
  }

  for (int i = 0; i < LtL.outerSize(); i++)
  {
    if (vPointType[i] != 0)
    {
      for (typename SparseType::InnerIterator it(LtL, i); it; ++it)
      {
        if (vPointType[it.col()] != 0)
        {
          ATriplets.emplace_back(reorderMap.at(it.row()),
                                 reorderMap.at(it.col()), it.value());
        }
        else
        {
          BTriplets.emplace_back(reorderMap.at(it.row()),
                                 reorderMap.at(it.col()), it.value());
        }
      }
    }
  }

  A.resize(mCnt, mCnt);
  A.setFromTriplets(ATriplets.begin(), ATriplets.end());

  // A+W
  for (int i = 0; i < vPointType.rows(); i++)
  {
    if (vPointType[i] != 0)
    {
      int row = reorderMap.at(i);
      A.coeffRef(row, row) += vWeight[i];
    }
  }

  B.resize(mCnt, size - mCnt);
  B.setFromTriplets(BTriplets.begin(), BTriplets.end());

  Eigen::VectorXf b = -B * vXn + vXm;

  auto t2 = std::chrono::steady_clock::now();
  // 求解
  Eigen::ConjugateGradient<SparseType, Eigen::Upper | Eigen::Lower> solver;
  solver.compute(A);
  solver.setTolerance(0.0001);
  Eigen::VectorXf vXmRet = solver.solve(b);

  // Eigen::SimplicialCholesky<SparseMatrix<float>> solver(A);
  // Eigen::VectorXf vXmRet = solver.solve(b);


  auto t3 = std::chrono::steady_clock::now();

  // 返回结果
  vDepthRet = vDepth;
  for (int i = 0; i < vXmRet.rows(); i++)
  {
    vDepthRet[iReorderMap.at(i)] = vXmRet[i];
  }

  std::cout << vDepthRet.transpose() << std::endl;

  std::cout
      << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
      << "ms" << std::endl;

  std::cout
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << "ms" << std::endl;

  std::cout
      << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
      << "ms" << std::endl;

  return true;
}

int main()
{
  int row = 128;
  int col = 128;
  int N = row * col;
  cv::Mat connectityMap(row, col, CV_8UC4);
  connectityMap.setTo(cv::Vec4b(0, 0, 0, 0));

  int row4[4] = {0, 0, -1, 1};
  int col4[4] = {-1, 1, 0, 0};

  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      for (int k = 0; k < 4; k++)
      {
        if (i + row4[k] >= 0 && i + row4[k] < row && j + col4[k] >= 0 &&
            j + col4[k] < col)
        {
          connectityMap.at<cv::Vec4b>(i, j)[k] = 1;
        }
      }
    }
  }

  Eigen::VectorXi vPointType = Eigen::VectorXi::Ones(N);

  std::default_random_engine gen(time(0));
  std::uniform_real_distribution<double> dist(0, 1);
  dist(gen);

  Eigen::VectorXf vWeight(N);
  vWeight.setRandom();
  vWeight = vWeight.cwiseAbs();
  vWeight.setZero();
  vWeight[1290] = 10;
  vWeight[3870] = 10;
  vWeight[9030] = 10;

  Eigen::VectorXf vDepth(N);
  vDepth.setZero();
  Eigen::VectorXf vDepthRet(N);

  // vPointType[0] = 0;
  // vPointType[N - 1] = 0;

  vPointType[0] = 0;
  vPointType[16383] = 0;
  vPointType[127] = 0;
  vPointType[16256] = 0;
  vPointType[8127] = 0;

  //   vPointType[2] = 0;
  vDepth[0] = 0;
  vDepth[16383] = 0;
  vDepth[127] = 0;
  vDepth[16256] = 0;
  vDepth[8127] = 100;
  vDepth[1290] = 0;
  vDepth[3870] = 0;
  vDepth[9030] = 0;


  // vDepth[0] = 10;
  // vDepth[N - 1] = 10;
  //   vDepth[2] = 10;
  //   vDepth[2] = 18;

  laplaceSmoothing(connectityMap, vWeight, vPointType, vDepth, vDepthRet);

  std::ofstream fout("cxx_optimize.txt");
  for (int i = 0; i < N; i++)
  {
    fout << i << " " << vDepthRet(i) << std::endl;
  }
  fout.close();

  return 0;
}