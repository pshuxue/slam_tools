#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>
#include <iostream>
using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
  { //简单操作
    Eigen::Matrix2d mat;
    mat << 1, -2,
        3, 4;
    cout << "Here is mat.sum():       " << mat.sum() << endl;
    cout << "Here is mat.连乘():      " << mat.prod() << endl;
    cout << "Here is mat.mean():      " << mat.mean() << endl;
    cout << "Here is mat.minCoeff():  " << mat.minCoeff() << endl;
    cout << "Here is mat.maxCoeff():  " << mat.maxCoeff() << endl;
    cout << "Here is mat.trace():     " << mat.trace() << endl;
    cout << "Here is mat.绝对值():     " << mat.cwiseAbs() << endl;
  }

  { //向量和矩阵的各种范数
    Eigen::VectorXf v(2);
    Eigen::MatrixXf m(2, 2), n(2, 2);
    v << -1,
        2;
    m << 1, -2,
        -3, 4;
    std::cout << "v.squaredNorm() = " << v.squaredNorm() << std::endl;
    std::cout << "v.norm() = " << v.norm() << std::endl;
    std::cout << "v.lpNorm<1>() = " << v.lpNorm<1>() << std::endl;
    std::cout << "v.lpNorm<Infinity>() = " << v.lpNorm<Eigen::Infinity>() << std::endl;

    std::cout << std::endl;
    std::cout << "m.squaredNorm() = " << m.squaredNorm() << std::endl;
    std::cout << "m.norm() = " << m.norm() << std::endl;
    std::cout << "m.lpNorm<1>() = " << m.lpNorm<1>() << std::endl;
    std::cout << "m.lpNorm<Infinity>() = " << m.lpNorm<Eigen::Infinity>() << std::endl;
  }

  { //矩阵查找最大最小元素
    Eigen::MatrixXf m(2, 2);
    m << 1, 2,
        3, 4;

    // get location of maximum
    Eigen::Index maxRow, maxCol;
    float max = m.maxCoeff(&maxRow, &maxCol);

    // get location of minimum
    Eigen::Index minRow, minCol;
    float min = m.minCoeff(&minRow, &minCol);

    std::cout << "Max: " << max << ", at: " << maxRow << "," << maxCol << std::endl;
    std::cout << "Min: " << min << ", at: " << minRow << "," << minCol << std::endl;

    Eigen::MatrixXf mat(2, 4);
    mat << 1, 2, 6, 9,
        3, 1, 7, 2;
    std::cout << "Column's maximum: " << std::endl
              << mat.colwise().maxCoeff() << std::endl; // colwise是逐列
    std::cout << "Row's maximum: " << std::endl
              << mat.rowwise().maxCoeff() << std::endl;
  }

  { //逐列找最临近元素
    Eigen::MatrixXf m(2, 4);
    Eigen::VectorXf v(2);
    m << 1, 23, 6, 9,
        3, 11, 7, 2;
    v << 2,
        3;
    Eigen::Index index;
    // find nearest neighbour
    (m.colwise() - v).colwise().squaredNorm().minCoeff(&index);

    std::cout << "Nearest neighbour is column " << index << ":" << std::endl;
    std::cout << m.col(index) << std::endl;
  }

  { // Map映射
    int array[8];
    for (int i = 0; i < 8; ++i)
      array[i] = i;
    cout << "Column-major:\n"
         << Map<Matrix<int, 2, 4>>(array) << endl;
    cout << "Row-major:\n"
         << Map<Matrix<int, 2, 4, RowMajor>>(array) << endl;
    cout << "Row-major using stride:\n"
         << Map<Matrix<int, 2, 4>, Unaligned, Stride<1, 4>>(array) << endl;
  }

  { // eval的使用，返回一个中间的变量，避免还没拷贝完就被改写了
    {
      MatrixXi mat(3, 3);
      mat << 1, 2, 3, 4, 5, 6, 7, 8, 9;
      cout << "Here is the matrix mat:\n"
           << mat << endl;
      // This assignment shows the aliasing problem
      mat.bottomRightCorner(2, 2) = mat.topLeftCorner(2, 2);
      cout << "After the assignment, mat = \n"
           << mat << endl;
    }
    {
      MatrixXi mat(3, 3);
      mat << 1, 2, 3, 4, 5, 6, 7, 8, 9;
      cout << "Here is the matrix mat:\n"
           << mat << endl;
      // The eval() solves the aliasing problem
      mat.bottomRightCorner(2, 2) = mat.topLeftCorner(2, 2).eval();
      cout << "After the assignment, mat = \n"
           << mat << endl;
    }
  }


  return 0;
}