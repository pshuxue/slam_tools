#include <eigen3/Eigen/Sparse>
#include <vector>
#include <iostream>

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{

  { //稀疏矩阵的构造
    std::vector<T> triplets;
    SpMat matrix(6, 4);
    triplets.emplace_back(1, 2, 1.14); //第1行第2列设置为3.14
    triplets.emplace_back(0, 2, 2.14); //第1行第2列设置为3.14
    triplets.emplace_back(2, 1, 0.14); //第1行第2列设置为3.14
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    std::cout << matrix << std::endl;
    std::cout << matrix.rows() << std::endl;
    std::cout << matrix.cols() << std::endl;
    std::cout << matrix.innerSize() << std::endl; //在内存中排布的列数，当ColMajor时 等于rows
    std::cout << matrix.outerSize() << std::endl; //在内存中排布的行数，当ColMajor时 等于cols
    std::cout << matrix.nonZeros() << std::endl;
    std::cout << matrix.adjoint() << std::endl; //伴随矩阵

    for (int k = 0; k < matrix.outerSize(); ++k) //按内存遍历，outerSize保证了无论是行存储还是列存储，最终都能用最高的效率遍历
      for (SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it)
      {
        std::cout << "it.value(): " << it.value() << std::endl;
        std::cout << "it.row(): " << it.row() << std::endl;     // row index
        std::cout << "it.col(): " << it.col() << std::endl;     // col index (here it is equal to k)
        std::cout << "it.index(): " << it.index() << std::endl; // inner index, here it is equal to it.row()
      }
  }

  { //矩阵块操作
    SparseMatrix<double, ColMajor> sm1;
    sm1.resize(4,4);
    // sm1.leftCols(ncols) 
    std::cout << sm1.middleCols(1, 3).transpose() << std::endl;
    // sm1.rightCols(ncols) 
// 
    // SparseMatrix<double, RowMajor> sm2;
    // sm2.row(i) = ...;
    // sm2.topRows(nrows) = ...;
    // sm2.middleRows(i, nrows) = ...;
    // sm2.bottomRows(nrows) = ...;
  }

  return 0;
}