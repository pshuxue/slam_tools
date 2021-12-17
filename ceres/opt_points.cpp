//使用opencv进行pnp估计，然后再使用ceres优化
#include <iostream>
#include "points_optimizer.h"
#include <Eigen/Core>
using namespace std;

Eigen::Vector3d pw(2, 3, 10);

int num_camere = 100;
Eigen::Matrix3d K;
vector<Eigen::Matrix4d> Tcws;
vector<Eigen::Vector2d> uvs;

void GetData();
int main()
{
  GetData();
  lvm::PointsOptimizer solver;
  solver.SetInitialVal(Eigen::Vector3d(2, 3, 10));
  for (int idx = 0; idx < num_camere; ++idx)
  {
    solver.AddResidualItemAutoDiff(uvs[idx], Tcws[idx], K);
  }
  solver.Solve();
  cout << solver.GetPoint() << endl;
}

void GetData()
{
  K << 400, 0, 400, 0, 400, 400, 0, 0, 1;
  for (int i = 0; i < num_camere; ++i)
  {
    Eigen::AngleAxisd vec(double(rand() % 10000), Eigen::Vector3d::Random().normalized());
    Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
    Tcw.topLeftCorner(3, 3) = vec.toRotationMatrix();
    Tcw.topRightCorner(3, 1) = Eigen::Vector3d::Random() * 10;
    Tcws.push_back(Tcw);

    Eigen::Vector3d pc = K * (Tcw.topLeftCorner(3, 3) * pw + Tcw.topRightCorner(3, 1));
    Eigen::Vector2d uv;
    uv.x() = int(pc.x() / pc.z()) + rand() % 8 - 4;
    uv.y() = int(pc.y() / pc.z()) + rand() % 8 - 4;
    cout << uv.transpose() << endl;
    uvs.push_back(uv);
  }
}