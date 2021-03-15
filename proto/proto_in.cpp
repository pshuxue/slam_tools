#include <iostream>
#include <fstream>
#include "megslam_proto_keyframe.pb.h"
using namespace std;

int main(int, char **)
{
    megslam::proto::PointCloud cloud;

    //数据加载，反序列化
    ifstream fin("out.bin");
    cloud.ParseFromIstream(&fin);

    cout << "点云中的点有：" << endl;
    for (auto iter = cloud.map_points().begin(); iter != cloud.map_points().end(); ++iter)
    {
        cout << " ( " << iter->x() << ", " << iter->y() << ", " << iter->z() << " )" << endl;
    }
    cout << endl;

    cout << "点云中的描述子有: " << endl;
    for (auto iter1 = cloud.descriptors().begin(); iter1 != cloud.descriptors().end(); ++iter1)
    {
        cout << " ( " << iter1->rows() << ", " << iter1->cols() << " ) : ";
        for (auto iter2 = iter1->data().begin(); iter2 != iter1->data().end(); ++iter2)
        {
            cout << *iter2 << " "; //基本变量指针直接就可以访问
        }
        cout << endl;
    }
    cout << endl;

    cout << "点云中的重投影误差有：" << endl;
    for (auto iter = cloud.reprojection_errors().begin(); iter != cloud.reprojection_errors().end(); ++iter)
    {
        cout << *iter << endl;
    }
    return 0;
}
