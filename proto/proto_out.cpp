#include <iostream>
#include <fstream>
#include "megslam_proto_keyframe.pb.h"
using namespace std;

int main(int, char **)
{
    megslam::proto::PointCloud cloud;

    //写入数据
    cout << "向点云中加入点：" << endl;
    for (int i = 0; i < 5; ++i)
    {
        cout << " ( " << i << ", " << i + 1 << ", " << i + 2 << " )" << endl;
        megslam::proto::Point3f *p = cloud.add_map_points(); //非基本变量不能直接加入，只能这样
        p->set_x(i);
        p->set_y(i + 1);
        p->set_z(i + 2);
    }
    cout << endl;

    cout << "向点云中加入描述子：" << endl;
    for (int i = 0; i < 5; ++i)
    {
        cout << " ( " << 1 << ", " << 3 << " ) : " << i << " " << i << " " << i << endl;
        megslam::proto::MatXf *desp = cloud.add_descriptors();
        desp->set_rows(1);
        desp->set_cols(3);
        desp->add_data(i); //基本变量可以直接加入
        desp->add_data(i);
        desp->add_data(i);
    }
    cout << endl;

    cout << "向点云中加入重投影误差：" << endl;
    for (int i = 0; i < 5; ++i)
    {
        cout << i / 10.0 << endl;
        cloud.add_reprojection_errors(i / 10.0);
    }

    //数据保存，序列化
    ofstream fout("out.bin");
    cloud.SerializePartialToOstream(&fout);

    return 0;
}
