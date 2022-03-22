#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
using namespace std;

string gt_txt;
string traj_txt;
string out_txt;

struct Imu
{
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;
    double t;
    /* data */
};

void fun()
{
    ifstream f_gt(gt_txt);
    vector<Imu> gt_imus;
    while (!f_gt.eof())
    {
        string str;
        getline(f_gt, str);
        if (str.length() == 0 || str[0] == '#')
            continue;
        for (char &c : str)
            if (c == ',')
                c = ' ';
        stringstream ss(str);
        Imu imu;
        ss >> imu.t >> imu.twc.x() >> imu.twc.y() >> imu.twc.z() >> imu.qwc.x() >> imu.qwc.y() >> imu.qwc.z() >> imu.qwc.w();
        gt_imus.push_back(imu);
    }

    vector<Imu> tmp;
    for (int i = 1; i < gt_imus.size() - 1; ++i)
    {
        Eigen::Vector3d x1 = gt_imus[i - 1].twc;
        Eigen::Vector3d x2 = gt_imus[i + 1].twc;
        Eigen::Vector3d x_average = (gt_imus[i + 1].twc + gt_imus[i - 1].twc) / 2.0;
        Eigen::Vector3d x = gt_imus[i].twc;
        Eigen::Vector3d dx = x_average - x;
        double dis = dx.norm();
        if (dis < 0.2)
            tmp.push_back(gt_imus[i]);
    }
    gt_imus = tmp;
    // for (auto gt : gt_imus)
    // cout << gt.qwc.coeffs().transpose() << std::endl;

    ifstream f_traj(traj_txt);
    ofstream f_out(out_txt);
    while (!f_traj.eof())
    {
        string str;
        getline(f_traj, str);
        if (str.length() == 0 || str[0] == '#')
            continue;
        for (auto &c : str)
            if (c == ',')
                c == ' ';
        stringstream ss(str);
        double t;
        ss >> t;

        int lo = 0, hi = gt_imus.size() - 1;
        // cout << setprecision(10) << t << "ggg" << endl;
        while (hi != lo + 1)
        {
            const int mid = (lo + hi) / 2;
            if (gt_imus[mid].t <= t)
            {
                lo = mid;
            }
            else if (gt_imus[mid].t >= t)
            {
                hi = mid;
            }
        }

        if (hi >= gt_imus.size())
            continue;

        Eigen::Quaterniond qwc1 = gt_imus[lo].qwc;
        Eigen::Quaterniond qwc2 = gt_imus[hi].qwc;
        Eigen::Vector3d twc1 = gt_imus[lo].twc;
        Eigen::Vector3d twc2 = gt_imus[hi].twc;
        double t1 = gt_imus[lo].t;
        double t2 = gt_imus[hi].t;

        double dt1 = (t - t1) / (t2 - t1);
        double dt2 = (t2 - t) / (t2 - t1);
        Eigen::Quaterniond qq = qwc1.slerp(dt1, qwc2);
        Eigen::Vector3d tt = dt1 * twc2 + dt2 * twc1;
        f_out << setprecision(10) << t << " " << tt.x() << " " << tt.y() << " " << tt.z() << " " << qq.w() << " " << qq.x() << " " << qq.y() << " " << qq.z() << endl;
    }
}

int main(int argc, char **argv)
{
    gt_txt = argv[1];
    traj_txt = argv[2];
    out_txt = argv[3];
    fun();
    std::cout << "Hello, world!\n";
}
