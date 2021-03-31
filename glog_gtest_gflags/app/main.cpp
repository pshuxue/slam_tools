#include <gtest/gtest.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>

using namespace std;

TEST(GLog, glog)
{
  CHECK_EQ(1, 1);                //检测两个数是否相等，如果不相等会直接中断程序,还有很多CHECK,也可以自定义
  FLAGS_colorlogtostderr = true; //这样搞终端会显示不同颜色

  LOG(INFO) << "info";
  LOG(WARNING) << "warning";
  LOG(ERROR) << "error";
}

DEFINE_int32(flag1, 9527, "int");
DEFINE_string(flag2, "hello", "string");
DEFINE_double(flag3, 1.34, "double");

//gflags调用的时候是:  可执行文件 --flag1=1234 --flag2="test"
int main(int argc, char *argv[])
{
  //这两行代码不打开,log直接输出到终端,打开就会输出到指定文件夹保存为log文件
  //google::InitGoogleLogging(argv[0]);
  //FLAGS_log_dir = "/home/psx/workspace/colmap_learning/";

  gflags::ParseCommandLineFlags(&argc, &argv, true); // 这一句如果不加,则在终端输入的时候不会解析命令行

  cout << "flag1:" << FLAGS_flag1 << endl;
  cout << "flag2:" << FLAGS_flag2 << endl;
  cout << "flag3:" << FLAGS_flag3 << endl;

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
