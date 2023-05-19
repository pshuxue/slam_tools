#include <iostream>
#include <thread>
#include <unistd.h>
#include <mutex>
#include <queue>

struct Data{
  uint64_t timestamp;
};

class ThreadDemo{

public:
  //一个线程读取一个线程pop
  std::mutex mtx;  //临界资源的锁
  std::queue<Data> ques; 
  int que_max_size = 10;

  void read()
  {
    while (1)
    {
      {
        std::unique_lock<std::mutex> lck(mtx);
        if (!ques.empty())
        {
          std::cout << "read thread " << ques.front().timestamp << std::endl;
          std::cout << "read thread " << ques.size()<< std::endl;
          ques.pop();
        }
      }
      //这里建议读取的操作时间尽可能短，也就是上面这个大括号内的内容运行时间尽可能短，这里的usleep可以改成处理函数
      sleep(2);
    }
  }

  void write(int test)
  {
    std::cout <<"单纯的测试下传参-"<< test << std::endl;
    while (1)
    {
      static int count = 0;
      Data data;
      data.timestamp = count++;
      {
        std::unique_lock<std::mutex> lck (mtx);
        while(ques.size() >= que_max_size) {
          std::cout << "white thread data overflow, data = " << ques.front().timestamp << std::endl; 
          ques.pop();
        }
        ques.push(data);
        std::cout << "write thread "<< data.timestamp <<std::endl;
      }
      //还是那句话，上面大括号的用时尽可能短，处理函数放在下面usleep
      sleep(1);
    }
  }

  void TestThread() {
    t1 = std::thread(&ThreadDemo::write,this,3);
    t2 = std::thread(&ThreadDemo::read,this);
    //这里哪一个想要阻塞主线程，直接改成join就行
    t1.detach();
    t2.detach();
  }
  std::thread t1,t2;
};

int main(int, char **)
{
  ThreadDemo demo;
  demo.TestThread();
  //主线程结束，子线程也就结束了。
  while (1)
  {
    std::cout<<"main thread "<<std::endl;
    sleep(1);
  }
  std::cout << "Hello, world!\n";
}
