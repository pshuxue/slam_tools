#include <iostream>
#include <thread>
#include <unistd.h>
#include <mutex>
#include <queue>
#include <condition_variable>

//类中函数的多线程用法，以及析构阻塞，防止主线程消失程序中断

class ThreadDemo
{
private:
  std::queue<int> que;
  std::thread thread_push;
  std::thread thread_pop;
  std::mutex mtx;

  bool is_running;
  std::condition_variable cv; //条件变量

public:
  ThreadDemo()
  {
    is_running = true;
    thread_push = std::thread(&ThreadDemo::PushThread, this);
    thread_pop = std::thread(&ThreadDemo::PopThread, this);
  }

  ~ThreadDemo() //主线程结束，程序就结束了，类中线程也会结束，这里让程序在析构函数中阻塞，直到程序完全结束
  {
    if (thread_push.joinable())
      thread_push.join();

    is_running = false;
    cv.notify_all(); //唤起所有线程
    if (thread_pop.joinable())
      thread_pop.join();
  }

private:
  void PushThread()
  {
    for (int i = 1; i <= 20; ++i)
    {
      {
        std::unique_lock<std::mutex> lck(mtx);
        que.push(i);
        std::cout << "push thread, push:" << i << std::endl;
      }
      if (i % 2 == 0)
        cv.notify_all(); //唤起所有线程
      sleep(1);
    }
  }

  void PopThread()
  {
    while (is_running)
    {
      if (que.empty())
      {
        std::unique_lock<std::mutex> lck(mtx);
        cv.wait(lck); // pop处理结束以后将线程挂起
      }
      while (!que.empty())
      {
        std::unique_lock<std::mutex> lck(mtx);
        std::cout << "pop thread, pop:" << que.front() << std::endl;
        que.pop();
      }
    }
  }
};

int main(int, char **)
{
  ThreadDemo demo;
  sleep(10); //主程序会先结束，但是析够函数阻塞所以不会结束
  return 0;
}
