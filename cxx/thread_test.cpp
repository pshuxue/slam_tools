#include <iostream>
#include <thread>
#include <unistd.h>
#include <mutex>
#include <queue>

std::mutex mtx;
std::queue<int> ques;

void fun()
{
    while (1)
    {
        {
            std::unique_lock<std::mutex> lck(mtx);
            if (!ques.empty())
            {
                std::cout << "thread 2 " << ques.front() << std::endl;
                ques.pop();
            }
        }
        usleep(1000000);
    }
}

void test_thread()
{
    std::thread t1([]()
                   {
        for(int i=0;i<10;i++){
        {
            std::unique_lock<std::mutex> lck (mtx);
            ques.push(i);
            std::cout << "thread 1 "<<i <<std::endl;
        }
        usleep(1000000);
    } });

    std::thread t2(fun);

    // t1.join();
    // t2.join();  //阻塞主线程

    t1.detach(); //不阻塞主线程
    t2.detach();
}

int main(int, char **)
{
    test_thread();
    while (1)
    {
        std::cout << "mmm" << std::endl;
        usleep(111111111);
    }
    std::cout << "Hello, world!\n";
}
