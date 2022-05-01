#include <iostream>
#include <thread>
#include <unistd.h>

void fun()
{
    for (int i = 0; i < 10; i++)
    {
        std::cout << "thread 2" << std::endl;
        usleep(1000000);
    }
}

void test_thread()
{
    std::thread t1([]()
                   {
        for(int i=0;i<10;i++){
        std::cout << "thread 1" <<std::endl;
        usleep(1000000);
    } });

    std::thread t2(fun);

    // t1.join();
    // t2.join();  //阻塞主线程

    t1.detach(); //不阻塞主线程
    t2.detach();
    for (int i = 0; i < 2; i++)
    {
        std::cout << "thread main" << std::endl;
        usleep(1000000);
    }
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
