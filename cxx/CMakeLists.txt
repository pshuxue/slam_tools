cmake_minimum_required(VERSION 3.0.0)
project(cxx VERSION 0.1.0 )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


add_executable(thread_test1 thread_test1.cpp)
add_executable(thread_test2 thread_test2.cpp)

