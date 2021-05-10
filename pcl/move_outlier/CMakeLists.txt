cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(pcl_outlier)
find_package(PCL 1.3 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(move_outlier move_outlier.cpp)
target_link_libraries(move_outlier ${PCL_LIBRARIES})

