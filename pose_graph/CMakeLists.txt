cmake_minimum_required(VERSION 3.0.0)
project(pose_graph VERSION 0.1.0)
find_package(Ceres REQUIRED)


include_directories(    ${CERES_INCLUDE_DIRS} "/usr/include/eigen3")


add_executable(pose_graph pose_graph.cpp so3.cpp)
target_link_libraries(pose_graph ${CERES_LIBRARIES})

