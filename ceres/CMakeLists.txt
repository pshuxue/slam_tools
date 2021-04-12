cmake_minimum_required(VERSION 3.0.0)
project(ceres_demo VERSION 0.1.0)

find_package(Ceres REQUIRED)
include_directories( ${CERES_INCLUDE_DIRS} )
include_directories( src )

add_executable(ceres_demo main.cpp )
target_link_libraries(ceres_demo ${CERES_LIBRARIES})

