cmake_minimum_required(VERSION 3.10)
project(test_glog_gtest_gflags)
set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Release" )

set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -std=c++11 -pthread")  #没有-pthread,gtest不能运行
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin ) #设定可执行文件的输出路径

############### dependencies ######################
#opencv
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

#eigen3
include_directories( "/usr/include/eigen3" )

#glog
find_package(Glog REQUIRED)
include_directories( ${Glog_INCLUDE_DIRS})

#gtest
find_package(GTest REQUIRED)
include_directories( ${GTest_INCLUDE_DIRS})

#gflags
find_package(GFlags REQUIRED)
include_directories( ${GFlags_INCLUDE_DIRS})

set( THIRD_PARTY_LIBS
        ${OpenCV_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GTEST_LIBRARIES}
        ${GTEST_MAIN_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        )

include_directories(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(app)
