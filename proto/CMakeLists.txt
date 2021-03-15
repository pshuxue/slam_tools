cmake_minimum_required(VERSION 3.0.0)
project(proto_test VERSION 0.1.0)

find_package(Protobuf REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/proto)

add_executable(proto_out proto_out.cpp ${PROJECT_SOURCE_DIR}/proto/slam_proto_cloud.pb.cc)
target_link_libraries(proto_out ${Protobuf_LIBRARIES})

add_executable(proto_in proto_in.cpp ${PROJECT_SOURCE_DIR}/proto/slam_proto_cloud.pb.cc)
target_link_libraries(proto_in ${Protobuf_LIBRARIES})
