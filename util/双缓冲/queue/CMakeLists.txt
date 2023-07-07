cmake_minimum_required(VERSION 3.17)
project(test_queue CXX)

set( CMAKE_BUILD_TYPE "Debug" )
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wall")

if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
 set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -frounding-math -O0 -g -ggdb")
 option(SAN_ADDR "option for sanitize=address" ON)
 if(SAN_ADDR)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
 endif()

 option(SAN_LEAK "option for sanitize=leak" ON)
 if(SAN_LEAK)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak")
 endif()

 option(SAN_THREAD "option for sanitize=thread" ON)
 if((NOT SAN_ADDR) AND SAN_THREAD)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread")
 endif()

 message("Debug mode:${CMAKE_C_FLAGS_DEBUG}")
 add_definitions(-D_DEBUG)
endif()

include_directories(/home/psx/workspace/AnchorEngine/3rdparty/waterdrop/3rdparty/flatbuffers)
add_executable(test_queue queue.cpp )
target_link_libraries(test_queue /home/psx/workspace/AnchorEngine/3rdparty/waterdrop/3rdparty/flatbuffers/ubuntu20.04/libflatbuffers.a)