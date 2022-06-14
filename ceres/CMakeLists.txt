cmake_minimum_required(VERSION 3.0.0)
project(pnp_test VERSION 0.1.0)
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("/usr/include/eigen3")
include_directories(include)

find_package(Ceres REQUIRED)
include_directories( ${CERES_INCLUDE_DIRS} )


add_executable(opt_Rt opt_Rt.cpp src/Rt_optimizer.cpp )
target_link_libraries(opt_Rt ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})

add_executable(opt_points opt_points.cpp   src/points_optimizer.cpp)
target_link_libraries(opt_points ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})

add_executable(inverse_depth inverse_depth.cpp  src/so3.cpp)
target_link_libraries(inverse_depth ${CERES_LIBRARIES})
add_executable(number_diff number_diff.cpp   src/so3.cpp)
target_link_libraries(number_diff ${CERES_LIBRARIES})

add_executable(inverse_depth_muti_cam inverse_depth_muti_cam.cpp   src/so3.cpp)
target_link_libraries(inverse_depth_muti_cam ${CERES_LIBRARIES})

add_executable(muti_cam_inverse_depth_opti_uvp muti_cam_inverse_depth_opti_uvp.cpp   src/so3.cpp)
target_link_libraries(muti_cam_inverse_depth_opti_uvp ${CERES_LIBRARIES})