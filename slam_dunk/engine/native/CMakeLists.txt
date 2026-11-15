cmake_minimum_required(VERSION 3.18)
project(aria_vio)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -Wno-deprecated-declarations")
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
include_directories(include ../deps/usr/include ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
add_compile_definitions(BASALT_INSTANTIATIONS_DOUBLE EIGEN_DONT_PARALLELIZE)
add_library(bvio STATIC
src/vi_estimator/ba_base.cpp src/vi_estimator/sc_ba_base.cpp src/vi_estimator/sqrt_ba_base.cpp
src/vi_estimator/landmark_database.cpp src/vi_estimator/marg_helper.cpp src/vi_estimator/sqrt_keypoint_vio.cpp
src/linearization/landmark_block.cpp src/linearization/linearization_base.cpp src/linearization/linearization_abs_qr.cpp
src/linearization/linearization_abs_sc.cpp src/linearization/linearization_rel_sc.cpp
src/utils/vio_config.cpp src/utils/system_utils.cpp src/utils/time_utils.cpp)
target_link_libraries(bvio PUBLIC TBB::tbb fmt::fmt ${OpenCV_LIBS})
add_executable(aria_vio main.cpp)
target_link_libraries(aria_vio bvio)
find_package(Ceres REQUIRED)
add_executable(batch_time batch_time.cpp)
target_link_libraries(batch_time Ceres::ceres)
add_library(track_union SHARED track_union.cpp)

add_executable(batch_joint batch_joint.cpp)
target_link_libraries(batch_joint Ceres::ceres TBB::tbb)
