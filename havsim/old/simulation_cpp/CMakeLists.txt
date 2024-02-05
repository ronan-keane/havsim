cmake_minimum_required(VERSION 3.16)

project(havsim_cpp)

set(CMAKE_CXX_STANDARD 11)

add_library(havsim_cpp vehicle.cpp vehicle.h lane.cpp lane.h model.cpp model.h test_set_lc.cpp)

target_include_directories(havsim_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_infinite_road test_infinite_road.cpp)

add_executable(test_set_lc test_set_lc.cpp)


target_link_libraries(test_infinite_road LINK_PUBLIC havsim_cpp)

target_link_libraries(test_set_lc LINK_PUBLIC havsim_cpp)
