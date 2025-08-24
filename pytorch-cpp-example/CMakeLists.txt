cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pytorch-cpp-example)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(mytest mytest.cpp)
target_link_libraries(mytest "${TORCH_LIBRARIES}")
set_property(TARGET mytest PROPERTY CXX_STANDARD 17)