cmake_minimum_required(VERSION 3.12)
project(postprocess)

# find_package(pybind11 REQUIRED)
# set(LIBRARY_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tools")

# add_library(postprocess src/postprocess.cpp src/utils.cpp)
add_executable(out src/postprocess.cpp src/postprocess.cpp)
