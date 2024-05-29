cmake_minimum_required(VERSION 3.12)
project(postprocess)

find_package(pybind11 REQUIRED)

set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES src/postprocess.cpp)

# For doing standalone tests we make a cpp lib:
add_library(${PROJECT_NAME} ${SOURCES})
add_subdirectory(tests)

# For using standalone executable 

# set(LIBRARY_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tools")
# pybind11_add_module(${PROJECT_NAME} src/postprocess.cpp )
