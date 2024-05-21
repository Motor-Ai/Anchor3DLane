cmake_minimum_required(VERSION 3.12)
project(postprocess)

find_package(pybind11 REQUIRED)
set(LIBRARY_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tools")

pybind11_add_module(postprocess src/postprocess.cpp )
