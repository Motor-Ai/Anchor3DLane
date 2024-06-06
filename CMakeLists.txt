cmake_minimum_required(VERSION 3.12)
project(postprocess)

find_package(pybind11 REQUIRED)

set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES src/postprocess.cpp src/utils.cpp)
set(HEADERS include/postprocess.hpp include/utils.hpp)

include_directories(include)


# For using standalone executable 
set(LIBRARY_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tools")
pybind11_add_module(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} PUBLIC)


# For doing standalone tests we make a cpp lib:
# add_library(${PROJECT_NAME} ${SOURCES})
# add_subdirectory(tests)

