cmake_minimum_required(VERSION 3.30)

include(cmake/preamble.cmake)

project(
    SpGEMMThesis
    VERSION 0.1.0
    LANGUAGES CXX CUDA)

include(cmake/ipo.cmake)
include(cmake/dependencies.cmake)

add_subdirectory(src)
