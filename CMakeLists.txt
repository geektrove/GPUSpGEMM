cmake_minimum_required(VERSION 3.24)

include(cmake/preamble.cmake)

project(
    SpGEMMThesis
    VERSION 1.0.0
    LANGUAGES CXX CUDA)

include(cmake/options.cmake)
include(cmake/ipo.cmake)
include(cmake/dependencies.cmake)
include(cmake/wrappers.cmake)

add_subdirectory(src)
