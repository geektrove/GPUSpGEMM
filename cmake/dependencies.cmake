include(FetchContent)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

FetchContent_Declare(
    gsl-lite
    GIT_REPOSITORY https://github.com/gsl-lite/gsl-lite.git
    GIT_TAG v0.42.0
    SYSTEM)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.1.4
    SYSTEM)
set(BENCHMARK_ENABLE_GTEST_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING
    OFF
    CACHE BOOL "" FORCE)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
    SYSTEM)

FetchContent_MakeAvailable(gsl-lite)
FetchContent_MakeAvailable(fmt)
FetchContent_MakeAvailable(benchmark)
