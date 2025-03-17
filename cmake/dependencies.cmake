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

FetchContent_MakeAvailable(gsl-lite)
FetchContent_MakeAvailable(fmt)
