# Library

add_library(libopsparse src/CSR.cu src/Meta.cu src/Timings.cu)
target_include_directories(libopsparse PUBLIC inc)

target_compile_features(libopsparse PRIVATE cxx_std_11 cuda_std_11)
set_target_properties(
    libopsparse
    PROPERTIES CXX_EXTENSIONS OFF
               CUDA_EXTENSIONS OFF
               CUDA_SEPARABLE_COMPILATION ON)
if(${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
    target_compile_options(
        libopsparse PRIVATE --extra-device-vectorization --use_fast_math
                            --restrict --expt-relaxed-constexpr)
endif()

target_link_libraries(libopsparse PUBLIC CUDA::cusparse)
target_link_libraries(libopsparse PUBLIC OpenMP::OpenMP_CUDA)

# Executable

add_executable(opsparse src/opsparse.cu)
target_include_directories(opsparse PUBLIC inc)
target_compile_features(opsparse PRIVATE cxx_std_11 cuda_std_11)
set_target_properties(
    opsparse
    PROPERTIES CXX_EXTENSIONS OFF
               CUDA_EXTENSIONS OFF
               CUDA_SEPARABLE_COMPILATION ON)
if(${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
    target_compile_options(
        opsparse PRIVATE --extra-device-vectorization --use_fast_math
                         --restrict --expt-relaxed-constexpr)
endif()

target_link_libraries(opsparse PRIVATE libopsparse)
target_link_libraries(opsparse PRIVATE utils)
target_link_libraries(opsparse PRIVATE CUDA::cusparse)
target_link_libraries(opsparse PRIVATE OpenMP::OpenMP_CUDA)
