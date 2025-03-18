add_executable(cusparse cusparse.cu)

target_compile_features(cusparse PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(
    cusparse
    PROPERTIES CXX_EXTENSIONS OFF
               CUDA_EXTENSIONS OFF
               CUDA_SEPARABLE_COMPILATION ON)
if(${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
    target_compile_options(
        cusparse PRIVATE --extra-device-vectorization --use_fast_math
                         --restrict --expt-relaxed-constexpr)
endif()

target_link_libraries(cusparse PRIVATE utils)
target_link_libraries(cusparse PRIVATE CUDA::cusparse)
target_link_libraries(cusparse PRIVATE gsl-lite)
target_link_libraries(cusparse PRIVATE fmt::fmt)
