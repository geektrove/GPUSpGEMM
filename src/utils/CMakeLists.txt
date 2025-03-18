add_library(utils INTERFACE)
target_include_directories(utils INTERFACE include)

target_compile_features(utils INTERFACE cxx_std_20 cuda_std_20)
set_target_properties(
    utils
    PROPERTIES CXX_EXTENSIONS OFF
               CUDA_EXTENSIONS OFF
               CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(utils INTERFACE CUDA::cusparse)
target_link_libraries(utils INTERFACE gsl-lite)
target_link_libraries(utils INTERFACE fmt::fmt)
