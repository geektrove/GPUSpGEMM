add_library(utils INTERFACE)
target_sources(
    utils
    PUBLIC FILE_SET
           pubheaders
           TYPE
           HEADERS
           BASE_DIRS
           include/
           FILES
           include/utils/utils.cuh
           include/utils/csr.cuh
           include/utils/errors.cuh)

target_compile_features(utils INTERFACE cxx_std_20 cuda_std_20)
set_target_properties(
    utils
    PROPERTIES CXX_EXTENSIONS OFF
               CUDA_EXTENSIONS OFF
               CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(utils INTERFACE gsl-lite)
target_link_libraries(utils INTERFACE fmt::fmt)
