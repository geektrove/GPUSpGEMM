# Library

add_library_wrapped(libopsparse PUBLIC src/CSR.cu src/Meta.cu src/Timings.cu)
target_include_directories(libopsparse PUBLIC inc)

target_link_libraries(libopsparse PUBLIC utils)
target_link_libraries(libopsparse PUBLIC CUDA::cusparse)
target_link_libraries(libopsparse PRIVATE OpenMP::OpenMP_CXX)

# Executable (opsparse)

add_executable_wrapped(opsparse src/opsparse.cu)
target_include_directories(opsparse PUBLIC inc)

target_link_libraries(opsparse PRIVATE libopsparse)
target_link_libraries(opsparse PRIVATE utils)
target_link_libraries(opsparse PRIVATE CUDA::cusparse)
target_link_libraries(opsparse PRIVATE OpenMP::OpenMP_CXX)
