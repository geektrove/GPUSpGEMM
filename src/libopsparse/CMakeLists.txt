add_library_wrapped(libopsparse PUBLIC src/CSR.cu src/Meta.cu src/Timings.cu
                    src/OpSparse.cu)
target_include_directories(libopsparse PUBLIC include)
target_include_directories(libopsparse PRIVATE include/opsparse)

target_link_libraries(libopsparse PUBLIC libutils)
target_link_libraries(libopsparse PUBLIC CUDA::cusparse)
target_link_libraries(libopsparse PRIVATE OpenMP::OpenMP_CXX)
