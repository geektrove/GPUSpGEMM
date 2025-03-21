add_library_wrapped(libcusparse INTERFACE)
target_include_directories(libcusparse INTERFACE include)

target_link_libraries(libcusparse INTERFACE libutils)
target_link_libraries(libcusparse INTERFACE CUDA::cusparse)
