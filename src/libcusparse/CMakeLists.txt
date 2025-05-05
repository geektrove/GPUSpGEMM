# Add the cusparse library containing the cuSPARSE 1 and cuSPARSE 2
# implementations
add_library_wrapped(libcusparse INTERFACE)
target_include_directories(libcusparse INTERFACE include)
target_link_libraries(libcusparse INTERFACE libutils)
target_link_libraries(libcusparse INTERFACE CUDA::cusparse)
