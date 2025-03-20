add_library_wrapped(utils INTERFACE)
target_include_directories(utils INTERFACE include)

target_link_libraries(utils INTERFACE CUDA::cusparse)
