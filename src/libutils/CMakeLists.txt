# Add the utilities library
add_library_wrapped(libutils INTERFACE)
target_include_directories(libutils INTERFACE include)
target_link_libraries(libutils INTERFACE CUDA::cusparse)
