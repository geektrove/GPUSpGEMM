add_executable_wrapped(cusparse cusparse.cu)
target_link_libraries(cusparse PRIVATE libcusparse)
