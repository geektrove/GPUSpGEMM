function(add_executable_wrapped_benchmark name)
    add_executable_wrapped(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE benchmark::benchmark)
endfunction()

add_executable_wrapped(cusparse cusparse.cu)
target_link_libraries(cusparse PRIVATE libcusparse)

add_executable_wrapped_benchmark(cusparse_benchmark cusparse_benchmark.cu)
target_link_libraries(cusparse_benchmark PRIVATE libcusparse)

add_executable_wrapped(opsparse opsparse.cu)
target_link_libraries(opsparse PRIVATE libopsparse)
target_link_libraries(opsparse PRIVATE libcusparse)
target_link_libraries(opsparse PRIVATE CLI11::CLI11)

add_executable_wrapped(proposal proposal.cu)
target_link_libraries(proposal PRIVATE libproposal)
target_link_libraries(proposal PRIVATE libcusparse)
target_link_libraries(proposal PRIVATE CLI11::CLI11)
