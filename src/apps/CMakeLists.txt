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

add_executable_wrapped_benchmark(opsparse_benchmark opsparse_benchmark.cu)
target_link_libraries(opsparse_benchmark PRIVATE libopsparse)

add_executable_wrapped(proposal proposal.cu)
target_link_libraries(proposal PRIVATE libproposal)

add_executable_wrapped_benchmark(proposal_benchmark proposal_benchmark.cu)
target_link_libraries(proposal_benchmark PRIVATE libproposal)
