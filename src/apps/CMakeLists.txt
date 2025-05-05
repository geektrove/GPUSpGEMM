# Build cuSPARSE algorithm 1 executable
add_executable_wrapped(cusparse1 cusparse1.cu)
target_link_libraries(cusparse1 PRIVATE libcusparse)
target_link_libraries(cusparse1 PRIVATE CLI11::CLI11)

# Build cuSPARSE algorithm 2 executable
add_executable_wrapped(cusparse2 cusparse2.cu)
target_link_libraries(cusparse2 PRIVATE libcusparse)
target_link_libraries(cusparse2 PRIVATE CLI11::CLI11)

# Build OpSparse executable
add_executable_wrapped(opsparse opsparse.cu)
target_link_libraries(opsparse PRIVATE libopsparse)
target_link_libraries(opsparse PRIVATE libcusparse) # For validation
target_link_libraries(opsparse PRIVATE CLI11::CLI11)

# Build the proposed implementation executable
add_executable_wrapped(proposal proposal.cu)
target_link_libraries(proposal PRIVATE libproposal)
target_link_libraries(proposal PRIVATE libcusparse) # For validation
target_link_libraries(proposal PRIVATE CLI11::CLI11)
