# Add the proposal library containing the proposal implementation
add_library_wrapped(libproposal INTERFACE)
target_include_directories(libproposal INTERFACE include)
target_link_libraries(libproposal INTERFACE libutils)
