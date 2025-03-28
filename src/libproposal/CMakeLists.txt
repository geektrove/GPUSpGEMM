add_library_wrapped(libproposal PUBLIC src/sym_binning.cu)
target_include_directories(libproposal PUBLIC include)

target_link_libraries(libproposal PUBLIC libutils)
