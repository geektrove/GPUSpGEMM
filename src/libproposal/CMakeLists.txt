add_library_wrapped(libproposal PUBLIC src/binning.cu src/sym.cu src/sym2.cu)
target_include_directories(libproposal PUBLIC include)

target_link_libraries(libproposal PUBLIC libutils)
