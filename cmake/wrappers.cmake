function(add_executable_wrapped name)
    add_executable(${name} ${ARGN})
    _customize_target_wrapped(${name} PRIVATE)
endfunction()

function(add_library_wrapped name visibility)
    if(visibility STREQUAL "INTERFACE")
        add_library(${name} INTERFACE ${ARGN})
    else()
        add_library(${name} ${ARGN})
    endif()
    _customize_target_wrapped(${name} ${visibility})
endfunction()

function(_customize_target_wrapped name visibility)
    target_compile_features(${name} ${visibility} cxx_std_20 cuda_std_20)
    set_target_properties(
        ${name}
        PROPERTIES CXX_EXTENSIONS OFF
                   CUDA_EXTENSIONS OFF
                   CUDA_SEPARABLE_COMPILATION ON)

    if(${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
        target_compile_options(
            ${name} ${visibility} --extra-device-vectorization --use_fast_math
            --expt-relaxed-constexpr --extended-lambda)
    endif()

    target_compile_definitions(${name} ${visibility}
                                       CUDA_API_PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(
        ${name}
        ${visibility}
        $<$<CONFIG:Release>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_WARN>
        $<$<CONFIG:Debug,RelWithDebInfo>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG>
    )

    target_link_libraries(${name} ${visibility} gsl-lite)
    target_link_libraries(${name} ${visibility} fmt::fmt)
    target_link_libraries(${name} ${visibility} spdlog::spdlog)
    target_link_libraries(${name} ${visibility} nvtx3-cpp)
endfunction()
