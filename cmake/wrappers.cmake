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

    # Do not add "lib" prefix to the target name
    set_target_properties(${name} PROPERTIES PREFIX "")

    _customize_target_wrapped(${name} ${visibility})
endfunction()

function(_customize_target_wrapped name visibility)
    # Use C++20 and CUDA 20 without extensions and with separable compilation
    target_compile_features(${name} ${visibility} cxx_std_20 cuda_std_20)
    set_target_properties(
        ${name}
        PROPERTIES CXX_EXTENSIONS OFF
                   CUDA_EXTENSIONS OFF
                   CUDA_SEPARABLE_COMPILATION ON)

    # Add flags for NVIDIA compiler
    if(${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
        target_compile_options(
            ${name} ${visibility} --extra-device-vectorization --use_fast_math
            --expt-relaxed-constexpr --extended-lambda)
    endif()

    # Add compute macros if the target is not an interface. The macros are used
    # to enable compilation for different compute capabilities.
    if(NOT visibility STREQUAL "INTERFACE")
        get_target_property(TARGET_ARCHS ${name} CUDA_ARCHITECTURES)
        if(NOT TARGET_ARCHS STREQUAL "TARGET_ARCHS-NOTFOUND")
            if(TARGET_ARCHS IN_LIST "all;all-major;native")
                message(
                    FATAL_ERROR
                        "CUDA_ARCHITECTURES is set to 'all', 'all-major' or 'native' which is not supported"
                )
            endif()
            set(COMPUTE_MACROS "")
            foreach(arch IN LISTS TARGET_ARCHS)
                string(REGEX REPLACE "-.*" "" stripped_arch "${arch}")
                list(APPEND COMPUTE_MACROS "COMPUTE_${stripped_arch}")
            endforeach()
            list(REMOVE_DUPLICATES COMPUTE_MACROS)
            target_compile_definitions(${name} ${visibility} ${COMPUTE_MACROS})
        endif()
    endif()

    # Use double precision if the option is enabled
    if(USE_DOUBLE_PRECISION)
        target_compile_definitions(${name} ${visibility} USE_DOUBLE_PRECISION)
    endif()

    # Enable CUDA API per-thread default stream
    target_compile_definitions(${name} ${visibility}
                                       CUDA_API_PER_THREAD_DEFAULT_STREAM)

    # Disable NVTX in release and disable logging in release and relwithdebinfo
    target_compile_definitions(
        ${name}
        ${visibility}
        $<$<CONFIG:Release>:NVTX_DISABLE>
        $<$<CONFIG:Release,RelWithDebInfo>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_WARN>
        $<$<CONFIG:Debug>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE>)

    # Link CUDA runtime library
    target_link_libraries(${name} ${visibility} CUDA::cudart)

    # Link other dependencies
    target_link_libraries(${name} ${visibility} gsl-lite)
    target_link_libraries(${name} ${visibility} fmt::fmt)
    target_link_libraries(${name} ${visibility} spdlog::spdlog)
    target_link_libraries(${name} ${visibility} nvtx3-cpp)
endfunction()
