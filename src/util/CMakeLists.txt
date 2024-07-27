add_library(luisa-render-util SHARED
        sampling.cpp sampling.h
        frame.cpp frame.h
        imageio.cpp imageio.h
        xform.cpp xform.h
        spec.cpp spec.h
        colorspace.h
        rng.cpp rng.h
        ies.cpp ies.h
        scattering.cpp scattering.h
        bluenoise.cpp bluenoise.h
        sobolmatrices.cpp sobolmatrices.h
        pmj02tables.cpp pmj02tables.h
        complex.h
        medium_tracker.cpp medium_tracker.h
        progress_bar.cpp progress_bar.h
        loop_subdiv.cpp loop_subdiv.h
        vertex.h
        counter_buffer.cpp counter_buffer.h
        polymorphic_closure.h
        command_buffer.cpp command_buffer.h
        thread_pool.cpp thread_pool.h
        mesh_base.cpp mesh_base.h
        spheres_base.cpp spheres_base.h)

find_package(Threads)

target_link_libraries(luisa-render-util PUBLIC
        Threads::Threads
        luisa::compute
        luisa-render-include
        luisa-render-ext)

# work around the clang issue: https://github.com/Homebrew/homebrew-core/issues/169820
if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH CLANG_BIN_PATH)
    cmake_path(GET CLANG_BIN_PATH PARENT_PATH CLANG_ROOT_PATH)
    set(LIBCXX_PATH "${CLANG_ROOT_PATH}/lib/c++")
    if (EXISTS ${LIBCXX_PATH})
        target_link_libraries(luisa-compute-core PUBLIC "-L${LIBCXX_PATH}")
    endif ()
endif ()

set_target_properties(luisa-render-util PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ${LUISA_RENDER_ENABLE_UNITY_BUILD})
install(TARGETS luisa-render-util
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
