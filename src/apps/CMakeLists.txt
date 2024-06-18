function(luisa_render_add_application name)
    cmake_parse_arguments(APP "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${APP_SOURCES})
    target_link_libraries(${name} PRIVATE luisa::render)
    install(TARGETS ${name}
            LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

luisa_render_add_application(luisa-render-pipe-render SOURCES pipe_render.cpp)

find_package(pybind11 REQUIRED)

if(DEFINED PYTHON_VERSIONS)
    foreach(version ${PYTHON_VERSIONS})
        set(PYBIND11_PYTHON_VERSION ${version})
        find_package(Python ${version} EXACT REQUIRED COMPONENTS Interpreter Development)
        pybind11_add_module(LuisaRenderPy py_interface.cpp)
        target_link_libraries(LuisaRenderPy PRIVATE luisa::render)
    endforeach()
else()
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
    message("Default Python ${Python_VERSION} in environment was found.")
    pybind11_add_module(LuisaRenderPy py_interface.cpp)
    target_link_libraries(LuisaRenderPy PRIVATE luisa::render)
endif()