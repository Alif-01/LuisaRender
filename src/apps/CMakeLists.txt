function(luisa_render_add_application name)
    cmake_parse_arguments(APP "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${APP_SOURCES})
    target_link_libraries(${name} PRIVATE luisa::render)
    message("${name} created!")
endfunction()

luisa_render_add_application(luisa-render-cli SOURCES cli.cpp)
luisa_render_add_application(luisa-render-export SOURCES export.cpp)
luisa_render_add_application(luisa-render-pipe SOURCES pipe.cpp)
luisa_render_add_application(luisa-render-pipe-render SOURCES pipe_render.cpp)

find_package(pybind11 REQUIRED)

pybind11_add_module(LuisaRenderPy py_interface.cpp)
target_link_libraries(LuisaRenderPy PRIVATE luisa::render)
