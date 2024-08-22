add_library(luisa-render-films INTERFACE)
luisa_render_add_plugin(color CATEGORY film SOURCES color.cpp)

if (LUISA_COMPUTE_ENABLE_GUI)
    luisa_render_add_plugin(display CATEGORY film SOURCES display.cpp)
endif ()