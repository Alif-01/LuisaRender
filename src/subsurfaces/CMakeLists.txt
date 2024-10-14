add_library(luisa-render-subsurfaces INTERFACE)
luisa_render_add_plugin(null CATEGORY subsurface SOURCES null.cpp)
luisa_render_add_plugin(uniform CATEGORY subsurface SOURCES uniform.cpp)
