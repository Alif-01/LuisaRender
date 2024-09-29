//
// Created by Mike on 2022/1/7.
//

#include <base/shape.h>
#include <util/mesh_base.h>

namespace luisa::render {

class Mesh : public Shape {

private:
    std::shared_future<MeshGeometry> _geometry;

public:
    Mesh(Scene *scene, const SceneNodeDesc *desc) noexcept:
        Shape{scene, desc} {

        if (desc->property_string_or_default("file").empty()) {
            _geometry = MeshGeometry::create(
                desc->property_float_list("positions"),
                desc->property_uint_list("indices"),
                desc->property_float_list_or_default("normals"),
                desc->property_float_list_or_default("uvs"));
        } else {
            _geometry = MeshGeometry::create(
                desc->property_path("file"),
                desc->property_uint_or_default("subdivision", 0u),
                desc->property_bool_or_default("flip_uv", false),
                desc->property_bool_or_default("drop_normal", false),
                desc->property_bool_or_default("drop_uv", false));
        }
    }

    void update(Scene *scene, const SceneNodeDesc *desc) noexcept override {
        Shape::update(scene, desc);
    }

    [[nodiscard]] luisa::string info() const noexcept override {
        return luisa::format("{} geometry=[{}]", Shape::info(), _geometry.get().info());
    }
    
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] bool empty() const noexcept override { 
        const MeshGeometry &g = _geometry.get();
        return g.vertices().empty() || g.triangles().empty();
    }
    [[nodiscard]] MeshView mesh() const noexcept override {
        const MeshGeometry &g = _geometry.get();
        return { g.vertices(), g.triangles() }; 
    }
    [[nodiscard]] uint vertex_properties() const noexcept override {
        const MeshGeometry &g = _geometry.get();
        return (g.has_normal() ? Shape::property_flag_has_vertex_normal : 0u) | 
               (g.has_uv() ? Shape::property_flag_has_vertex_uv : 0u);
    }
};

using MeshWrapper = VisibilityShapeWrapper<ShadingShapeWrapper<Mesh>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MeshWrapper)
