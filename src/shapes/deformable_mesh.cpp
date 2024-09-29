//
// Created by Mike on 2022/2/18.
//

#include <base/shape.h>
#include <util/mesh_base.h>

namespace luisa::render {

class DeformableMesh : public Shape {

private:
    std::shared_future<MeshGeometry> _geometry;

public:
    DeformableMesh(Scene *scene, const SceneNodeDesc *desc) noexcept :
        Shape{scene, desc},
        _geometry{MeshGeometry::create(
            desc->property_float_list_or_default("positions"),
            desc->property_uint_list_or_default("indices"),
            desc->property_float_list_or_default("normals"),
            desc->property_float_list_or_default("uvs")
        )} { }

    void update(Scene *scene, const SceneNodeDesc *desc) noexcept override {
        _geometry = MeshGeometry::create(
            desc->property_float_list_or_default("positions"),
            desc->property_uint_list_or_default("indices"),
            desc->property_float_list_or_default("normals"),
            desc->property_float_list_or_default("uvs")
        );
        set_updated(true);
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

using DeformableMeshWrapper = VisibilityShapeWrapper<ShadingShapeWrapper<DeformableMesh>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DeformableMeshWrapper)