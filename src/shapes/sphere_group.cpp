//
// Created by Mike Smith on 2022/11/8.
//

#include <future>

#include <base/shape.h>
#include <util/mesh_base.h>
#include <util/loop_subdiv.h>

namespace luisa::render {

class SphereGroup : public Shape {

private:
    std::shared_future<SphereGroupGeometry> _geometry;
    uint _subdiv;

public:
    SphereGroup(Scene *scene, const SceneNodeDesc *desc) noexcept :
        Shape{scene, desc},
        _subdiv{desc->property_uint_or_default("subdivision", 0u)} { 
        _geometry = SphereGroupGeometry::create(
            desc->property_float_list("centers"),
            desc->property_float_list("radii"), _subdiv
        );
    }

    [[nodiscard]] bool update(Scene *scene, const SceneNodeDesc *desc) noexcept override {
        _geometry = SphereGroupGeometry::create(
            desc->property_float_list("centers"),
            desc->property_float_list("radii"), _subdiv
        );
        return true;
    }

    [[nodiscard]] luisa::string info() const noexcept override {
        return luisa::format("{} geometry=[{}]", Shape::info(), _geometry.get().info());
    }

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] bool empty() const noexcept override { 
        const SphereGroupGeometry &g = _geometry.get();
        return g.vertices().empty() || g.triangles().empty();
    }
    [[nodiscard]] MeshView mesh() const noexcept override {
        const SphereGroupGeometry &g = _geometry.get();
        return { g.vertices(), g.triangles() };
    }
    [[nodiscard]] uint vertex_properties() const noexcept override { 
        return Shape::property_flag_has_vertex_normal |
               Shape::property_flag_has_vertex_uv;
    }
};

using SphereGroupWrapper = VisibilityShapeWrapper<ShadingShapeWrapper<SphereGroup>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SphereGroupWrapper)