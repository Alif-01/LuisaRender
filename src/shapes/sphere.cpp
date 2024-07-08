//
// Created by Mike Smith on 2022/11/8.
//

#include <future>

#include <base/shape.h>
#include <util/mesh_base.h>

namespace luisa::render {

class Sphere : public Shape {

private:
    std::shared_future<SphereGeometry> _geometry;

public:
    Sphere(Scene *scene, const SceneNodeDesc *desc) noexcept :
        Shape{scene, desc},
        _geometry{SphereGeometry::create(
            std::min(desc->property_uint_or_default("subdivision", 0u),
            SphereGeometry::max_subdivision_level))} {}

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] bool empty() const noexcept override { 
        const SphereGeometry &g = _geometry.get();
        return g.vertices().empty() || g.triangles().empty();
    }
    [[nodiscard]] MeshView mesh() const noexcept override {
        const SphereGeometry &g = _geometry.get();
        return { g.vertices(), g.triangles() };
    }
    [[nodiscard]] uint vertex_properties() const noexcept override {
        return Shape::property_flag_has_vertex_normal |
               Shape::property_flag_has_vertex_uv;
    }
};

using SphereWrapper = VisibilityShapeWrapper<ShadingShapeWrapper<Sphere>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SphereWrapper)