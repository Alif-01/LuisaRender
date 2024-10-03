#include <future>

#include <base/shape.h>
#include <util/mesh_base.h>

namespace luisa::render {

class Plane : public Shape {

private:
    std::shared_future<PlaneGeometry> _geometry;

public:
    Plane(Scene *scene, const SceneNodeDesc *desc) noexcept:
        Shape{scene, desc},
        _geometry{PlaneGeometry::create(
            std::min(desc->property_uint_or_default("subdivision", 0u),
                     PlaneGeometry::max_subdivision_level))} {}

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] bool empty() const noexcept override { 
        const PlaneGeometry &g = _geometry.get();
        return g.vertices().empty() || g.triangles().empty();
    }
    [[nodiscard]] MeshView mesh() const noexcept override {
        const PlaneGeometry &g = _geometry.get();
        return { g.vertices(), g.triangles() };
    }
    [[nodiscard]] uint vertex_properties() const noexcept override {
        return Shape::property_flag_has_vertex_normal |
               Shape::property_flag_has_vertex_uv;
    }
};

using PlaneWrapper = VisibilityShapeWrapper<ShadingShapeWrapper<Plane>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PlaneWrapper)