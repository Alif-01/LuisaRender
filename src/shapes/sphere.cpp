//
// Created by Mike Smith on 2022/11/8.
//

#include <future>

#include <util/thread_pool.h>
#include <base/shape.h>
#include <shapes/sphere_base.h>
#include <util/loop_subdiv.h>

namespace luisa::render {

class Sphere : public Shape {

private:
    std::shared_future<SphereGeometry> _geometry;

public:
    Sphere(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc},
          _geometry{SphereGeometry::create(
              std::min(desc->property_uint_or_default("subdivision", 0u),
                       sphere_max_subdivision_level))} {}
    
    // Sphere(Scene *scene, const RawSphereInfo &sphere_info) noexcept
    //     : Shape{scene, sphere_info.shape_info},
    //       _geometry{SphereGeometry::create(0u)} {}

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] MeshView mesh() const noexcept override { return _geometry.get().mesh(); }
    [[nodiscard]] uint vertex_properties() const noexcept override {
        return Shape::property_flag_has_vertex_normal |
               Shape::property_flag_has_vertex_uv;
    }
};

using SphereWrapper =
    VisibilityShapeWrapper<
        ShadowTerminatorShapeWrapper<
            IntersectionOffsetShapeWrapper<Sphere>>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SphereWrapper)

// LUISA_EXPORT_API luisa::render::SceneNode *create_raw(
//     luisa::render::Scene *scene,
//     const luisa::render::RawSphereInfo &sphere_info) LUISA_NOEXCEPT {
//     return luisa::new_with_allocator<luisa::render::SphereWrapper>(scene, sphere_info);
// }