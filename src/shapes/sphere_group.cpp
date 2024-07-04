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

private:
    // void _build_mesh(
    //     const luisa::vector<float> &centers,
    //     float radius, uint subdiv
    // ) noexcept {
    //     static std::mutex mutex;
    //     std::scoped_lock lock{mutex};
    //     _geometry = SphereGroupGeometry::create(centers, radius, subdiv);
    //     _geometry.wait();
    // }

public:
    SphereGroup(Scene *scene, const SceneNodeDesc *desc) noexcept :
        Shape{scene, desc},
        _subdiv{desc->property_uint_or_default("subdivision", 0u)},
        _geometry(SphereGroupGeometry::create(
            desc->property_float_list("centers"),
            desc->property_float_list("radii"), _subdiv
        )) { _geometry.wait(); }
    
    // SphereGroup(Scene *scene, const RawShapeInfo &shape_info) noexcept :
    //     Shape{scene, shape_info} {
    //     LUISA_ASSERT(shape_info.spheres_info != nullptr, "Invalid spheres info.");
    //     auto spheres_info = shape_info.spheres_info.get();
    //     _build_mesh(
    //         spheres_info->centers, spheres_info->radius, spheres_info->subdivision
    //     );
    // }

    [[nodiscard]] bool update(Scene *scene, const SceneNodeDesc *desc) noexcept override {
        _geometry = SphereGroupGeometry::create(
            desc->property_float_list("centers"),
            desc->property_float_list("radii"), _subdiv
        );
        _geometry.wait();
        return true;
    }

    // void update_shape(Scene *scene, const RawShapeInfo &shape_info) noexcept override {
    //     Shape::update_shape(scene, shape_info);
    //     LUISA_ASSERT(shape_info.spheres_info != nullptr, "Invalid spheres info.");
    //     auto spheres_info = shape_info.spheres_info.get();
    //     _build_mesh(
    //         spheres_info->centers, spheres_info->radius, spheres_info->subdivision
    //     );
    // }

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

// LUISA_EXPORT_API luisa::render::SceneNode *create_raw(
//     luisa::render::Scene *scene,
//     const luisa::render::RawShapeInfo &shape_info) LUISA_NOEXCEPT {
//     return luisa::new_with_allocator<luisa::render::SphereGroupWrapper>(scene, shape_info);
// }