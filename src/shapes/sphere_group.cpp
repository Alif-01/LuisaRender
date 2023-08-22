//
// Created by Mike Smith on 2022/11/8.
//

#include <future>

#include <util/thread_pool.h>
#include <util/loop_subdiv.h>
#include <base/shape.h>
#include <shapes/sphere_base.h>

namespace luisa::render {

class SphereGroup : public Shape {

private:
    luisa::vector<Vertex> _vertices;
    luisa::vector<Triangle> _triangles;

private:
    void _build_mesh(const luisa::vector<uint> &centers, uint subdiv) noexcept {
        auto sphere_mesh = SphereGeometry::create(subdiv).get().mesh();
        auto vertex_count = sphere_mesh.vertices.size();
        auto triangle_count = sphere_mesh.triangles.size();
        if (centers.size() % 3u != 0u) {
            LUISA_ERROR_WITH_LOCATION("Invalid center count.");
        }
        auto center_count = centers.size() / 3u;
        
        _vertices.clear();
        _triangles.clear();
        _vertices.reserve(center_count * vertex_count);
        _triangles.reserve(center_count * triangle_count);
        for (auto i = 0u; i < center_count; ++i) {
            auto center = make_float3(
                centers[i * 3u + 0u], centers[i * 3u + 1u], centers[i * 3u + 2u]
            );
            for (auto v: sphere_mesh.vertices) {
                auto vertex = Vertex::encode(p.position() + center, p.normal(), p.uv);
                _vertices.emplace_back(std::move(vertex));
            }
            for (auto t: sphere_mesh.triangles) {
                auto triangle = Triangle{
                    t.i0 + vertex_count * i,
                    t.i1 + vertex_count * i,
                    t.i2 + vertex_count * i
                };
                _triangles.emplace_back(std::move(triangle));
            }
        }
    }

public:
    // SphereGroup(Scene *scene, const SceneNodeDesc *desc) noexcept
    //     : Shape{scene, desc},
    //       _geometry{SphereGeometry::create(
    //           std::min(desc->property_uint_or_default("subdivision", 0u),
    //                    sphere_max_subdivision_level))} {}
    
    SphereGroup(Scene *scene, const RawShapeInfo &shape_info) noexcept
        : Shape{scene, shape_info} {

        if (shape_info.spheres_info == nullptr) [[unlikely]]
            LUISA_ERROR_WITH_LOCATION("Invalid spheres info!");
        auto sphere_info = shape_info.spheres_info.get();
        _build_mesh(spheres_info->centers, spheres_info->subdiv);
    }

    void update_shape(Scene *scene, const RawShapeInfo &shape_info) noexcept override {
        Shape::update_shape(scene, shape_info);
        if (shape_info.spheres_info == nullptr) [[unlikely]]
            LUISA_ERROR_WITH_LOCATION("Invalid spheres info!");
        auto sphere_info = shape_info.spheres_info.get();

        if (!spheres_info->centers.empty())
            _build_mesh(spheres_info->centers, spheres_info->subdiv);
    }

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] MeshView mesh() const noexcept override { return MeshView{_vertices, _triangles}; }
    [[nodiscard]] uint vertex_properties() const noexcept override {
        return Shape::property_flag_has_vertex_normal |
               Shape::property_flag_has_vertex_uv;
    }
};

using SphereGroupWrapper =
    VisibilityShapeWrapper<
        ShadowTerminatorShapeWrapper<
            IntersectionOffsetShapeWrapper<SphereGroup>>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SphereGroupWrapper)

LUISA_EXPORT_API luisa::render::SceneNode *create_raw(
    luisa::render::Scene *scene,
    const luisa::render::RawShapeInfo &shape_info) LUISA_NOEXCEPT {
    return luisa::new_with_allocator<luisa::render::SphereGroupWrapper>(scene, shape_info);
}