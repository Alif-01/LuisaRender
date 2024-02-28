//
// Created by Mike Smith on 2022/11/8.
//

#include <future>

#include <base/shape.h>
#include <shapes/sphere_base.h>
#include <util/loop_subdiv.h>
#include <util/mesh_reconstruct.h>

namespace luisa::render {

class SphereGroup : public Shape {

private:
    luisa::vector<Vertex> _vertices;
    luisa::vector<Triangle> _triangles;

private:
    void _build_mesh(
        const luisa::vector<float> &centers, float radius, uint subdiv,
        bool reconstruction, float voxel_scale, float smooth_scale
    ) noexcept {
        if (reconstruction) {
            static auto constructor = getConstructor();
            auto future = global_thread_pool().async([&] {
                return constructor->reconstruct(centers, radius, voxel_scale, smoothing_scale);
            });
            auto recon_mesh = future.get();
            _vertices = std::move(recon_mesh.vertices);
            _triangles = std::move(recon_mesh.triangles);
        } else {
            auto sphere_mesh = SphereGeometry::create(subdiv).get().mesh();
            uint32_t vertex_count = sphere_mesh.vertices.size();
            uint32_t triangle_count = sphere_mesh.triangles.size();
            if (centers.size() % 3u != 0u) {
                LUISA_ERROR_WITH_LOCATION("Invalid center count.");
            }
            uint32_t center_count = centers.size() / 3u;
            
            _vertices.clear();
            _triangles.clear();
            _vertices.reserve(center_count * vertex_count);
            _triangles.reserve(center_count * triangle_count);
            for (uint32_t i = 0u; i < center_count; ++i) {
                auto center = make_float3(
                    centers[i * 3u + 0u], centers[i * 3u + 1u], centers[i * 3u + 2u]
                );
                for (auto v: sphere_mesh.vertices) {
                    auto vertex = Vertex::encode(v.position() * radius + center, v.normal(), v.uv());
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
    }

public:
    SphereGroup(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc} {
        LUISA_NOT_IMPLEMENTED();
    }
    
    SphereGroup(Scene *scene, const RawShapeInfo &shape_info) noexcept
        : Shape{scene, shape_info} {

        if (shape_info.spheres_info == nullptr) [[unlikely]]
            LUISA_ERROR_WITH_LOCATION("Invalid spheres info!");
        auto spheres_info = shape_info.spheres_info.get();
        _build_mesh(
            spheres_info->centers, spheres_info->radius, spheres_info->subdivision,
            spheres_info->reconstruction, spheres_info->voxel_scale, spheres_info->smoothing_scale
        );
    }

    void update_shape(Scene *scene, const RawShapeInfo &shape_info) noexcept override {
        Shape::update_shape(scene, shape_info);
        
        if (shape_info.spheres_info == nullptr) [[unlikely]]
            LUISA_ERROR_WITH_LOCATION("Invalid spheres info!");
        auto spheres_info = shape_info.spheres_info.get();
        _build_mesh(
            spheres_info->centers, spheres_info->radius, spheres_info->subdivision,
            spheres_info->reconstruction, spheres_info->voxel_scale, spheres_info->smoothing_scale
        );
    }

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] MeshView mesh() const noexcept override { return MeshView{_vertices, _triangles}; }
    [[nodiscard]] uint vertex_properties() const noexcept override {
        return Shape::property_flag_has_vertex_normal |
               Shape::property_flag_has_vertex_uv;
    }
};

using SphereGroupWrapper = VisibilityShapeWrapper<ShadingShapeWrapper<SphereGroup>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SphereGroupWrapper)

LUISA_EXPORT_API luisa::render::SceneNode *create_raw(
    luisa::render::Scene *scene,
    const luisa::render::RawShapeInfo &shape_info) LUISA_NOEXCEPT {
    return luisa::new_with_allocator<luisa::render::SphereGroupWrapper>(scene, shape_info);
}