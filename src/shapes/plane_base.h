//
// Created by Mike Smith on 2022/11/8.
//
#pragma once

#include <future>
#include <base/shape.h>
#include <util/loop_subdiv.h>
#include <util/thread_pool.h>

namespace luisa::render {

static constexpr auto plane_max_subdivision_level = 8u;

// Icosahedron
static constexpr std::array plane_base_vertices{
    make_float3(1.f, 1.f, 0.f),
    make_float3(-1.f, 1.f, 0.f),
    make_float3(-1.f, -1.f, 0.f),
    make_float3(1.f, -1.f, 0.f)
};

static constexpr std::array plane_base_triangles{
    Triangle{0u, 1u, 2u},
    Triangle{0u, 2u, 3u}
};

class PlaneGeometry {

private:
    luisa::vector<Vertex> _vertices;
    luisa::vector<Triangle> _triangles;

public:
    PlaneGeometry() noexcept = default;
    PlaneGeometry(luisa::vector<Vertex> vertices,
                  luisa::vector<Triangle> triangles) noexcept
        : _vertices{std::move(vertices)},
          _triangles{std::move(triangles)} {}

public:
    [[nodiscard]] auto mesh() const noexcept { return MeshView{_vertices, _triangles}; }
    [[nodiscard]] static auto create(uint subdiv) noexcept {
        LUISA_ASSERT(subdiv <= plane_max_subdivision_level, "Subdivision level {} is too high.", subdiv);
        static constexpr auto position_to_uv = [](float3 w) noexcept {
            return make_float2(.5f * (w.x + 1.f), .5f * (w.y + 1.f));
        };
        static auto base_vertices = [] {
            std::array<Vertex, plane_base_vertices.size()> bv{};
            for (auto i = 0u; i < plane_base_vertices.size(); i++) {
                auto p = plane_base_vertices[i];
                bv[i] = Vertex::encode(p, make_float3(0.f, 0.f, 1.f), make_float2());
            }
            return bv;
        }();

        static std::array<std::shared_future<PlaneGeometry>, plane_max_subdivision_level + 1u> cache;
        static std::mutex mutex;
        std::scoped_lock lock{mutex};
        if (auto g = cache.at(subdiv); g.valid()) { return g; }

        auto future = global_thread_pool().async([subdiv] {
            auto [vertices, triangles, _] = loop_subdivide(base_vertices, plane_base_triangles, subdiv);
            for (auto &v : vertices) {
                auto p = v.position();
                auto uv = position_to_uv(p);
                v = Vertex::encode(p, make_float3(0.f, 0.f, 1.f), uv);
            }
            return PlaneGeometry{std::move(vertices), std::move(triangles)};
        });
        cache[subdiv] = future;
        return future;
    }
};

}