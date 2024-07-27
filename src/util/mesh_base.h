//
// Created by Mike Smith on 2022/11/8.
//
#pragma once

#include <future>
#include <luisa/core/stl.h>
#include <luisa/runtime/rtx/triangle.h>
#include <util/vertex.h>

namespace luisa::render {

class ShapeGeometry {

protected:
    luisa::vector<Vertex> _vertices;
    luisa::vector<Triangle> _triangles;

public:
    ShapeGeometry() noexcept = default;
    [[nodiscard]] const luisa::vector<Vertex> &vertices() const noexcept { return _vertices; }
    [[nodiscard]] const luisa::vector<Triangle> &triangles() const noexcept { return _triangles; }
    [[nodiscard]] virtual luisa::string info() const noexcept {
        return luisa::format(
            "num_vertices={} num_triangles={}",
            _vertices.size(), _triangles.size());
    }
};


class PlaneGeometry : public ShapeGeometry {

private:
    // Icosahedron
    static constexpr std::array base_points{
        make_float3(1.f, 1.f, 0.f),
        make_float3(-1.f, 1.f, 0.f),
        make_float3(-1.f, -1.f, 0.f),
        make_float3(1.f, -1.f, 0.f)
    };
    static constexpr std::array base_triangles{
        Triangle{0u, 1u, 2u},
        Triangle{0u, 2u, 3u}
    };

    [[nodiscard]] inline static float2 position_to_uv(const float3 &w) noexcept {
        return make_float2(.5f * (w.x + 1.f), .5f * (w.y + 1.f));
    };

public:
    static constexpr uint max_subdivision_level = 8u;

public:
    PlaneGeometry(uint subdiv) noexcept;
    [[nodiscard]] static std::shared_future<PlaneGeometry> create(uint subdiv) noexcept;
};


class SphereGeometry : public ShapeGeometry {

private:
    // Icosahedron
    static constexpr std::array base_points{
        make_float3(0.f, -0.525731f, 0.850651f),
        make_float3(0.850651f, 0.f, 0.525731f),
        make_float3(0.850651f, 0.f, -0.525731f),
        make_float3(-0.850651f, 0.f, -0.525731f),
        make_float3(-0.850651f, 0.f, 0.525731f),
        make_float3(-0.525731f, 0.850651f, 0.f),
        make_float3(0.525731f, 0.850651f, 0.f),
        make_float3(0.525731f, -0.850651f, 0.f),
        make_float3(-0.525731f, -0.850651f, 0.f),
        make_float3(0.f, -0.525731f, -0.850651f),
        make_float3(0.f, 0.525731f, -0.850651f),
        make_float3(0.f, 0.525731f, 0.850651f)
    };

    static constexpr std::array base_triangles{
        Triangle{1u, 2u, 6u},
        Triangle{1u, 7u, 2u},
        Triangle{3u, 4u, 5u},
        Triangle{4u, 3u, 8u},
        Triangle{6u, 5u, 11u},
        Triangle{5u, 6u, 10u},
        Triangle{9u, 10u, 2u},
        Triangle{10u, 9u, 3u},
        Triangle{7u, 8u, 9u},
        Triangle{8u, 7u, 0u},
        Triangle{11u, 0u, 1u},
        Triangle{0u, 11u, 4u},
        Triangle{6u, 2u, 10u},
        Triangle{1u, 6u, 11u},
        Triangle{3u, 5u, 10u},
        Triangle{5u, 4u, 11u},
        Triangle{2u, 7u, 9u},
        Triangle{7u, 1u, 0u},
        Triangle{3u, 9u, 8u},
        Triangle{4u, 8u, 0u}
    };

    [[nodiscard]] inline static float2 direction_to_uv(const float3 &w) noexcept {
        auto theta = acos(w.y);
        auto phi = atan2(w.x, w.z);
        return fract(make_float2(.5f * inv_pi * phi, theta * inv_pi));
    };

    [[nodiscard]] inline static float3 spherical_tangent(const float3 &w) noexcept {
        if (w.y > 1.f - 1e-8f) { return make_float3(1.f, 0.f, 0.f); }
        return normalize(make_float3(-w.z, 0.f, w.x));
    };

public:
    static constexpr uint max_subdivision_level = 8u;

public:
    SphereGeometry(uint subdiv) noexcept;
    [[nodiscard]] static std::shared_future<SphereGeometry> create(uint subdiv) noexcept;
};


class SpheresMeshGeometry : public ShapeGeometry {

private:
    int _num_spheres = 0u;

public:
    SpheresMeshGeometry(
        const luisa::vector<float> &centers,
        const luisa::vector<float> &radii, uint subdiv
    ) noexcept;
    [[nodiscard]] static std::shared_future<SpheresMeshGeometry> create(
        luisa::vector<float> centers,
        luisa::vector<float> radii, uint subdiv
    ) noexcept;
    [[nodiscard]] virtual luisa::string info() const noexcept override {
        return luisa::format("{} num_spheres={}", ShapeGeometry::info(), _num_spheres);
    }
};


class MeshGeometry : public ShapeGeometry {

private:
    bool _has_normal = false, _has_uv = false;

public:
    MeshGeometry(
        const luisa::vector<float> &positions,
        const luisa::vector<uint> &triangles,
        const luisa::vector<float> &normals,
        const luisa::vector<float> &uvs
    ) noexcept;
    [[nodiscard]] static std::shared_future<MeshGeometry> create(
        luisa::vector<float> positions,
        luisa::vector<uint> triangles,
        luisa::vector<float> normals,
        luisa::vector<float> uvs
    ) noexcept;

    MeshGeometry(
        std::filesystem::path path, uint subdiv,
        bool flip_uv, bool drop_normal, bool drop_uv
    ) noexcept;
    [[nodiscard]] static std::shared_future<MeshGeometry> create(
        std::filesystem::path path, uint subdiv,
        bool flip_uv, bool drop_normal, bool drop_uv
    ) noexcept;

    [[nodiscard]] virtual luisa::string info() const noexcept override {
        return luisa::format("{} has_normal={} has_uv={}", ShapeGeometry::info(), _has_normal, _has_uv);
    }
    [[nodiscard]] bool has_normal() const noexcept { return _has_normal; }
    [[nodiscard]] bool has_uv() const noexcept { return _has_uv; }
};

}// namespace luisa::render