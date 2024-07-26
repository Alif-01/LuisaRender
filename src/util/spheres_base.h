//
// Created by Mike Smith on 2022/11/8.
//
#pragma once

#include <future>
#include <luisa/core/stl.h>
#include <luisa/runtime/rtx/aabb.h>

namespace luisa::render {

class ProceduralGeometry {

protected:
    luisa::vector<AABB> _aabbs;

public:
    ProceduralGeometry() noexcept = default;
    [[nodiscard]] const luisa::vector<Vertex> &aabbs() const noexcept { return _aabbs; }
    [[nodiscard]] virtual luisa::string info() const noexcept {
        return luisa::format("num_aabbs={}", _aabbs.size());
    }
};

class SphereGroupGeometry : public ProceduralGeometry {

private:
    int _num_spheres = 0u;

public:
    SphereGroupGeometry(
        const luisa::vector<float> &centers,
        const luisa::vector<float> &radii
    ) noexcept;
    [[nodiscard]] static std::shared_future<SphereGroupGeometry> create(
        luisa::vector<float> centers,
        luisa::vector<float> radii
    ) noexcept;
    [[nodiscard]] virtual luisa::string info() const noexcept override {
        return luisa::format("{} num_spheres={}", ProceduralGeometry::info(), _num_spheres);
    }
};

}