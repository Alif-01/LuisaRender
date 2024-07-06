//
// Created by Mike Smith on 2022/1/26.
//
#pragma once

#include <base/texture.h>
#include <base/pipeline.h>
#include <base/scene.h>
#include <util/rng.h>

namespace luisa::render {

[[nodiscard]] static std::pair<float4, uint> build_constant(
    const luisa::vector<float> &v, float scale = 1.f
) noexcept {
    luisa::vector<float> c(v);
    if (c.empty()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No value for ConstantTexture. "
            "Fallback to single-channel zero.");
        c.emplace_back(0.f);
    } else if (c.size() > 4u) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Too many values (count = {}) for ConstantTexture. "
            "Additional values will be discarded.", c.size());
        c.resize(4u);
    }
    float4 fv;
    for (auto i = 0u; i < c.size(); ++i) 
        fv[i] = scale * c[i];
    return std::make_pair(fv, c.size());
}

}// namespace luisa::render