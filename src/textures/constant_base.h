//
// Created by Mike Smith on 2022/1/26.
//
#pragma once

#include <base/texture.h>
#include <base/pipeline.h>
#include <base/scene.h>
#include <util/rng.h>

namespace luisa::render {

[[nodiscard]] static float4 build_constant(
    luisa::vector<float> &v, float scale = 1.f
) noexcept {
    if (v.empty()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No value for ConstantTexture. "
            "Fallback to single-channel zero.");
        v.emplace_back(0.f);
    } else if (v.size() > 4u) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Too many values (count = {}) for ConstantTexture. "
            "Additional values will be discarded.", v.size());
        v.resize(4u);
    }
    float4 fv;
    for (auto i = 0u; i < v.size(); ++i) 
        fv[i] = scale * v[i];
    return fv;
}

}// namespace luisa::render