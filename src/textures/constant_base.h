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
    const luisa::vector<float> &ov, float scale = 1.f
) noexcept {
    luisa::vector<float> v(ov.begin(), ov.end());
    if (v.empty()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("No value for ConstantTexture!");
    } else if (v.size() > 4u) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Too many values (count = {}) for ConstantTexture. "
            "Additional values will be discarded.", v.size());
        v.resize(4u);
    }
    float4 fv;
    for (auto i = 0u; i < v.size(); ++i) 
        fv[i] = scale * v[i];
}

}// namespace luisa::render