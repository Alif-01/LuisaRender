//
// Created by Mike Smith on 2022/1/15.
//

#include <base/transform.h>

namespace luisa::render {

class MatrixTransform final : public Transform {

private:
    float4x4 _matrix;

private:
    void _build_matrix(const luisa::vector<float> &m) noexcept {
        if (m.size() == 16u) {
            luisa::vector<float> nm(m.begin(), m.end());
            if (!all(make_float4(m[12], m[13], m[14], m[15]) ==
                     make_float4(0.0f, 0.0f, 0.0f, 1.0f))) {
                LUISA_WARNING(
                    "Expected affine transform matrices, "
                    "while the last row is ({}, {}, {}, {}). "
                    "This will be fixed but might lead to "
                    "unexpected transforms", m[12], m[13], m[14], m[15]);
                nm[12] = 0.0f, nm[13] = 0.0f,
                nm[14] = 0.0f, nm[15] = 1.0f;
            }
            for (auto row = 0u; row < 4u; row++) {
                for (auto col = 0u; col < 4u; col++) {
                    _matrix[col][row] = nm[row * 4u + col];
                }
            }
        } else if (!m.empty()) [[unlikely]] {
            LUISA_ERROR("Invalid matrix entries.");
        }
    }

public:
    MatrixTransform(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Transform{scene, desc}, _matrix{make_float4x4(1.0f)} {
        auto m = desc->property_float_list_or_default("m");
        _build_matrix(m);
    }

    MatrixTransform(Scene *scene, const luisa::vector<float> &m) noexcept
        : Transform{scene}, _matrix{make_float4x4(1.0f)} {
        _build_matrix(m);
    }

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] float4x4 matrix(float) const noexcept override { return _matrix; }
    [[nodiscard]] bool is_static() const noexcept override { return true; }
    [[nodiscard]] bool is_identity() const noexcept override {
        return all(_matrix[0] == make_float4(1.0f, 0.0f, 0.0f, 0.0f)) &&
               all(_matrix[1] == make_float4(0.0f, 1.0f, 0.0f, 0.0f)) &&
               all(_matrix[2] == make_float4(0.0f, 0.0f, 1.0f, 0.0f)) &&
               all(_matrix[3] == make_float4(0.0f, 0.0f, 0.0f, 1.0f));
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MatrixTransform)

LUISA_EXPORT_API luisa::render::SceneNode *create_raw(
    luisa::render::Scene *scene, const luisa::vector<float> &m) LUISA_NOEXCEPT {
    return luisa::new_with_allocator<luisa::render::MatrixTransform>(scene, m);
}