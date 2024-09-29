//
// Created by Mike Smith on 2022/1/15.
//

#include <base/transform.h>
#include <base/shape.h>

namespace luisa::render {

class MatrixTransform final : public Transform {

private:
    float4x4 _matrix;

private:
    void _build_matrix(luisa::vector<float> m) noexcept {
        if (m.size() == 16u) {
            if (!all(make_float4(m[12], m[13], m[14], m[15]) ==
                     make_float4(0.0f, 0.0f, 0.0f, 1.0f))) [[unlikely]] {
                LUISA_WARNING(
                    "Expected affine transform matrices, "
                    "while the last row is ({}, {}, {}, {}). "
                    "This will be fixed but might lead to "
                    "unexpected transforms", m[12], m[13], m[14], m[15]);
                m[12] = 0.0f, m[13] = 0.0f, m[14] = 0.0f, m[15] = 1.0f;
            }
            for (auto row = 0u; row < 4u; row++) {
                for (auto col = 0u; col < 4u; col++) {
                    _matrix[col][row] = m[row * 4u + col];     // read matrix.T
                }
            }
        } else if (!m.empty()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Invalid matrix entries: {}.", m.size());
        }
    }

public:
    MatrixTransform(Scene *scene, const SceneNodeDesc *desc) noexcept:
        Transform{scene, desc}, _matrix{make_float4x4(1.0f)} {
        _build_matrix(desc->property_float_list_or_default("m"));
    }

    void update(Scene *scene, const SceneNodeDesc *desc) noexcept override {
        _build_matrix(desc->property_float_list_or_default("m"));
        set_updated(true);
    }
    [[nodiscard]] luisa::string info() const noexcept override {
        return luisa::format("{} matrix=[{}]", Transform::info(), _matrix);
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