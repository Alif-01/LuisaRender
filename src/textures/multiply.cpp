//
// Created by mike on 4/17/24.
//

#include <base/texture.h>
#include <base/scene.h>
#include <base/pipeline.h>

namespace luisa::render {

class MultiplyTexture final : public Texture {

private:
    Texture *_a;
    Texture *_b;

public:
    MultiplyTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc},
          _a{scene->load_texture(desc->property_node("a"))},
          _b{scene->load_texture(desc->property_node("b"))} {}

    [[nodiscard]] auto a() const noexcept { return _a; }
    [[nodiscard]] auto b() const noexcept { return _b; }
    [[nodiscard]] bool is_black() const noexcept override { return _a->is_black() || _b->is_black(); }
    [[nodiscard]] bool is_constant() const noexcept override { return is_black() || (_a->is_constant() && _b->is_constant()); }
    [[nodiscard]] luisa::optional<float4> evaluate_static() const noexcept override {
        if (auto a = _a->evaluate_static(),
            b = _b->evaluate_static();
            a && b) {
            if (_a->channels() == _b->channels()) {
                return a.value() * b.value();
            }
            if (_a->channels() == 1u) {
                a = make_float4(a.value().x);
            } else if (_b->channels() == 1u) {
                b = make_float4(b.value().x);
            } else if (_a->channels() < _b->channels()) {
                for (auto i = _a->channels(); i < _b->channels(); i++) {
                    a.value()[i] = 1.f;
                }
            } else {
                for (auto i = _b->channels(); i < _a->channels(); i++) {
                    b.value()[i] = 1.f;
                }
            }
            return a.value() * b.value();
        }
        return nullopt;
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint2 resolution() const noexcept override { return max(_a->resolution(), _b->resolution()); }
    [[nodiscard]] uint channels() const noexcept override { return std::max(_a->channels(), _b->channels()); }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MultiplyTextureInstance final : public Texture::Instance {

private:
    const Texture::Instance *_a;
    const Texture::Instance *_b;

public:
    MultiplyTextureInstance(Pipeline &pipeline,
                            const Texture *node,
                            const Texture::Instance *a,
                            const Texture::Instance *b) noexcept:
        Texture::Instance{pipeline, node}, _a{a}, _b{b} {}
    [[nodiscard]] Float4 evaluate(const Interaction &it,
                                  Expr<float> time) const noexcept override {
        auto a = _a->evaluate(it, time);
        auto b = _b->evaluate(it, time);
        if (_a->node()->channels() == _b->node()->channels()) {
            return a * b;
        }
        if (_a->node()->channels() == 1u) {
            a = make_float4(a.x);
        } else if (_b->node()->channels() == 1u) {
            b = make_float4(b.x);
        } else if (_a->node()->channels() < _b->node()->channels()) {
            for (auto i = _a->node()->channels(); i < _b->node()->channels(); i++) {
                a[i] = 1.f;
            }
        } else {
            for (auto i = _b->node()->channels(); i < _a->node()->channels(); i++) {
                b[i] = 1.f;
            }
        }
        return a * b;
    }
};

luisa::unique_ptr<Texture::Instance> MultiplyTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto a = pipeline.build_texture(command_buffer, _a);
    auto b = pipeline.build_texture(command_buffer, _b);
    return luisa::make_unique<MultiplyTextureInstance>(pipeline, this, a, b);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MultiplyTexture)
