//
// Created by mike on 4/17/24.
//

#include <base/texture.h>
#include <base/scene.h>
#include <base/pipeline.h>

namespace luisa::render {

class ScaleTexture final : public Texture {

private:
    Texture *_base;
    float4 _scale;
    float4 _offset;

public:
    ScaleTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc},
          _base{scene->load_texture(desc->property_node("base"))},
          _scale{[desc] {
              auto s = desc->property_float_list_or_default("scale");
              if (s.size() == 1u) { return make_float4(s[0]); }
              s.reserve(4u);
              if (s.size() < 4u) { s.resize(4u, 1.f); }
              s.resize(4u);
              return make_float4(s[0], s[1], s[2], s[3]);
          }()},
          _offset{[desc] {
              auto o = desc->property_float_list_or_default("offset");
              if (o.size() == 1u) { return make_float4(o[0]); }
              o.reserve(4u);
              if (o.size() < 4u) { o.resize(4u, 0.f); }
              o.resize(4u);
              return make_float4(o[0], o[1], o[2], o[3]);
          }()} {}
    [[nodiscard]] auto base() const noexcept { return _base; }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] bool is_constant() const noexcept override { return _base->is_constant(); }
    [[nodiscard]] luisa::optional<float4> evaluate_static() const noexcept override {
        if (auto v = _base->evaluate_static()) {
            return v.value() * _scale;
        }
        return nullopt;
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override { return _base->channels(); }
    [[nodiscard]] uint2 resolution() const noexcept override { return _base->resolution(); }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class ScaleTextureInstance final : public Texture::Instance {

private:
    const Texture::Instance *_base;

public:
    ScaleTextureInstance(Pipeline &pipeline, const Texture *node,
                         const Texture::Instance *base) noexcept:
        Texture::Instance{pipeline, node}, _base{base} {}
    [[nodiscard]] Float4 evaluate(const Interaction &it,
                                  Expr<float> time) const noexcept override {
        return _base->evaluate(it, time) * node<ScaleTexture>()->scale() + node<ScaleTexture>()->offset();
    }
};

luisa::unique_ptr<Texture::Instance> ScaleTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto base = pipeline.build_texture(command_buffer, _base);
    return luisa::make_unique<ScaleTextureInstance>(pipeline, this, base);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ScaleTexture)
