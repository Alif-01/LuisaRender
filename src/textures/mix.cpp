#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

class MixTexture final : public Texture {

public:
    enum struct MixMethod {
        ADD,
        SUBSTRACT,
        MULTIPLY,
        MIX
    };

private:
    Texture *_top;
    Texture *_bottom;
    float _factor;
    MixMethod _method;

public:
    MixTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc},
          _top{scene->load_texture(desc->property_node_or_default("top"))},
          _bottom{scene->load_texture(desc->property_node_or_default("bottom"))},
          _factor{desc->property_float_or_default("factor", 0.5f)} {
        auto method = desc->property_string_or_default("method", "mix");
        for (auto &c : method) { c = static_cast<char>(tolower(c)); }
        if (method == "mix") {
            _method = MixMethod::MIX;
        } else if (method == "add") {
            _method = MixMethod::ADD;
        } else if (method == "subtract") {
            _method = MixMethod::SUBSTRACT;
        } else if (method == "multiply") {
            _method = MixMethod::MULTIPLY;
        } else {
            LUISA_WARNING_WITH_LOCATION("Unknown mix method '{}'. Fallback to 'mix'", method);
            _method = MixMethod::MIX;
        }
    }
    // is_black() and is_constant is not precise
    [[nodiscard]] bool is_black() const noexcept override {    
        auto top_is_black = _top != nullptr && _top->is_black();    // both default to all white
        auto bottom_is_black = _bottom != nullptr && _bottom->is_black();
        return top_is_black && bottom_is_black;
    }
    [[nodiscard]] bool is_constant() const noexcept override {
        auto top_is_constant = _top == nullptr || _top->is_constant();
        auto bottom_is_constant = _bottom == nullptr || _bottom->is_constant();
        return top_is_constant && bottom_is_constant;
    }
    [[nodiscard]] uint2 resolution() const noexcept override { return max(_top->resolution(), _bottom->resolution()); }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override {
        auto top_channels = _top == nullptr ? 4u : _top->channels();
        auto bottom_channels = _bottom == nullptr ? 4u : _bottom->channels();
        if (top_channels != bottom_channels) {
            LUISA_WARNING_WITH_LOCATION(
                "MixTexture: top and bottom textures have different channel counts ({} vs {}).",
                top_channels, bottom_channels);
        }
        return std::min(top_channels, bottom_channels);
    }
    [[nodiscard]] auto factor() const noexcept { return _factor; }
    [[nodiscard]] auto method() const noexcept { return _method; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MixTextureInstance final : public Texture::Instance {

private:
    const Texture::Instance *_top;
    const Texture::Instance *_bottom;

public:
    MixTextureInstance(Pipeline &pipeline, const Texture *node,
                       const Texture::Instance *top, const Texture::Instance *bottom) noexcept:
        Texture::Instance{pipeline, node}, _top{top}, _bottom{bottom} {}
    [[nodiscard]] Float4 evaluate(const Interaction &it,
                                  Expr<float> time) const noexcept override {
        auto value = def(make_float4());
        auto top_value = _top == nullptr ? make_float4(1.f) : _top->evaluate(it, time);
        auto bottom_value = _bottom == nullptr ? make_float4(1.f) : _bottom->evaluate(it, time);
        auto method = node<MixTexture>()->method();
        auto factor = node<MixTexture>()->factor();
        if (method == MixTexture::MixMethod::MIX) {
            value = top_value * (1 - factor) + bottom_value * factor;
        } else if (method == MixTexture::MixMethod::ADD) {
            value = top_value * (1 - factor) + (bottom_value + top_value) * factor;
        } else if (method == MixTexture::MixMethod::SUBSTRACT) {
            value = top_value * (1 - factor) + (bottom_value - top_value) * factor;
        } else if (method == MixTexture::MixMethod::MULTIPLY) {
            value = top_value * (1 - factor) + (bottom_value * top_value) * factor;
        }
        return value;
    }
};

luisa::unique_ptr<Texture::Instance> MixTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto top = pipeline.build_texture(command_buffer, _top);
    auto bottom = pipeline.build_texture(command_buffer, _bottom);
    return luisa::make_unique<MixTextureInstance>(pipeline, this, top, bottom);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MixTexture)
