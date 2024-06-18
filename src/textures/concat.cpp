//
// Created by ChenXin on 2024/4/17.
//

#include <base/texture.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

class ConcatTexture final : public Texture {

private:
    luisa::vector<Texture *> _sub_textures;
    uint _channels{0u};
    uint _last_channel_size{-1u};
    bool _is_constant{true};
    bool _is_black{true};
    bool _evaluate_static{true};

public:
    ConcatTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc} {
        // Load all the channels
        auto channels = desc->property_node_list("channels");
        for (auto channel: channels) {
            auto texture = scene->load_texture(channel);
            _channels = _channels + texture->channels();
            _sub_textures.push_back(texture);
        }
        // Discard extra channels
        if (_channels > 4u) {
            LUISA_WARNING_WITH_LOCATION(
                "Too many channels (count = {}) for ConcatTexture. "
                "Additional channels will be discarded. [{}]",
                _channels, desc->source_location().string());

            while (_channels - _sub_textures.back()->channels() >= 4u) {
                _channels = _channels - _sub_textures.back()->channels();
                _sub_textures.pop_back();
            }
            _last_channel_size = _sub_textures.back()->channels() - (_channels - 4u);
            _channels = 4u;
        } else {
            _last_channel_size = _sub_textures.back()->channels();
        }
        // Check if the texture is constant, black or static
        for (auto texture : _sub_textures) {
            _is_constant = _is_constant && texture->is_constant();
            _is_black = _is_black && texture->is_black();
            _evaluate_static = _evaluate_static && texture->evaluate_static().has_value();
        }
    }
    [[nodiscard]] const auto &sub_textures() const noexcept { return _sub_textures; }
    [[nodiscard]] auto last_channel_size() const noexcept { return _last_channel_size; }
    [[nodiscard]] bool is_constant() const noexcept override { return _is_constant; }
    [[nodiscard]] bool is_black() const noexcept override { return _is_black; }
    [[nodiscard]] luisa::optional<float4> evaluate_static() const noexcept override {
        if (_evaluate_static) {
            auto s = make_float4(0.f);
            auto index = 0u;
            for (auto i = 0u; i < _sub_textures.size() - 1; i++) {
                auto sub_texture = _sub_textures[i];
                auto v = sub_texture->evaluate_static().value();
                switch (auto n = sub_texture->channels(); n) {
                    case 1u: s[index++] = v.x; break;
                    case 2u: s[index] = v.x; s[index + 1u] = v.y; index += 2u; break;
                    case 3u: s[index] = v.x; s[index + 1u] = v.y; s[index + 2u] = v.z; index += 3u; break;
                    default: LUISA_ERROR_WITH_LOCATION("Unreachable");
                }
            }
            {
                auto last_sub_texture = _sub_textures.back();
                auto v = last_sub_texture->evaluate_static().value();
                switch (auto n = last_sub_texture->channels(); n) {
                    case 1u: s[index++] = v.x; break;
                    case 2u: s[index] = v.x; s[index + 1u] = v.y; index += 2u; break;
                    case 3u: s[index] = v.x; s[index + 1u] = v.y; s[index + 2u] = v.z; index += 3u; break;
                    case 4u: s[index] = v.x; s[index + 1u] = v.y; s[index + 2u] = v.z; s[index + 3u] = v.w; index += 4u; break;
                    default: LUISA_ERROR_WITH_LOCATION("Unreachable");
                }
            }
            LUISA_ASSERT(index == _channels, "Invalid channel count for ConcatTexture.");
            return s;
        }
        return nullopt;
    }
    [[nodiscard]] uint2 resolution() const noexcept override {
        auto res = make_uint2(1u);
        for (auto texture : _sub_textures) {
            res = max(res, texture->resolution());
        }
        return res;
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override { return _channels; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;

};

class ConcatTextureInstance final : public Texture::Instance {

private:
    luisa::vector<const Texture::Instance *> _sub_textures;

public:
    ConcatTextureInstance(Pipeline &pipeline,
                          const ConcatTexture *texture,
                          luisa::vector<const Texture::Instance *> &&sub_textures) noexcept
        : Texture::Instance{pipeline, texture},
          _sub_textures{sub_textures} {}
    [[nodiscard]] Float4 evaluate(const Interaction &it,
                                  Expr<float> time) const noexcept override {
        auto s = def(make_float4(0.f));
        auto index = 0u;
        for (auto i = 0u; i < _sub_textures.size() - 1; i++) {
            auto sub_texture = _sub_textures[i];
            auto v = sub_texture->evaluate(it, time);
            switch (auto n = sub_texture->node(); n->channels()) {
                case 1u: s[index++] = v.x; break;
                case 2u: s[index] = v.x; s[index + 1u] = v.y; index += 2u; break;
                case 3u: s[index] = v.x; s[index + 1u] = v.y; s[index + 2u] = v.z; index += 3u; break;
                default: LUISA_ERROR_WITH_LOCATION("Unreachable");
            }
        }
        {
            auto last_sub_texture = _sub_textures.back();
            auto v = last_sub_texture->evaluate(it, time);
            switch (auto n = node<ConcatTexture>(); n->last_channel_size()) {
                case 1u: s[index++] = v.x; break;
                case 2u: s[index] = v.x; s[index + 1u] = v.y; index += 2u; break;
                case 3u: s[index] = v.x; s[index + 1u] = v.y; s[index + 2u] = v.z; index += 3u; break;
                case 4u: s[index] = v.x; s[index + 1u] = v.y; s[index + 2u] = v.z; s[index + 3u] = v.w; index += 4u; break;
                default: LUISA_ERROR_WITH_LOCATION("Unreachable");
            }
        }
        LUISA_ASSERT(index == node()->channels(), "Invalid channel count for ConcatTexture.");
        return s;
    }
};

luisa::unique_ptr<Texture::Instance> ConcatTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    luisa::vector<const Texture::Instance *> sub_textures;
    for (auto texture : _sub_textures) {
        auto sub_texture = pipeline.build_texture(command_buffer, texture);
        sub_textures.emplace_back(sub_texture);
    }
    return luisa::make_unique<ConcatTextureInstance>(
        pipeline, this, std::move(sub_textures));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ConcatTexture)
