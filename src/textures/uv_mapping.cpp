#include <base/texture.h>
#include <base/scene.h>
#include <base/pipeline.h>

namespace luisa::render {

class UVMappingTexture final : public Texture {

private:
    Texture *_uv_map;
    Texture *_texture;

public:
    UVMappingTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc},
          _uv_map{scene->load_texture(desc->property_node_or_default("uv_map"))},
          _texture{scene->load_texture(desc->property_node_or_default("texture"))} {}

    // is_black() and is_constant is not precise
    [[nodiscard]] bool is_black() const noexcept override {
        return _texture == nullptr || _texture->is_black();
    }
    [[nodiscard]] bool is_constant() const noexcept override {
        return _texture == nullptr || _texture->is_constant();
    }
    [[nodiscard]] uint2 resolution() const noexcept override { return _uv_map->resolution(); }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override {
        return _texture == nullptr ? 4u : _texture->channels();
    }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class UVMappingTextureInstance final : public Texture::Instance {

private:
    const Texture::Instance *_uv_map;
    const Texture::Instance *_texture;

public:
    UVMappingTextureInstance(Pipeline &pipeline, const Texture *node,
                             const Texture::Instance *uv_map, const Texture::Instance *texture) noexcept:
        Texture::Instance{pipeline, node}, _uv_map{uv_map}, _texture{texture} {}
    [[nodiscard]] Float4 evaluate(const Interaction &it,
                                  Expr<float> time) const noexcept override {
        if (_texture == nullptr) {
            return make_float4(0.f);
        }
        if (_uv_map == nullptr) {
            return _texture->evaluate(it, time);
        }

        auto mapped_uv = _uv_map->evaluate(it, time).xy();
        auto mapped_it = it;
        mapped_it.set_uv(mapped_uv);
        return _texture->evaluate(mapped_it, time);
    }
};

luisa::unique_ptr<Texture::Instance> UVMappingTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto uv_map = pipeline.build_texture(command_buffer, _uv_map);
    auto texture = pipeline.build_texture(command_buffer, _texture);
    return luisa::make_unique<UVMappingTextureInstance>(pipeline, this, uv_map, texture);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::UVMappingTexture)
