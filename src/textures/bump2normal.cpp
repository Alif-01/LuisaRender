//
// Created by mike on 4/17/24.
//

#include <util/thread_pool.h>
#include <util/imageio.h>
#include <base/texture.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

using namespace luisa::compute;

class Bump2NormalTexture final : public Texture {

private:
    Texture *_bump_texture;

public:
    Bump2NormalTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc},
          _bump_texture{scene->load_texture(desc->property_node("bump"))} {
        if (_bump_texture->channels() != 1u) {
            LUISA_WARNING_WITH_LOCATION("Bump image {} should only have 1 channel. {} found.",
                                        desc->identifier(),
                                        _bump_texture->channels());
        }
    }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] bool is_constant() const noexcept override { return false; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override { return 3u; }
    [[nodiscard]] uint2 resolution() const noexcept override { return _bump_texture->resolution(); }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class Bump2NormalTextureInstance final : public Texture::Instance {

private:
    const Texture::Instance *_bump;

public:
    Bump2NormalTextureInstance(Pipeline &pipeline,
                               const Texture *node,
                               const Texture::Instance *bump) noexcept
        : Texture::Instance{pipeline, node}, _bump{bump} {}
    [[nodiscard]] Float4 evaluate(const Interaction &it, Expr<float> time) const noexcept override {
        auto n = def(make_float3(0.f));
        $outline {
            auto size = static_cast<float>(std::max(node()->resolution().x, node()->resolution().y));
            auto step = std::min(1.f / size, 1.f / 256.f);
            auto dx = _bump->evaluate(Interaction{it.uv() + make_float2(step, 0.f)}, time).x -
                      _bump->evaluate(Interaction{it.uv() - make_float2(step, 0.f)}, time).x;
            auto dy = _bump->evaluate(Interaction{it.uv() + make_float2(0.f, step)}, time).x -
                      _bump->evaluate(Interaction{it.uv() - make_float2(0.f, step)}, time).x;
            auto dxy = make_float2(dx / (2.f * step), -dy / (2.f * step));
            n = normalize(make_float3(dxy, max(1.f, length(dxy))));
        };
        return make_float4(n * 0.5f + 0.5f, 1.f);
    }
};

luisa::unique_ptr<Texture::Instance> Bump2NormalTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto bump = pipeline.build_texture(command_buffer, _bump_texture);
    return luisa::make_unique<Bump2NormalTextureInstance>(pipeline, this, bump);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::Bump2NormalTexture)