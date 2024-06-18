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

/**
* # bump_image = np.power(
            #     cv.imread(material_bump, cv.IMREAD_GRAYSCALE) / 255, 2.2
            # )
            # h, w = bump_image.shape[:2]
            # scale = 4
            # bump_image = cv.resize(bump_image, (w * scale, h * scale), interpolation=cv.INTER_LANCZOS4)
            # bump_image = cv.GaussianBlur(bump_image, (5, 5), 0)
            # dx_image = cv.copyMakeBorder(bump_image, 0, 0, 1, 1, cv.BORDER_REPLICATE)
            # dy_image = cv.copyMakeBorder(bump_image, 1, 1, 0, 0, cv.BORDER_REPLICATE)
            # strength = min(w, h) / 50 * scale
            # dx_image = np.clip(strength * (dx_image[:, 2:] - dx_image[:, :-2]), -5, 5)
            # dy_image = np.clip(-strength * (dy_image[2:, :] - dy_image[:-2, :]), -5, 5)
            # dx_image = cv.resize(dx_image, (w, h), interpolation=cv.INTER_AREA)
            # dy_image = cv.resize(dy_image, (w, h), interpolation=cv.INTER_AREA)
            # dz_image = np.ones_like(dx_image)
            # norm = np.sqrt(dx_image**2 + dy_image**2 + dz_image**2)
            # normal_image = np.dstack([dz_image, dy_image, dx_image])
            # normal_image = (normal_image / norm[:, :, np.newaxis]) * 0.5 + 0.5
            # # normal_image = cv.GaussianBlur(normal_image, (3, 3), 0)
            # # normal_image = cv.resize(normal_image, (w, h), interpolation=cv.INTER_CUBIC)
            # normal_image = np.uint8(np.clip(normal_image * 255, 0, 255))
            # save_name = f"lr_exported_textures/{material_bump.split('/')[-1]}"
            # cv.imwrite(save_name, normal_image)
*/

inline auto builtin_gaussian_filter_desc(uint kernel_size) noexcept {
    static const auto nodes = [] {
        auto make_desc = [](uint kernel_size) noexcept {
            auto desc = luisa::make_shared<SceneNodeDesc>(
                luisa::format("__bump2normal_texture_builtin_gaussian{}x{}", kernel_size, kernel_size),
                SceneNodeTag::FILTER);
            desc->define(SceneNodeTag::FILTER, "Gaussian", {});
            auto radius = (kernel_size + 1u) / 2.f;
            auto sigma = radius / 3.f;
            desc->add_property("radius", radius);
            desc->add_property("sigma", sigma);
            return eastl::make_pair(std::move(kernel_size), std::move(desc));
        };
        using namespace std::string_view_literals;
        return luisa::fixed_map<uint, luisa::shared_ptr<SceneNodeDesc>, 2>{
            make_desc(3),
            make_desc(5)};
    }();
    auto iter = nodes.find(kernel_size);
    return iter == nodes.cend() ? nullptr : iter->second.get();
}

class Bump2NormalTexture final : public Texture {

private:
    Texture *_bump_texture;
    Filter *_gaussian5x5, *_gaussian3x3;// TODO
    float _scale{1.f};

public:
    Bump2NormalTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc} {
        _scale = desc->property_float_or_default("scale", 1.f);

        _bump_texture = scene->load_texture(desc->property_node("bump"));
        if (_bump_texture->channels() != 1u) {
            LUISA_WARNING_WITH_LOCATION("Bump image {} should only have 1 channel. {} found.",
                                        desc->identifier(),
                                        _bump_texture->channels());
        }
        LUISA_ASSERT(all(_bump_texture->resolution() >= 3u),
                     "Bump image {} resolution conflicts with the algorithm.",
                     desc->identifier());

        _gaussian5x5 = scene->load_filter(builtin_gaussian_filter_desc(5));
        _gaussian3x3 = scene->load_filter(builtin_gaussian_filter_desc(3));
    }
    //    [[nodiscard]] auto bump_texture() const noexcept { return _bump_texture; }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] bool is_constant() const noexcept override { return false; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override { return 3u; }
    [[nodiscard]] uint2 resolution() const noexcept override {
        return _bump_texture->resolution();
    }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class Bump2NormalTextureInstance final : public Texture::Instance {

private:
    uint _normal_texture_id;

public:
    Bump2NormalTextureInstance(Pipeline &pipeline,
                               const Texture *texture,
                               uint normal_texture_id) noexcept
        : Texture::Instance{pipeline, texture},
          _normal_texture_id{normal_texture_id} {}
    [[nodiscard]] Float4 evaluate(
        const Interaction &it, Expr<float> time) const noexcept override {
        auto v = pipeline().tex2d(_normal_texture_id).sample(it.uv());
        return v;
    }
};

luisa::unique_ptr<Texture::Instance> Bump2NormalTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {

    Clock clk;

    auto bump_texture = pipeline.build_texture(command_buffer, _bump_texture);
    auto resolution = bump_texture->node()->resolution();
    auto resolution_scaled = make_uint2(resolution.x * _scale, resolution.y * _scale);
    auto strength = std::min(resolution.x, resolution.y) * _scale;

    auto bump_scaled = pipeline.create<Image<float>>(PixelStorage::FLOAT1, resolution_scaled, 1u);
    auto gaussian_blurred = pipeline.create<Image<float>>(PixelStorage::FLOAT1, resolution_scaled, 1u);
    auto dx = pipeline.create<Image<float>>(PixelStorage::FLOAT1, resolution_scaled, 1u);
    auto dy = pipeline.create<Image<float>>(PixelStorage::FLOAT1, resolution_scaled, 1u);
    auto normal = pipeline.create<Image<float>>(PixelStorage::FLOAT4, resolution, 1u);

    Kernel2D scale_kernel = [&](ImageFloat target) {
        auto dispatch_id = compute::dispatch_id().xy();
        auto resolution = dispatch_size().xy();
        auto uv = make_float2(
            dispatch_id.x / Float(resolution.x),
            dispatch_id.y / Float(resolution.y));
        auto value = bump_texture->evaluate(Interaction(uv), 0.f);
        target->write(dispatch_id, value);
    };
    auto scale_shader = pipeline.device().compile<2>(scale_kernel);
    command_buffer << scale_shader(*bump_scaled).dispatch(resolution_scaled) << commit();

    Kernel2D gaussian_kernel = [&](ImageFloat src, ImageFloat target, UInt kernel_size) {
        auto dispatch_id = compute::dispatch_id().xy();
        auto resolution = dispatch_size().xy();
        auto dispatch_id_int = make_int2(dispatch_id);
        auto resolution_int = make_int2(resolution);
        auto half_kernel_size = Int(kernel_size) / 2;
        auto value = def(0.f);
        $for (i, -half_kernel_size, half_kernel_size + 1) {
            $for (j, -half_kernel_size, half_kernel_size + 1) {
                auto offset = make_int2(i, j);
                auto index = abs(dispatch_id_int + offset);
                index = ite(
                    index > resolution_int - 1,
                    2 * resolution_int - index - 2,
                    index);
                auto index_uint = make_uint2(index);
                auto v = src->read(index_uint).x;
                // TODO: Gaussian Filter only have host API
                auto x_squared = length_squared(make_float2(offset));
                value += v / std::sqrt(pi * 2.f) * exp(-x_squared / 2.f);
            };
        };
        target->write(dispatch_id, make_float4(value));
    };
    auto gaussian_shader = pipeline.device().compile<2>(gaussian_kernel);
    command_buffer << gaussian_shader(*bump_scaled, *gaussian_blurred, 5u).dispatch(resolution_scaled)
                   << commit();

    Kernel2D dxdy_kernel = [](ImageFloat src, ImageFloat dx, ImageFloat dy, Float strength) {
        auto id = dispatch_id().xy();
        auto resolution = dispatch_size().xy();

        auto x1_id = id.x + 1u;
        x1_id = ite(x1_id == resolution.x, resolution.x - 1u, x1_id);
        auto x2_id = id.x - 1u;
        x2_id = ite(x2_id == -1u, 0u, x2_id);
        auto x1 = src->read(make_uint2(x1_id, id.y));
        auto x2 = src->read(make_uint2(x2_id, id.y));
        auto dx_value = clamp(strength * (x1 - x2), -5.f, 5.f);
        dx->write(id, dx_value);

        auto y1_id = id.y + 1u;
        y1_id = ite(y1_id == resolution.y, resolution.y - 1u, y1_id);
        auto y2_id = id.y - 1u;
        y2_id = ite(y2_id == -1u, 0u, y2_id);
        auto y1 = src->read(make_uint2(id.x, y1_id));
        auto y2 = src->read(make_uint2(id.x, y2_id));
        auto dy_value = clamp(-strength * (y1 - y2), -5.f, 5.f);
        dy->write(id, dy_value);
    };
    auto dxdy_shader = pipeline.device().compile<2>(dxdy_kernel);
    command_buffer << dxdy_shader(*gaussian_blurred, *dx, *dy, strength).dispatch(resolution_scaled)
                   << commit();

    // TODO: area interpolation
    TextureSampler sampler{TextureSampler::Filter::LINEAR_LINEAR, TextureSampler::Address::MIRROR};
    auto dx_tex_id = pipeline.register_bindless(*dx, sampler);
    auto dy_tex_id = pipeline.register_bindless(*dy, sampler);

    Kernel2D normal_kernel = [&](UInt dx_tex_id, UInt dy_tex_id, ImageFloat normal) {
        auto id = dispatch_id().xy();
        auto resolution = dispatch_size().xy();
        auto uv = make_float2(
            id.x / Float(resolution.x),
            id.y / Float(resolution.y));
        auto dx = pipeline.tex2d(dx_tex_id).sample(uv).x;
        auto dy = pipeline.tex2d(dy_tex_id).sample(uv).y;
        auto dz = 1.f;
        auto norm = sqrt(dx * dx + dy * dy + dz * dz);
        auto normal_value = make_float4(dx, dy, dz, 1.f) / norm;
        normal_value = (normal_value + 1.f) * 0.5f;
        normal->write(id, normal_value);
    };
    auto normal_shader = pipeline.device().compile<2>(normal_kernel);
    command_buffer << normal_shader(dx_tex_id, dy_tex_id, *normal).dispatch(resolution)
                   << synchronize();
    auto normal_map_tex_id = pipeline.register_bindless(*normal, sampler);

    LUISA_INFO_WITH_LOCATION("Bump2NormalTextureInstance build time: {} ms", clk.toc());
    return luisa::make_unique<Bump2NormalTextureInstance>(pipeline, this, normal_map_tex_id);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::Bump2NormalTexture)