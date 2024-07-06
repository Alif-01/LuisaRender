//
// Created by Mike Smith on 2022/3/23.
//

#include <util/thread_pool.h>
#include <util/imageio.h>
#include <base/texture.h>
#include <base/pipeline.h>
#include <textures/constant_base.h>

namespace luisa::render {

using namespace luisa::compute;

class ImageTexture final : public Texture {

public:
    enum struct Encoding : uint {
        LINEAR,
        SRGB,
        GAMMA,
    };

private:
    std::shared_future<LoadedImage> _image; // TODO: release host memory after all builds
    float2 _uv_scale;
    float2 _uv_offset;
    TextureSampler _sampler{};
    Encoding _encoding{};
    float3 _scale{make_float3(1.f)};
    float _gamma{1.f};
    uint _mipmaps{0u};

private:
    void _load_image(std::filesystem::path path) noexcept {
        _image = global_thread_pool().async([path = std::move(path)] {
            auto image = LoadedImage::load(path);
            LUISA_ASSERT(all(image.size() > 0u),
                         "Invalid image resolution for '{}'.",
                         path.string());
            return image;
        });
    }
    
    void _load_image(
        const luisa::vector<float> &image_data,
        const uint2 &resolution, uint channel
    ) noexcept {
        _image = global_thread_pool().async([&] {
            return LoadedImage::load(
                image_data, resolution, channel
            );
        });
    }

    [[nodiscard]]luisa::string _get_encoding(const std::filesystem::path &path) noexcept {
        auto ext = path.extension().string();
        for (auto &c : ext) { c = static_cast<char>(tolower(c)); }
        if (ext == ".exr" || ext == ".hdr") { return "linear"; }
        return "sRGB";
    }

    void _generate_mipmaps_gamma(Pipeline &pipeline, CommandBuffer &command_buffer, Image<float> &image) const noexcept;
    void _generate_mipmaps_linear(Pipeline &pipeline, CommandBuffer &command_buffer, Image<float> &image) const noexcept;
    void _generate_mipmaps_sRGB(Pipeline &pipeline, CommandBuffer &command_buffer, Image<float> &image) const noexcept;

public:
    ImageTexture(Scene *scene, const SceneNodeDesc *desc) noexcept:
        Texture{scene, desc} {
            
        auto filter = desc->property_string_or_default("filter", "bilinear");
        auto address = desc->property_string_or_default("address", "repeat");
        for (auto &c : filter) { c = static_cast<char>(tolower(c)); }
        for (auto &c : address) { c = static_cast<char>(tolower(c)); }
        auto address_mode = [&address, desc] {
            for (auto &c : address) { c = static_cast<char>(tolower(c)); }
            if (address == "zero") { return TextureSampler::Address::ZERO; }
            if (address == "edge") { return TextureSampler::Address::EDGE; }
            if (address == "mirror") { return TextureSampler::Address::MIRROR; }
            if (address == "repeat") { return TextureSampler::Address::REPEAT; }
            LUISA_ERROR(
                "Invalid texture address mode '{}'. [{}]",
                address, desc->source_location().string());
        }();
        auto filter_mode = [&filter, desc] {
            for (auto &c : filter) { c = static_cast<char>(tolower(c)); }
            if (filter == "point") { return TextureSampler::Filter::POINT; }
            if (filter == "bilinear") { return TextureSampler::Filter::LINEAR_POINT; }
            if (filter == "trilinear") { return TextureSampler::Filter::LINEAR_LINEAR; }
            if (filter == "anisotropic" || filter == "aniso") { return TextureSampler::Filter::ANISOTROPIC; }
            LUISA_ERROR(
                "Invalid texture filter mode '{}'. [{}]",
                filter, desc->source_location().string());
        }();
        
        _sampler = {filter_mode, address_mode};
        _uv_scale = desc->property_float2_or_default(
            "uv_scale", lazy_construct([desc] {
                return make_float2(desc->property_float_or_default("uv_scale", 1.0f));
            }));
        _uv_offset = desc->property_float2_or_default(
            "uv_offset", lazy_construct([desc] {
                return make_float2(desc->property_float_or_default("uv_offset", 0.0f));
            }));

        auto path = desc->property_path_or_default("file");
        auto encoding = desc->property_string_or_default(
            "encoding", lazy_construct([&path, this]() noexcept -> luisa::string { return _get_encoding(path); })
        );
        for (auto &c : encoding) { c = static_cast<char>(tolower(c)); }
        if (encoding == "srgb") {
            _encoding = Encoding::SRGB;
        } else if (encoding == "gamma") {
            _encoding = Encoding::GAMMA;
            _gamma = desc->property_float_or_default("gamma", 1.f);
        } else {
            if (encoding != "linear") [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Unknown texture encoding '{}'. "
                    "Fallback to linear encoding. [{}]",
                    encoding, desc->source_location().string());
            }
            _encoding = Encoding::LINEAR;
        }

        _scale = desc->property_float3_or_default(
            "scale", lazy_construct([desc] {
                return make_float3(desc->property_float_or_default("scale", 1.0f));
            })
        );

        _mipmaps = desc->property_uint_or_default(
            "mipmaps", filter_mode == TextureSampler::Filter::ANISOTROPIC ? 0u : 1u);
        if (filter_mode == TextureSampler::Filter::POINT) { _mipmaps = 1u; }
        if (path.string().empty()) {
            _load_image(path);
        } else {
            _load_image(
                desc->property_float_list("image_data"),
                desc->property_uint2("resolution"),
                desc->property_uint("channel")
            );
        }
    }
    
    ImageTexture(Scene *scene, const RawTextureInfo &texture_info) noexcept:
        Texture{scene},
        _sampler{TextureSampler::Filter::LINEAR_POINT, TextureSampler::Address::REPEAT},
        _uv_scale{make_float2(1.0f)}, _uv_offset{make_float2(0.0f)}, _mipmaps{1u} {
        
        if (texture_info.image_info == nullptr) [[unlikely]]
            LUISA_ERROR_WITH_LOCATION("Invalid image info!");
        auto image_info = texture_info.image_info.get();

        _scale = build_constant(image_info->scale).first.xyz();
        std::filesystem::path path = image_info->image;
        auto encoding = _get_encoding(path);
        for (auto &c : encoding) { c = static_cast<char>(tolower(c)); }
        if (encoding == "srgb") _encoding = Encoding::SRGB;
        else _encoding = Encoding::LINEAR;

        if (path.string().empty()) {
            _load_image(image_info->image_data, image_info->resolution, image_info->channel);
        } else {
            _load_image(path);
        }
    }

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_black() const noexcept override { return all(_scale == 0.f); }
    [[nodiscard]] bool is_constant() const noexcept override { return false; }
    [[nodiscard]] uint2 resolution() const noexcept override { return _image.get().size(); }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto gamma() const noexcept { return _gamma; }
    [[nodiscard]] auto uv_scale() const noexcept { return _uv_scale; }
    [[nodiscard]] auto uv_offset() const noexcept { return _uv_offset; }
    [[nodiscard]] auto encoding() const noexcept { return _encoding; }
    [[nodiscard]] uint channels() const noexcept override { return _image.get().channels(); }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class ImageTextureInstance final : public Texture::Instance {

private:
    uint _texture_id;

private:
    [[nodiscard]] Float2 _compute_uv(const Interaction &it) const noexcept {
        auto texture = node<ImageTexture>();
        auto uv_scale = texture->uv_scale();
        auto uv_offset = texture->uv_offset();
        return it.uv() * uv_scale + uv_offset;
    }

    [[nodiscard]] Float4 _decode(Expr<float4> rgba) const noexcept {
        auto texture = node<ImageTexture>();
        auto encoding = texture->encoding();
        auto scale = texture->scale();
        auto rgb = rgba.xyz();
        if (encoding == ImageTexture::Encoding::SRGB) {
            auto linear = ite(
                rgb <= 0.04045f,
                rgb * (1.0f / 12.92f),
                pow((rgb + 0.055f) * (1.0f / 1.055f), 2.4f));
            return make_float4(scale * linear, rgba.w);
        }
        if (encoding == ImageTexture::Encoding::GAMMA) {
            auto gamma = texture->gamma();
            return make_float4(scale * pow(rgb, gamma), rgba.w);
        }
        return make_float4(scale * rgb, rgba.w);
    }

public:
    ImageTextureInstance(const Pipeline &pipeline,
                         const Texture *texture,
                         uint texture_id) noexcept
        : Texture::Instance{pipeline, texture},
          _texture_id{texture_id} {}
    [[nodiscard]] Float4 evaluate(
        const Interaction &it, Expr<float> time) const noexcept override {
        auto uv = _compute_uv(it);
        auto v = pipeline().tex2d(_texture_id).sample(uv);  // TODO: LOD
        return _decode(v);
    }
};

luisa::unique_ptr<Texture::Instance> ImageTexture::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto &&image = _image.get();
    auto device_image = pipeline.create<Image<float>>(image.pixel_storage(), image.size(), _mipmaps);
    auto tex_id = pipeline.register_bindless(*device_image, _sampler);
    command_buffer << device_image->copy_from(image.pixels()) << compute::commit();
    if (device_image->mip_levels() > 1u) {
        switch (_encoding) {
            case Encoding::LINEAR: _generate_mipmaps_linear(pipeline, command_buffer, *device_image); break;
            case Encoding::SRGB: _generate_mipmaps_sRGB(pipeline, command_buffer, *device_image); break;
            case Encoding::GAMMA: _generate_mipmaps_gamma(pipeline, command_buffer, *device_image); break;
            default: LUISA_ERROR_WITH_LOCATION("Unknown texture encoding.");
        }
    }
    return luisa::make_unique<ImageTextureInstance>(pipeline, this, tex_id);
}

void ImageTexture::_generate_mipmaps_gamma(Pipeline &pipeline, CommandBuffer &command_buffer, Image<float> &image) const noexcept {
    // TODO
}

void ImageTexture::_generate_mipmaps_linear(Pipeline &pipeline, CommandBuffer &command_buffer, Image<float> &image) const noexcept {
    // TODO
}

void ImageTexture::_generate_mipmaps_sRGB(Pipeline &pipeline, CommandBuffer &command_buffer, Image<float> &image) const noexcept {
    // TODO
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ImageTexture)

LUISA_EXPORT_API luisa::render::SceneNode *create_raw(
    luisa::render::Scene *scene, const luisa::render::RawTextureInfo &texture_info) LUISA_NOEXCEPT {
    return luisa::new_with_allocator<luisa::render::ImageTexture>(scene, texture_info);
}