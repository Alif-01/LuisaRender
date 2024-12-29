//
// Created by Mike Smith on 2023/6/14.
//

#include <imgui.h>

#include <runtime/image.h>
#include <runtime/swapchain.h>

#include <gui/imgui_window.h>

#include <base/film.h>
#include <base/scene.h>
#include <base/pipeline.h>

namespace luisa::render {

class Display final : public Film {

public:
    enum struct ToneMapping : uint8_t {
        NONE,
        UNCHARTED2,
        ACES,
        AgX,
        AgX_GOLDEN,
        AgX_PUNCHY,
    };

private:
    Film *_base;
    float _target_fps;
    float _exposure;
    uint8_t _back_buffers;
    ToneMapping _tone_mapping;
    bool _hdr;
    bool _vsync;

public:
    Display(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Film{scene, desc},
          _base{scene->load_film(desc->property_node("base"))},
          _target_fps{std::clamp(desc->property_float_or_default("target_fps", 30.f), 1.f, 1024.f)},
          _back_buffers{static_cast<uint8_t>(std::clamp(
              desc->property_uint_or_default("back_buffers", 3u), 1u, 8u))},
          _exposure{std::clamp(
              desc->property_float_or_default(
                  "exposure", lazy_construct([desc]() noexcept {
                      return desc->property_float_or_default("exposure", 0.0f);
                  })),
              -10.f, 10.f)},
          _tone_mapping{[desc] {
              auto tm = desc->property_string_or_default(
                  "tone_mapping", lazy_construct([desc]() noexcept {
                      return desc->property_string_or_default("tonemapping", "AgX");
                  }));
              for (auto &c : tm) { c = static_cast<char>(std::tolower(c)); }
              if (tm == "uncharted2") { return ToneMapping::UNCHARTED2; }
              if (tm == "aces") { return ToneMapping::ACES; }
              if (tm == "agx") {
                  auto look = desc->property_string_or_default("look", "default");
                  for (auto &c : look) { c = static_cast<char>(std::tolower(c)); }
                  if (look == "golden") { return ToneMapping::AgX_GOLDEN; }
                  if (look == "punchy") { return ToneMapping::AgX_PUNCHY; }
                  if (!look.empty() || look != "default") {
                      LUISA_WARNING_WITH_LOCATION(
                          "Unknown AgX look: \"{}\". "
                          "Available options are: \"default\", \"golden\", \"punchy\". "
                          "Falling back to default look.",
                          look);
                  }
                  return ToneMapping::AgX;
              }
              if (tm != "none") {
                  LUISA_WARNING_WITH_LOCATION(
                      "Unknown tone mapping operator: \"{}\". "
                      "Available options are: \"none\", \"uncharted2\", \"aces\".",
                      tm);
              }
              return ToneMapping::NONE;
          }()},
          _hdr{desc->property_bool_or_default(
              "HDR", lazy_construct([desc]() noexcept {
                  return desc->property_bool_or_default("hdr", false);
              }))},
          _vsync{desc->property_bool_or_default(
              "VSync", lazy_construct([desc]() noexcept {
                  return desc->property_bool_or_default(
                      "vsync", lazy_construct([desc]() noexcept {
                          return desc->property_bool_or_default("vertical_sync", true);
                      }));
              }))} {}

    [[nodiscard]] luisa::string_view impl_type() const noexcept override {
        return LUISA_RENDER_PLUGIN_NAME;
    }

    [[nodiscard]] uint2 resolution() const noexcept override { return _base->resolution(); }
    [[nodiscard]] float clamp() const noexcept override { return _base->clamp(); }
    [[nodiscard]] auto target_fps() const noexcept { return _target_fps; }
    [[nodiscard]] auto hdr() const noexcept { return _hdr; }
    [[nodiscard]] auto vsync() const noexcept { return _vsync; }
    [[nodiscard]] auto back_buffers() const noexcept { return _back_buffers; }
    [[nodiscard]] float3 exposure() const noexcept override { return make_float3(_exposure); }
    [[nodiscard]] auto tone_mapping() const noexcept { return _tone_mapping; }
    [[nodiscard]] bool is_display() const noexcept override { return true; }

    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

using namespace compute;

class DisplayInstance final : public Film::Instance {

private:
    luisa::unique_ptr<Film::Instance> _base;
    luisa::unique_ptr<ImGuiWindow> _window;
    Image<float> _framebuffer;
    Shader2D<int, bool, float3, float3> _blit;
    Shader2D<Image<float>> _clear;
    Clock _clock;
    Stream *_stream{};
    ImTextureID _background{};
    mutable float3 _exposure{};
    mutable double _last_frame_time{};
    mutable float2 _white_balance{};
    mutable float _brightness{};
    mutable int _tone_mapping{};
    mutable int _background_fit{};
    mutable bool _link_rgb_exposure{true};
    bool _rendering_done{};

private:
    [[nodiscard]] static auto _tone_mapping_uncharted2(Expr<float3> color) noexcept {
        static constexpr auto a = 0.15f;
        static constexpr auto b = 0.50f;
        static constexpr auto c = 0.10f;
        static constexpr auto d = 0.20f;
        static constexpr auto e = 0.02f;
        static constexpr auto f = 0.30f;
        static constexpr auto white = 11.2f;
        auto op = [](auto x) noexcept {
            return (x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f) - e / f;
        };
        return op(1.6f * color) / op(white);
    }
    [[nodiscard]] static auto _tone_mapping_aces(Expr<float3> color) noexcept {
        constexpr auto a = 2.51f;
        constexpr auto b = 0.03f;
        constexpr auto c = 2.43f;
        constexpr auto d = 0.59f;
        constexpr auto e = 0.14f;
        return (color * (a * color + b)) / (color * (c * color + d) + e);
    }
    [[nodiscard]] static auto _tone_mapping_agx(Expr<float3> color, luisa::string_view look) noexcept {
        // The following source code is adapted from
        //   https://iolite-engine.com/blog_posts/minimal_agx_implementation
        // which is licensed under the MIT License:
        //
        // MIT License
        //
        // Copyright (c) 2024 Missing Deadlines (Benjamin Wrensch)
        //
        // Permission is hereby granted, free of charge, to any person obtaining a copy
        // of this software and associated documentation files (the "Software"), to deal
        // in the Software without restriction, including without limitation the rights
        // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        // copies of the Software, and to permit persons to whom the Software is
        // furnished to do so, subject to the following conditions:
        //
        // The above copyright notice and this permission notice shall be included in
        // all copies or substantial portions of the Software.
        //
        // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        // SOFTWARE.

        // All values used to derive this implementation are sourced from Troy’s initial AgX implementation/OCIO config file available here:
        //   https://github.com/sobotka/AgX

        // Mean error^2: 1.85907662e-06
        static Callable agxDefaultContrastApprox = [](Float3 x) noexcept {
            auto x2 = x * x;
            auto x4 = x2 * x2;
            auto x6 = x4 * x2;
            return -17.86f * x6 * x + 78.01f * x6 - 126.7f * x4 * x + 92.06f * x4 - 28.72f * x2 * x + 4.361f * x2 - 0.1718f * x + 0.002857f;
        };

        static Callable agx = [](Float3 val) noexcept {
            const auto agx_mat = make_float3x3(
                0.842479062253094f, 0.0423282422610123f, 0.0423756549057051f,
                0.0784335999999992f, 0.878468636469772f, 0.0784336f,
                0.0792237451477643f, 0.0791661274605434f, 0.879142973793104f);

            const auto min_ev = -12.47393f;
            const auto max_ev = 4.026069f;

            // Input transform (inset)
            val = agx_mat * val;

            // Log2 space encoding
            val = clamp(log2(val), min_ev, max_ev);
            val = (val - min_ev) / (max_ev - min_ev);

            // Apply sigmoid function approximation
            val = agxDefaultContrastApprox(val);

            return val;
        };

        static Callable agxEotf = [](Float3 val) noexcept {
            const auto agx_mat_inv = make_float3x3(
                +1.19687900512017f, -0.0528968517574562f, -0.0529716355144438f,
                -0.0980208811401368f, +1.15190312990417f, -0.0980434501171241f,
                -0.0990297440797205f, -0.0989611768448433f, +1.15107367264116f);

            // Inverse input transform (outset)
            val = agx_mat_inv * val;

            // sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
            // NOTE: We're linearizing the output here. Comment/adjust when
            // *not* using a sRGB render target
            val = pow(val, 2.2f);

            return val;
        };

        Callable agxLook = [look](Float3 val) noexcept {
            const auto lw = make_float3(0.2126, 0.7152, 0.0722);
            auto luma = dot(val, lw);

            // Default
            auto offset = make_float3(0.f);
            auto slope = make_float3(1.f);
            auto power = make_float3(1.f);
            auto sat = 1.f;

            if (look == "golden") {
                // Golden
                slope = make_float3(1.f, .9f, .5f);
                power = make_float3(.8f);
                sat = .8f;
            } else if (look == "punchy") {
                // Punchy
                slope = make_float3(1.f);
                power = make_float3(1.35f);
                sat = 1.4f;
            }

            // ASC CDL
            val = pow(val * slope + offset, power);
            return luma + sat * (val - luma);
        };
        return agxEotf(agxLook(agx(color)));// TODO: implement AgX tone mapping
    }
    [[nodiscard]] static auto _linear_to_srgb(Expr<float3> color) noexcept {
        return ite(color <= .0031308f,
                   color * 12.92f,
                   1.055f * pow(color, 1.f / 2.4f) - .055f);
    }
    [[nodiscard]] static auto _apply_white_balance(Expr<float3> rgb, Expr<float> brightness, Expr<float> temperature, Expr<float> tint) noexcept {
        auto xyz = linear_srgb_to_cie_xyz(rgb);
        auto lum = max(xyz.y, 1.f);
        auto lab = cie_xyz_to_lab(xyz / lum);
        auto v = make_float3(clamp(lab.x + brightness, 0.f, 100.f), lab.y + tint, lab.z + temperature);
        return cie_xyz_to_linear_srgb(lab_to_cie_xyz(v) * lum);
    }

public:
    DisplayInstance(Pipeline &pipeline, const Display *film,
                    luisa::unique_ptr<Film::Instance> base) noexcept:
        Film::Instance{pipeline, film},
        _base{std::move(base)},
        _tone_mapping{luisa::to_underlying(film->tone_mapping())},
        _exposure{film->exposure().x} {}

    [[nodiscard]] Film::Accumulation read(Expr<uint2> pixel) const noexcept override {
        return _base->read(pixel);
    }

    void prepare(CommandBuffer &command_buffer) noexcept override {
        _base->prepare(command_buffer);
        _rendering_done = false;
        auto &&device = pipeline().device();
        auto size = node()->resolution();
        if (!_window) {
            auto d = node<Display>();
            _stream = command_buffer.stream();
            auto window_size = size;
            while (std::max(window_size.x, window_size.y) > 2048u) {
                window_size /= 2u;
            }
            _window = luisa::make_unique<ImGuiWindow>(
                device, *command_buffer.stream(), "Display",
                ImGuiWindow::Config{
                    .size = window_size,
                    .vsync = d->vsync(),
                    .hdr = d->hdr(),
                    .back_buffers = d->back_buffers()});
            _framebuffer = device.create_image<float>(_window->swapchain().backend_storage(), size);
            _background = _window->register_texture(_framebuffer, TextureSampler::linear_point_zero());
            _blit = device.compile<2>([base = _base.get(), &framebuffer = _framebuffer](Int tonemapping, Bool ldr, Float3 scale, Float3 white_balance) noexcept {
                auto p = dispatch_id().xy();
                auto color = base->read(p).average * scale;
                $if (any(white_balance != 0.f)) {
                    color = _apply_white_balance(color, white_balance.z, white_balance.x, white_balance.y);
                };
                color = max(color, 0.f);
                $switch (tonemapping) {
                    $case (static_cast<int>(Display::ToneMapping::NONE)) {};
                    $case (static_cast<int>(Display::ToneMapping::UNCHARTED2)) { color = _tone_mapping_uncharted2(color); };
                    $case (static_cast<int>(Display::ToneMapping::ACES)) { color = _tone_mapping_aces(color); };
                    $case (static_cast<int>(Display::ToneMapping::AgX)) { color = _tone_mapping_agx(color, "default"); };
                    $case (static_cast<int>(Display::ToneMapping::AgX_GOLDEN)) { color = _tone_mapping_agx(color, "golden"); };
                    $case (static_cast<int>(Display::ToneMapping::AgX_PUNCHY)) { color = _tone_mapping_agx(color, "punchy"); };
                    $default { unreachable(); };
                };
                $if (ldr) {// LDR
                    color = _linear_to_srgb(color);
                };
                framebuffer->write(p, make_float4(color, 1.f));
            });
            _clear = device.compile<2>([](ImageFloat image) noexcept {
                image.write(dispatch_id().xy(), make_float4(0.f));
            });
        }
        _last_frame_time = _clock.toc();
    }

    void clear(CommandBuffer &command_buffer) noexcept override {
        _base->clear(command_buffer);
    }

    void download(CommandBuffer &command_buffer, float4 *framebuffer) const noexcept override {
        _base->download(command_buffer, framebuffer);
    }

    void release() noexcept override {
        _rendering_done = true;
        CommandBuffer command_buffer{_stream};
        while (!_window->should_close()) {
            this->show(command_buffer);
        }
        command_buffer << synchronize();
        _window = nullptr;
        _framebuffer = {};
        _base->release();
    }

private:
    [[nodiscard]] float2 _compute_background_size(const ImGuiViewport *viewport, int fit) const noexcept {
        auto frame_size = make_float2(_framebuffer.size());
        auto viewport_size = make_float2(viewport->Size.x, viewport->Size.y);
        switch (fit) {
            case 0: {// aspect fill
                auto aspect = frame_size.x / frame_size.y;
                auto viewport_aspect = viewport_size.x / viewport_size.y;
                auto ratio = aspect < viewport_aspect ? viewport_size.x / frame_size.x : viewport_size.y / frame_size.y;
                return frame_size * ratio;
            }
            case 1: {// aspect fit
                auto aspect = frame_size.x / frame_size.y;
                auto viewport_aspect = viewport_size.x / viewport_size.y;
                auto ratio = aspect > viewport_aspect ? viewport_size.x / frame_size.x : viewport_size.y / frame_size.y;
                return frame_size * ratio;
            }
            case 2: {// stretch
                return viewport_size;
            }
            default: break;
        }
        return viewport_size;
    }

    void _display() const noexcept {
        auto scale = _link_rgb_exposure ? make_float3(luisa::exp2(_exposure.x)) : luisa::exp2(_exposure);
        auto is_ldr = _window->framebuffer().storage() != PixelStorage::FLOAT4;
        auto size = _framebuffer.size();
        *_stream << _blit(_tone_mapping, is_ldr, scale, make_float3(_white_balance, _brightness)).dispatch(size);
        auto viewport = ImGui::GetMainViewport();
        auto bg_size = _compute_background_size(viewport, _background_fit);
        auto p_min = make_float2(viewport->Pos.x, viewport->Pos.y) +
                     .5f * (make_float2(viewport->Size.x, viewport->Size.y) - bg_size);
        ImGui::GetBackgroundDrawList()->AddImage(_background,
                                                 ImVec2{p_min.x, p_min.y},
                                                 ImVec2{p_min.x + bg_size.x, p_min.y + bg_size.y});
        ImGui::Begin("Console", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        {
            ImGui::Text("Render: %ux%u",
                        node()->resolution().x,
                        node()->resolution().y);
            ImGui::Text("Display: %ux%u (%.2ffps)",
                        static_cast<uint>(viewport->Size.x),
                        static_cast<uint>(viewport->Size.y),
                        ImGui::GetIO().Framerate);
            // Exposure
            if (ImGui::Button("Reset##Exposure")) { _exposure = make_float3(); }
            ImGui::SameLine();
            if (_link_rgb_exposure) {
                ImGui::SliderFloat("Exposure", &_exposure.x, -10.f, 10.f);
                _exposure = make_float3(_exposure.x);
            } else {
                ImGui::SliderFloat3("Exposure", &_exposure.x, -10.f, 10.f);
            }
            ImGui::SameLine();
            ImGui::Checkbox("Link", &_link_rgb_exposure);
            // Brightness
            if (ImGui::Button("Reset##Brightness")) { _brightness = 0.f; }
            ImGui::SameLine();
            ImGui::SliderFloat("Brightness", &_brightness, -100.f, 100.f);
            // Temperature
            if (ImGui::Button("Reset##Temperature")) { _white_balance.x = 0.f; }
            ImGui::SameLine();
            ImGui::SliderFloat("Temperature", &_white_balance.x, -100.f, 100.f);
            // Tint
            if (ImGui::Button("Reset##Tint")) { _white_balance.y = 0.f; }
            ImGui::SameLine();
            ImGui::SliderFloat("Tint", &_white_balance.y, -100.f, 100.f);
            constexpr const char *const tone_mapping_names[] = {"None", "Uncharted2", "ACES", "AgX", "AgX (Golden)", "AgX (Punchy)"};
            ImGui::Combo("Tone Mapping", &_tone_mapping, tone_mapping_names, std::size(tone_mapping_names));
            constexpr const char *const fit_names[] = {"Fill", "Fit", "Stretch"};
            ImGui::Combo("Background Fit", &_background_fit, fit_names, std::size(fit_names));
        }
        ImGui::End();
    }

    bool show(CommandBuffer &command_buffer) const noexcept override {
        LUISA_ASSERT(command_buffer.stream() == _stream, "Command buffer stream mismatch.");
        auto interval = 1. / node<Display>()->target_fps();
        if (auto current_time = _clock.toc();
            current_time - _last_frame_time >= interval) {
            _last_frame_time = current_time;
            if (!_rendering_done && _window->should_close()) {
                command_buffer << synchronize();
                exit(0);// FIXME: exit gracefully
            }
            command_buffer << commit();
            _window->prepare_frame();
            *_stream << _clear(_window->framebuffer()).dispatch(_window->framebuffer().size());
            _display();
            _window->render_frame();
            return true;
        }
        return false;
    }

protected:
    void _accumulate(Expr<uint2> pixel, Expr<float3> rgb, Expr<float> effective_spp) const noexcept override {
        _base->accumulate(pixel, rgb, effective_spp);
    }
};

luisa::unique_ptr<Film::Instance> Display::build(Pipeline &pipeline,
                                                 CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<DisplayInstance>(
        pipeline, this, _base->build(pipeline, command_buffer));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::Display)
