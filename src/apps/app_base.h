#pragma once

#include <span>
#include <iostream>
#include <vector>
#include <string>

#include <luisa/core/stl/format.h>
#include <sdl/scene_parser.h>
#include <cxxopts.hpp>

using namespace luisa;
using namespace luisa::render;
namespace fs = std::filesystem;

[[nodiscard]] auto parse_macros(int &argc, char *argv[], bool print_macro=true) noexcept {
    SceneParser::MacroMap macros;

    auto parse_macro = [&macros](luisa::string_view d) noexcept {
        if (auto p = d.find('='); p == luisa::string::npos) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION("Invalid definition: {}", d);
        } else {
            auto key = d.substr(0, p);
            auto value = d.substr(p + 1);
            LUISA_VERBOSE_WITH_LOCATION("Parameter definition: {} = '{}'", key, value);
            if (auto iter = macros.find(key); iter != macros.end()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate definition: {} = '{}'. "
                    "Ignoring the previous one: {} = '{}'.",
                    key, value, key, iter->second);
                iter->second = value;
            } else {
                macros.emplace(key, value);
            }
        }
    };

    /* parse all options starting with '-D' or '--define' */
    for (int i = 1; i < argc; i++) {
        auto arg = luisa::string_view{argv[i]};
        if (arg == "-D" || arg == "--define") {
            if (i + 1 == argc) {
                LUISA_WARNING_WITH_LOCATION("Missing definition after {}.", arg);
                argv[i] = nullptr;          // remove the option
            } else {
                parse_macro(argv[i + 1]);
                argv[i] = nullptr;          // remove the option and its argument
                argv[++i] = nullptr;
            }
        } else if (arg.starts_with("-D")) {
            parse_macro(arg.substr(2));
            argv[i] = nullptr;              // remove the option
        }
    }
    
    auto new_end = std::remove(argv, argv + argc, nullptr);     // remove all nullptrs
    argc = static_cast<int>(new_end - argv);
    
    if (print_macro) {
        for (auto &&[k, v] : macros) {
            LUISA_INFO("Found Macro: {} = {}", k, v);
        }
    }
    return macros;
}

void add_render_options(cxxopts::Options &parser) noexcept {
    parser.add_option("", "o", "output_dir", "Path to output image directory", cxxopts::value<fs::path>()->default_value(""), "<dir>");
    parser.add_option("", "m", "mark", "Identifier of the scene", cxxopts::value<luisa::string>()->default_value(""), "<mark>");
    parser.add_option("", "l", "log_level", "Logging level of renderer", cxxopts::value<luisa::string>()->default_value("info"), "<logging-level>");
    parser.add_option("", "r", "render_png", "Whether to render png", cxxopts::value<bool>(), "<render>");
}

void add_cli_options(cxxopts::Options &parser) noexcept {
    parser.add_option("", "v", "verbose", "Enable verbose logging", cxxopts::value<bool>()->default_value("false"), "");
}

[[nodiscard]] auto parse_options(
    int argc, const char *const *argv,
    luisa::string_view app_name
) noexcept {
    auto parser_name = luisa::format("luisa-render-{}", app_name);
    cxxopts::Options parser{std::string(parser_name)};

    if (app_name == "pipe-render") {
        add_render_options(parser);
    } else if (app_name == "cli") {
        add_cli_options(parser);
    }

    parser.add_option("", "b", "backend", "Compute backend name", cxxopts::value<luisa::string>(), "<backend>");
    parser.add_option("", "d", "device", "Compute device index", cxxopts::value<uint32_t>()->default_value("0"), "<index>");
    parser.add_option("", "", "scene", "Path to scene description file", cxxopts::value<std::filesystem::path>(), "<file>");
    parser.add_option("", "D", "define", "Parameter definitions to override scene description macros.",
        cxxopts::value<std::vector<luisa::string>>()->default_value("<none>"), "<key>=<value>");
    parser.add_option("", "h", "help", "Display this help message", cxxopts::value<bool>()->default_value("false"), "");

    parser.allow_unrecognised_options();
    parser.positional_help("<file>");
    parser.parse_positional("scene");

    auto options = [&] {
        try {
            return parser.parse(argc, argv);
        } catch (const std::exception &e) {
            LUISA_WARNING_WITH_LOCATION("Failed to parse command line arguments: {}.", e.what());
            std::cout << parser.help() << std::endl;
            exit(-1);
        }
    }();

    if (options["help"].as<bool>()) {
        std::cout << parser.help() << std::endl;
        exit(0);
    }
    if (options["scene"].count() == 0u) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION("Scene file not specified.");
        std::cout << parser.help() << std::endl;
        exit(-1);
    }
    if (auto unknown = options.unmatched(); !unknown.empty()) [[unlikely]] {
        luisa::string opts{unknown.front()};
        for (auto &&u : luisa::span{unknown}.subspan(1)) { opts.append("; ").append(u); }
        LUISA_WARNING_WITH_LOCATION("Unrecognized options: {}", opts);
    }
    return options;
}

void apply_gamma(float *buffer, uint2 resolution) noexcept {
    static const float gamma_factor = 2.2f;
    auto pixel_count = resolution.x * resolution.y;
    for (int i = 0; i < pixel_count * 4; ++i) {
        if ((i & 3) != 3) {
            buffer[i] = std::clamp(std::pow(buffer[i], 1.0f / gamma_factor), 0.0f, 1.0f);
        } else {
            buffer[i] = 1.0f;
        }
    }
}

void convert_uint8(uint8_t *uint_image, float *float_image, uint2 resolution) noexcept {
    auto pixel_count = resolution.x * resolution.y;
    for (int i = 0; i < pixel_count * 4; ++i) {
        uint_image[i] = static_cast<uint8_t>(std::clamp(float_image[i] * 255.0f, 0.0f, 255.0f));
    }
}

[[nodiscard]] luisa::unique_ptr<luisa::vector<uint8_t>> convert_to_int_pixel(
    const float *buffer, uint2 resolution
) noexcept {
    auto pixel_count = resolution.x * resolution.y;
    auto int_buffer_handle = luisa::make_unique<luisa::vector<uint8_t>>(pixel_count * 4);
    luisa::vector<uint8_t> &int_buffer = *int_buffer_handle;
    for (int i = 0; i < pixel_count * 4; ++i) {
        int_buffer[i] = std::clamp(int(buffer[i] * 255 + 0.5), 0, 255);
    }
    return int_buffer_handle;
}