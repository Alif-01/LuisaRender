//
// Modified from cli.cpp
//

#include <span>
#include <iostream>
#include <vector>

#include <luisa/core/stl/format.h>
#include <sdl/scene_desc.h>
#include <sdl/scene_parser.h>
#include <base/scene.h>
#include <base/pipeline.h>
#include <apps/app_base.h>

#include <luisa/backends/ext/denoiser_ext.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;
namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    luisa::compute::Context context{argv[0]};
    auto macros = parse_macros(argc, argv);
    auto options = parse_options(argc, argv, "pipe-render");
    log_level_info();
    auto backend = options["backend"].as<luisa::string>();
    auto index = options["device"].as<uint32_t>();
    auto path = options["scene"].as<fs::path>();
    auto mark = options["mark"].as<luisa::string>();
    auto output_dir = options["output_dir"].as<fs::path>();
    auto render_png = options["render_png"].as<bool>();
    auto filename = luisa::format("image_{}.exr", mark);
    if (output_dir.empty())
        output_dir = path.parent_path();
    auto img_path = output_dir / filename;
    
    DeviceConfig config;
    config.device_index = index;
    auto device = context.create_device(backend, &config);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto denoiser_ext = device.extension<DenoiserExt>();

    Clock clock;
    auto scene_desc = SceneParser::parse(path, macros);
    auto parse_time = clock.toc();
    LUISA_INFO("Parsed scene description file '{}' in {} ms.", path.string(), parse_time);
    auto desc = scene_desc.get();
    auto scene = Scene::create(context, desc);

    auto camera = *(scene->cameras().begin());
    auto resolution = camera->film()->resolution();
    uint pixel_count = resolution.x * resolution.y;

    auto pipeline = Pipeline::create(device, *scene);
    luisa::vector<float4> buffer;
    
    pipeline->set_time(0);
    pipeline->update(stream);
    pipeline->render_to_buffer(stream, camera, buffer);
    auto buffer_p = reinterpret_cast<float *>(buffer.data());

    auto color_buffer = device.create_buffer<float4>(pixel_count);
    auto output_buffer = device.create_buffer<float4>(pixel_count);
    auto denoiser = denoiser_ext->create(stream);
    {
        auto input = DenoiserExt::DenoiserInput{resolution.x, resolution.y};
        input.push_noisy_image(
            color_buffer.view(), output_buffer.view(),
            DenoiserExt::ImageFormat::FLOAT3,
            DenoiserExt::ImageColorSpace::HDR
        );
        input.noisy_features = false;
        input.filter_quality = DenoiserExt::FilterQuality::DEFAULT;
        input.prefilter_mode = DenoiserExt::PrefilterMode::NONE;
        denoiser->init(input);
    }
    stream << color_buffer.copy_from(buffer_p) << synchronize();
    denoiser->execute(true);
    stream << output_buffer.copy_to(buffer_p) << synchronize();
    auto denoise_time = clock.toc();
    LUISA_INFO("Denoised image in {} ms.", denoise_time - parse_time);

    if (render_png) {
        apply_gamma(buffer_p, resolution);
        std::filesystem::path png_path = img_path;
        png_path.replace_extension(".png");
        auto int_buffer = convert_to_int_pixel(buffer_p, resolution);
        save_image(png_path, (*int_buffer).data(), resolution);
    } else {
        save_image(img_path, buffer_p, resolution);
    }
}
