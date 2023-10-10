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
#include <apps/py_class.h>

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
    DenoiserExt::DenoiserMode mode{};

    Clock clock;
    auto scene_desc = SceneParser::parse(path, macros);
    auto parse_time = clock.toc();
    LUISA_INFO("Parsed scene description file '{}' in {} ms.", path.string(), parse_time);
    auto desc = scene_desc.get();
    auto scene = Scene::create(context, desc);

    auto camera = scene->cameras()[0];
    auto resolution = camera->film()->resolution();
    uint pixel_count = resolution.x * resolution.y * 4;
    auto hdr_buffer = device.create_buffer<float>(pixel_count);
    auto denoised_buffer = device.create_buffer<float>(pixel_count);

    auto pipeline = Pipeline::create(device, stream, *scene);
    auto picture = pipeline->render_to_buffer(stream, 0);
    auto buffer = reinterpret_cast<float *>((*picture).data());

    LUISA_INFO("Start denoising...");
    stream << hdr_buffer.copy_from(buffer);
    stream.synchronize();
    DenoiserExt::DenoiserInput data;
    data.beauty = &hdr_buffer;

    denoiser_ext->init(stream, mode, data, resolution);
    denoiser_ext->process(stream, data);
    denoiser_ext->get_result(stream, denoised_buffer);
    stream.synchronize();

    stream << denoised_buffer.copy_to(buffer);
    stream.synchronize();
    
    if (render_png) {
        apply_gamma(buffer, resolution);
        std::filesystem::path png_path = img_path;
        png_path.replace_extension(".png");
        auto int_buffer = convert_to_int_pixel(buffer, resolution);
        save_image(png_path, (*int_buffer).data(), resolution);
    } else {
        save_image(img_path, buffer, resolution);
    }

    denoiser_ext->destroy(stream);
    stream.synchronize();
}
