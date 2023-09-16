//
// Created by Mike on 2021/12/7.
//

#include <span>
#include <iostream>

#include <cxxopts.hpp>

#include <luisa/core/stl/format.h>
#include <sdl/scene_desc.h>
#include <sdl/scene_parser.h>
#include <base/scene.h>
#include <base/pipeline.h>
#include <apps/app_base.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

int main(int argc, char *argv[]) {
    luisa::compute::Context context{argv[0]};
    auto macros = parse_macros(argc, argv);
    auto options = parse_options(argc, argv, "cli");
    log_level_info();
    auto backend = options["backend"].as<luisa::string>();
    auto index = options["device"].as<uint32_t>();
    auto path = options["scene"].as<std::filesystem::path>();
    compute::DeviceConfig config;
    config.device_index = index;
    auto device = context.create_device(backend, &config);

    Clock clock;
    auto scene_desc = SceneParser::parse(path, macros);
    auto parse_time = clock.toc();
    LUISA_INFO("Parsed scene description file '{}' in {} ms.", path.string(), parse_time);

    luisa::unordered_map<luisa::string, CameraStorage> camera_storage;
    while (1) {
		auto scene = Scene::create(context, scene_desc.get(), device, camera_storage);
		auto stream = device.create_stream(StreamTag::COMPUTE);
		auto pipeline = Pipeline::create(device, stream, *scene, {});
		pipeline->render(stream);
		stream.synchronize();
    }
}
