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

#if defined(LUISA_PLATFORM_WINDOWS)
#include <windows.h>
[[nodiscard]] auto get_current_exe_path() noexcept {
    constexpr auto max_path_length = std::max<size_t>(MAX_PATH, 4096);
    std::filesystem::path::value_type path[max_path_length] = {};
    auto nchar = GetModuleFileNameW(nullptr, path, max_path_length);
    if (nchar == 0 ||
        (nchar == MAX_PATH &&
         ((GetLastError() == ERROR_INSUFFICIENT_BUFFER) ||
          (path[MAX_PATH - 1] != 0)))) {
        LUISA_ERROR_WITH_LOCATION("Failed to get current executable path.");
    }
    return std::filesystem::canonical(path).string();
}
#elif defined(LUISA_PLATFORM_APPLE)
#include <libproc.h>
#include <unistd.h>
[[nodiscard]] auto get_current_exe_path() noexcept {
    char pathbuf[PROC_PIDPATHINFO_MAXSIZE] = {};
    auto pid = getpid();
    if (auto size = proc_pidpath(pid, pathbuf, sizeof(pathbuf)); size > 0) {
        luisa::string_view path{pathbuf, static_cast<size_t>(size)};
        return std::filesystem::canonical(path).string();
    }
    LUISA_ERROR_WITH_LOCATION(
        "Failed to get current executable path (PID = {}): {}.",
        pid, strerror(errno));
}
#else
#include <unistd.h>
[[nodiscard]] auto get_current_exe_path() noexcept {
    char pathbuf[PATH_MAX] = {};
    for (auto p : {"/proc/self/exe", "/proc/curproc/file", "/proc/self/path/a.out"}) {
        if (auto size = readlink(p, pathbuf, sizeof(pathbuf)); size > 0) {
            luisa::string_view path{pathbuf, static_cast<size_t>(size)};
            return std::filesystem::canonical(path).string();
        }
    }
    LUISA_ERROR_WITH_LOCATION(
        "Failed to get current executable path.");
}
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

int main(int argc, char *argv[]) {

    auto exe_path = get_current_exe_path();
    luisa::compute::Context context{exe_path};
    auto macros = parse_macros(argc, argv);
    auto options = parse_options(argc, argv, "cli");
    if (options["verbose"].as<bool>()) {
        log_level_verbose();
    } else {
        log_level_info();
    }
    auto backend = options["backend"].as<luisa::string>();
    auto index = options["device"].as<std::size_t>();
    auto path = options["scene"].as<std::filesystem::path>();
    compute::DeviceConfig config {
        .device_index = index,
        .inqueue_buffer_limit = false // Do not limit the number of in-queue buffers --- we are doing offline rendering!    
    };
    auto device = context.create_device(backend, &config);

    Clock clock;
    auto scene_desc = SceneParser::parse(path, macros);
    auto parse_time = clock.toc();
    LUISA_INFO("Parsed scene description file '{}' in {} ms.", path.string(), parse_time);

    auto scene = Scene::create(context, scene_desc.get());
    auto stream = device.create_stream(StreamTag::GRAPHICS);
    auto pipeline = Pipeline::create(device, stream, *scene);
    pipeline->render(stream);
    stream.synchronize();
}
