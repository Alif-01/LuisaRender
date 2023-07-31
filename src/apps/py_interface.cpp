// Modified from cli.cpp
//

#include <span>
#include <iostream>
#include <vector>
#include <string>

#include <cxxopts.hpp>

#include <core/stl/format.h>
#include <core/basic_types.h>
#include <sdl/scene_desc.h>
#include <sdl/scene_parser.h>
#include <base/scene.h>
#include <base/pipeline.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/mesh.h>
#include <assimp/scene.h>
#include <assimp/Subdivision.h>
#include <util/thread_pool.h>

#include <backends/ext/denoiser_ext.h>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

[[nodiscard]] auto parse_cli_options(int argc, const char *const *argv) noexcept {
    cxxopts::Options cli{"luisa-render-cli"};
    cli.add_option("", "b", "backend", "Compute backend name", cxxopts::value<luisa::string>(), "<backend>");
    cli.add_option("", "d", "device", "Compute device index", cxxopts::value<uint32_t>()->default_value("0"), "<index>");
    cli.add_option("", "", "scene", "Path to scene description file", cxxopts::value<std::filesystem::path>(), "<file>");
    cli.add_option("", "D", "define", "Parameter definitions to override scene description macros.",
                   cxxopts::value<std::vector<luisa::string>>()->default_value("<none>"), "<key>=<value>");
    cli.add_option("", "h", "help", "Display this help message", cxxopts::value<bool>()->default_value("false"), "");
    cli.allow_unrecognised_options();
    cli.positional_help("<file>");
    cli.parse_positional("scene");
    auto options = [&] {
        try {
            return cli.parse(argc, argv);
        } catch (const std::exception &e) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to parse command line arguments: {}.",
                e.what());
            std::cout << cli.help() << std::endl;
            exit(-1);
        }
    }();
    if (options["help"].as<bool>()) {
        std::cout << cli.help() << std::endl;
        exit(0);
    }
    if (options["scene"].count() == 0u) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION("Scene file not specified.");
        std::cout << cli.help() << std::endl;
        exit(-1);
    }
    if (auto unknown = options.unmatched(); !unknown.empty()) [[unlikely]] {
        luisa::string opts{unknown.front()};
        for (auto &&u : luisa::span{unknown}.subspan(1)) {
            opts.append("; ").append(u);
        }
        LUISA_WARNING_WITH_LOCATION(
            "Unrecognized options: {}", opts);
    }
    return options;
}

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

[[nodiscard]] auto parse_cli_macros(int &argc, char *argv[]) {
    SceneParser::MacroMap macros;

    auto parse_macro = [&macros](luisa::string_view d) noexcept {
        if (auto p = d.find('='); p == luisa::string::npos) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Invalid definition: {}", d);
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
    // parse all options starting with '-D' or '--define'
    for (int i = 1; i < argc; i++) {
        auto arg = luisa::string_view{argv[i]};
        if (arg == "-D" || arg == "--define") {
            if (i + 1 == argc) {
                LUISA_WARNING_WITH_LOCATION(
                    "Missing definition after {}.", arg);
                // remove the option
                argv[i] = nullptr;
            } else {
                parse_macro(argv[i + 1]);
                // remove the option and its argument
                argv[i] = nullptr;
                argv[++i] = nullptr;
            }
        } else if (arg.starts_with("-D")) {
            parse_macro(arg.substr(2));
            // remove the option
            argv[i] = nullptr;
        }
    }
    // remove all nullptrs
    auto new_end = std::remove(argv, argv + argc, nullptr);
    argc = static_cast<int>(new_end - argv);
    return macros;
}

namespace py = pybind11;
using namespace py::literals;
// using buffer_t = py::array_t<float>;

DenoiserExt *denoiser_ext = nullptr;
Stream stream;
Buffer<float> hdr_buffer, denoised_buffer;

// luisa::unordered_map<luisa::string, Shape*> shape_setting;
// luisa::unordered_map<luisa::string, > cameras;
luisa::unique_ptr<Pipeline> pipeline;
luisa::unique_ptr<Scene> scene;
luisa::unordered_map<luisa::string, uint> camera_index;
Geometry::TemplateMapping mapping;

template <typename T>
luisa::vector<T> pyarray_to_vector(const py::array_t<T> &array) noexcept {
    auto pd = array.data();
    luisa::vector<T> v(pd, pd + array.size());
    return std::move(v);
}

luisa::string pystr_to_string(const py::str &s) noexcept {
    return luisa::string(py::cast<std::string>(s));
}

template <typename T>
py::array_t<T> get_default_array(const luisa::vector<T> &a) {
    auto buffer_info = py::buffer_info{
        (void *)a.data(), sizeof(T), py::format_descriptor<T>::format(),
        1, {a.size()}, {sizeof(T)}
    };
    return std::move(py::array_t<T>(buffer_info));
}

// template <typename T>
// luisa::vector<T> get_array_from_dict(const py::dict &dict, const std::string &name) noexcept {
//     if (dict.contains(name)) {
//         return pyarray_to_vector<T>(
//             py::cast<py::array_t<T>>(dict.get(name)));
//     }
//     else return {};
// }

template <typename T, uint N>
Vector<T, N> pyarray_to_pack(const py::array_t<T> &array) noexcept {
    LUISA_ASSERT(array.size() == N, "Array (size = {}) does not match N = {}", array.size(), N);
    LUISA_ASSERT(N >= 2 && N <= 4, "Invalid N = {}", N);
    auto pd = (T *)array.data();
    Vector<T, N> v;
    for (int i = 0; i < N; ++i) v[i] = pd[i];
    return std::move(v);
}

template <typename T, uint N>
luisa::string format_pack(const Vector<T, N> &v) noexcept {
    luisa::string a(std::to_string(v[0]));
    for (int i = 1; i < N; ++i) a += ", " + std::to_string(v[i]);
    return "(" + a + ")";
}


// luisa::string get_string_from_dict(const py::dict &dict, const std::string &name) noexcept {
//     if (dict.contains(name))
//         return luisa::string(py::cast<std::string>(dict.get(name)));
//     else return {};
// }

// RawMeshInfo get_mesh_from_dict(py::dict mesh_info, const SceneDesc *scene_desc) {
//     luisa::string surface_name = get_string_from_dict(mesh_info, "surface");
//     luisa::string light_name = get_string_from_dict(mesh_info, "light");
//     luisa::string medium_name = get_string_from_dict(mesh_info, "medium");

//     return std::move(RawMeshInfo {
//          // dynamic properties
//         get_array_from_dict<float>(mesh_info, "vertices"),
//         get_array_from_dict<uint>(mesh_info, "triangles"),
//         get_array_from_dict<float>(mesh_info, "uvs"),
//         get_array_from_dict<float>(mesh_info, "normals"),
//         get_array_from_dict<float>(mesh_info, "transform"),

//         // static properties
//         (surface_name.empty() ? nullptr : scene_desc.node(surface_name)),
//         (light_name.empty() ? nullptr : scene_desc.node(light_name)),
//         (medium_name.empty() ? nullptr : scene_desc.node(medium_name))
//     });
// }

void init(const std::string &context_path,
          uint32_t cuda_device = 0, uint scene_index = 0) noexcept {
    // auto context_path = "/home/winnie/LuisaRender/build/bin";
    /* add device */
    log_level_info();
    luisa::compute::Context context{context_path};
    luisa::string backend = "CUDA";
    compute::DeviceConfig config;
    config.device_index = cuda_device;
    auto device = context.create_device(backend, &config);

    /* build denoiser */
    auto resolution = make_uint2(256u, 256u);
    auto channel_count = 4u;
    denoiser_ext = device.extension<DenoiserExt>();
    stream = device.create_stream(StreamTag::COMPUTE);
    hdr_buffer = device.create_buffer<float>(resolution.x * resolution.y * 4);
    denoised_buffer = device.create_buffer<float>(resolution.x * resolution.y * 4);

    DenoiserExt::DenoiserMode mode{};
    DenoiserExt::DenoiserInput data;
    data.beauty = &hdr_buffer;
    denoiser_ext->init(stream, mode, data, resolution);

    /* build scene and pipeline */
    std::filesystem::path ctx(context_path);
    auto scene_path = ctx / luisa::format("/default_scene/scene_{}.luisa", scene_index);
    // std::filesystem::path path(scene_file);

    SceneParser::MacroMap macros;
    Clock clock;
    auto scene_desc = SceneParser::parse(scene_path, macros);
    auto parse_time = clock.toc();
    LUISA_INFO("Parsed scene description file '{}' in {} ms.",
               scene_path.string(), parse_time);

    auto desc = scene_desc.get();
    scene = Scene::create(context, desc, camera_index);
    pipeline = Pipeline::create(device, stream, *scene, mapping);
}


RawMeshInfo get_mesh_info(
    const py::str &name, const py::array_t<float> &vertices, const py::array_t<int> &triangles,
    const py::array_t<float> &uvs, const py::array_t<float> &normals, const py::array_t<float> &transform,
    const py::str &surface
) noexcept {
    return std::move(RawMeshInfo {
        pystr_to_string(name),
        pyarray_to_vector<float>(vertices),
        pyarray_to_vector<uint>(triangles),
        pyarray_to_vector<float>(uvs),
        pyarray_to_vector<float>(normals),
        pyarray_to_vector<float>(transform),
        pystr_to_string(surface),
        {}
    });
}

void update_body(
    py::str name, py::array_t<float> vertices = {}, py::array_t<int> triangles = {},
    py::array_t<float> uvs = {}, py::array_t<float> normals = {}, py::array_t<float> transform = {},
    py::str surface = {}, float time = 0
) noexcept {
    // luisa::unordered_map<luisa::string, RawMeshInfo> raw_meshes;
    auto mesh_info = get_mesh_info(
        name, vertices, triangles, uvs, normals, transform, surface
    );

    LUISA_INFO(
        "Updating shape {} => vertices: {}, triangles: {}, uvs: {}, normals: {} surface: {}",
        mesh_info.name, mesh_info.vertices.size(), mesh_info.triangles.size(),
        mesh_info.uvs.size(), mesh_info.normals.size(), mesh_info.surface
    );
    auto shape = scene->update_shape(mesh_info);
    // for (auto it: meshes) {
    //     luisa::string name = py::cast<std::string>(it.first);
    //     // raw_meshes.emplace(name, std::move(raw_meshes));
    // }
}

RawCameraInfo get_camera_info(
    const py::str &name, const py::array_t<float> &position, const py::array_t<float> &look_at,
    py::float_ fov, py::int_ spp, py::float_ radius, const py::array_t<int> &resolution
) noexcept {
    return std::move(RawCameraInfo {
        pystr_to_string(name),
        pyarray_to_pack<float, 3>(position),
        pyarray_to_pack<float, 3>(look_at),
        py::cast<float>(fov),
        uint(py::cast<int>(spp)),
        py::cast<float>(radius),
        pyarray_to_pack<uint, 2>(resolution)
    });
}

void add_camera(
    py::str name, py::array_t<float> position, py::array_t<float> look_at,
    py::float_ fov, py::int_ spp = 10000, py::float_ radius = 1.0f,
    py::array_t<int> resolution = get_default_array(luisa::vector<int>{ 512, 512 })
) noexcept {
    auto camera_info = get_camera_info(name, position, look_at, fov, spp, radius, resolution);
    LUISA_INFO(
        "Adding camera {} => from: {}, to: {}, fov: {}, spp: {}, res: {}", camera_info.name,
        format_pack<float, 3>(camera_info.position),
        format_pack<float, 3>(camera_info.look_at),
        camera_info.fov, camera_info.spp,
        format_pack<uint, 2>(camera_info.resolution)
    );
    auto camera = scene->add_camera(camera_info, camera_index);
}

RawSurfaceInfo get_surface_info (
    const py::str &name, RawSurfaceInfo::RawMaterial material,
    const py::array_t<float> &color, const py::str &image, py::float_ image_scale,
    py::float_ roughness, py::float_ alpha
) noexcept {
    bool is_color = (py::len(image) == 0);
    // return std::move(RawSurfaceInfo {});
    return std::move(RawSurfaceInfo {
        pystr_to_string(name), material, is_color,
        (is_color ? pyarray_to_pack<float, 3>(color) : make_float3(0.0f)),
        (is_color ? "" : pystr_to_string(image)),
        (is_color ? 0.0f : py::cast<float>(image_scale)),
        py::cast<float>(roughness),
        py::cast<float>(alpha)
    });
}

void add_surface(
    py::str name, RawSurfaceInfo::RawMaterial material,
    py::array_t<float> color = {}, py::str image = {}, py::float_ image_scale = 0.f,
    py::float_ roughness = -1.f, py::float_ alpha = 1.f
) noexcept {
    auto surface_info = get_surface_info(
        name, material, color, image, image_scale, roughness, alpha);
    LUISA_INFO(
        "Updating camera {} => material {} {}",
        surface_info.name, RawSurfaceInfo::mat_string[surface_info.material],
        surface_info.is_color ? luisa::format("Color: {}", format_pack<float, 3>(surface_info.color)):
            luisa::format("Image: path {}, scale {}", surface_info.image, surface_info.image_scale)
    );
    auto surface = scene->add_surface(surface_info);
}

    // pipeline->update_shapes(stream, *scene, time, mapping);
    // luisa::unordered_map<luisa::string, Shape*> update_shapes;
    // for (auto it: raw_meshes) {
    //     auto name = it.first;
    //     auto shape = scene->update_shape(name, it.second);
    //     // update_shapes.emplace(name, shape);
    // }

    // const luisa::unordered_map<luisa::string, RawMeshInfo> &raw_meshes;
    // if (auto iter = shape_setting.find(name);
    //     iter != shape_setting.end()) {
    //     shape = iter.second();
    //     shape->update(vertices, triangles, uvs, normals, texture, transform);
    //     update_shapes.emplace(name, shape);
    // } else {
    //     auto new_shape = luisa::make_unique<Shape>(
    //         Shape::create(vertices, triangles, uvs, normals, texture, transform)
    //     );
    //     update_shapes.emplace(name, new_shape);
    // }

void render_frame_exr(py::str name, py::str path, py::float_ time = 0.f) noexcept {
    pipeline->scene_update(stream, *scene, py::cast<float>(time), mapping);
    auto camera_name = luisa::string(py::cast<std::string>(name));
    if (auto it = camera_index.find(camera_name); it == camera_index.end()) {
        LUISA_ERROR_WITH_LOCATION("Failed to find camera name '{}'.", camera_name);
    }
    auto buffer = pipeline->render_to_buffer(stream, camera_index[camera_name]);
    stream.synchronize();

    // save image
    std::filesystem::path img_path = py::cast<std::string>(path); 
    save_image(img_path, buffer, scene->cameras()[0]->film()->resolution());

    // build hdr image
    stream << hdr_buffer.copy_from(buffer);
    stream.synchronize();
    // py::dict deformable_mesh
    // for (auto it : deformable_mesh) {
    //     auto mesh = py::cast<py::tuple>(it.second);
    //     auto vertices = py::cast<py::array_t<float>>(mesh[0]);
    //     auto triangles = py::cast<py::array_t<uint32_t>>(mesh[1]);
    //     auto v = vertices.unchecked<2>();
    //     auto f = triangles.unchecked<2>();
    //     LUISA_INFO("{} => vertices: {}, triangles: {}", py::cast<std::string>(it.first), vertices.ndim(), triangles.ndim());
    //     // mapping[it.first] = 
    // }
    // std::filesystem::path save_path(luisa::format("/home/winnie/LuisaRender/render/{}.exr", i));
    // std::filesystem::path save_path_denoised(luisa::format("/home/winnie/LuisaRender/render/{}_denoised.exr", i));
    // // mapping["liquid"] = mesh_pool[i];
    // auto pipeline = Pipeline::create(device, stream, *scene, mapping);
    // stream.synchronize();
    // save_image(save_path, buffer, scene->cameras()[0]->film()->resolution());
    // stream << hdr_buffer.copy_from(buffer);
    // stream.synchronize();
    // denoiser_ext->process(stream, data);
    // denoiser_ext->get_result(stream, denoised_buffer);
    // stream.synchronize();
    // float *new_buffer = new float[256*256*4];
    // stream << denoised_buffer.copy_to(new_buffer);
    // stream.synchronize();
    // save_image(save_path_denoised, new_buffer, scene->cameras()[0]->film()->resolution());
}

void destroy() {
    if (denoiser_ext != nullptr) {
        denoiser_ext->destroy(stream);
    }
}

PYBIND11_MODULE(LuisaRenderPy, m) {
    m.doc() = "Python binding of LuisaRender";
    m.def("init", &init);
    m.def("destroy", &destroy);
    m.def("update_body", &update_body);
    m.def("add_camera", &add_camera);
    m.def("add_surface", &add_surface);
    m.def("render_frame", &render_frame_exr);
    // m.def("add_body", &add_rigid_body);
    // m.def("add_deformable_body", &add_deformable_body);
    m.def("render_frame_exr", &render_frame_exr);
    py::enum_<RawSurfaceInfo::RawMaterial>(m, "Meterial")
        .value("METAL", RawSurfaceInfo::RawMaterial::RAW_METAL)
        .value("SUBSTRATE", RawSurfaceInfo::RawMaterial::RAW_SUBSTRATE)
        .value("MATTE", RawSurfaceInfo::RawMaterial::RAW_MATTE)
        .value("GLASS", RawSurfaceInfo::RawMaterial::RAW_GLASS)
        .value("NULL", RawSurfaceInfo::RawMaterial::RAW_NULL);
}