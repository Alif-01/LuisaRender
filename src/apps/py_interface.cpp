// Modified from cli.cpp

#include <span>
#include <iostream>
#include <vector>
#include <string>

#include <luisa/core/stl/format.h>
#include <luisa/core/basic_types.h>
#include <luisa/backends/ext/denoiser_ext.h>
#include <sdl/scene_desc.h>
#include <sdl/scene_parser.h>
#include <base/scene.h>
#include <base/pipeline.h>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>


using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

namespace py = pybind11;
using namespace py::literals;
using PyFloatArr = py::array_t<float>;
using PyIntArr = py::array_t<int>;
// using buffer_t = py::array_t<float>;

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

template <typename T, uint N>
Vector<T, N> pyarray_to_pack(const py::array_t<T> &array) noexcept {
    LUISA_ASSERT(array.size() == N, "Array (size = {}) does not match N = {}", array.size(), N);
    LUISA_ASSERT(N >= 2 && N <= 4, "Invalid N = {}", N);
    auto pd = (T *)array.data();
    Vector<T, N> v;
    for (int i = 0; i < N; ++i) v[i] = pd[i];
    return std::move(v);
}

struct PyTransform {
    PyTransform() noexcept = default;
    PyTransform(const PyFloatArr &transform) noexcept: empty{false} {
        auto vec = pyarray_to_vector<float>(transform);
        for (auto row = 0u; row < 4u; ++row) {
            for (auto col = 0u; col < 4u; ++col) {
                this->transform[col][row] = vec[row * 4u + col];
            }
        }
    }
    PyTransform(const PyFloatArr &translate, const PyFloatArr &rotate, const PyFloatArr &scale) noexcept:
        translate{pyarray_to_pack<float, 3>(translate)},
        rotate{pyarray_to_pack<float, 4>(rotate)},
        scale{pyarray_to_pack<float, 3>(scale)}, empty{false} {}

    bool empty{true};
    float4x4 transform{make_float4x4(1.f)};
    float3 translate{make_float3(0.f)};
    float4 rotate{make_float4(0.f)};
    float3 scale{make_float3(1.f)};
};

struct PyTexture {
    PyTexture() noexcept = default;
    PyTexture(const PyFloatArr &color) noexcept: 
        color{pyarray_to_pack<float, 4>(color)}, empty{false} {}
    PyTexture(std::string_view image, const PyFloatArr &image_scale) noexcept:
        image{image}, color{pyarray_to_pack<float, 4>(image_scale)}, empty{false} {}

    bool empty{true};
    luisa::string image{};
    float4 color{make_float4(1.f)};
};

struct CameraStorage {
    CameraStorage(uint index, Device* device, uint pixel_count) noexcept:
        index{index},
        hdr_buffer{device->create_buffer<float>(pixel_count)},
        denoised_buffer{device->create_buffer<float>(pixel_count)} {} 
    uint index;
    Buffer<float> hdr_buffer;
    Buffer<float> denoised_buffer;
};

enum LogLevel: uint { VERBOSE, INFO, WARNING };

luisa::unique_ptr<Stream> stream;
luisa::unique_ptr<Device> device;
luisa::unique_ptr<Context> context;
luisa::unique_ptr<DenoiserExt::DenoiserMode> mode;
DenoiserExt *denoiser_ext = nullptr;

luisa::unique_ptr<Pipeline> pipeline;
luisa::unique_ptr<Scene> scene;
luisa::unordered_map<luisa::string, luisa::unique_ptr<CameraStorage>> camera_storage;
luisa::string context_storage;
float gamma_factor = 2.2f;

template <typename T, uint N>
luisa::string format_pack(const Vector<T, N> &v) noexcept {
    luisa::string a(std::to_string(v[0]));
    for (int i = 1; i < N; ++i) a += ", " + std::to_string(v[i]);
    return "(" + a + ")";
}

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

void init(std::string_view context_path, uint cuda_device, uint scene_index, LogLevel log_level) noexcept {
    // auto context_path = "/home/winnie/LuisaRender/build/bin";
    /* add device */
    context_storage = context_path;
    context = luisa::make_unique<Context>(context_storage);
    switch (log_level) {
        case VERBOSE: log_level_verbose(); break; 
        case INFO: log_level_info(); break;
        case WARNING: log_level_warning(); break;
    }
    
    luisa::string backend = "CUDA";
    compute::DeviceConfig config;
    config.device_index = cuda_device;
    /* Please make sure that cuda:cuda_device has enough space */
    device = luisa::make_unique<Device>(context->create_device(backend, &config));

    /* build denoiser */
    auto channel_count = 4u;
    stream = luisa::make_unique<Stream>(device->create_stream(StreamTag::COMPUTE));
    denoiser_ext = device->extension<DenoiserExt>();
    mode = luisa::make_unique<DenoiserExt::DenoiserMode>();

    /* build scene and pipeline */
    auto scene_path = std::filesystem::path(context_storage) /
                      luisa::format("default_scene/scene_{}.luisa", scene_index);

    SceneParser::MacroMap macros;
    Clock clock;
    auto scene_desc = SceneParser::parse(scene_path, macros);
    auto parse_time = clock.toc();
    LUISA_INFO("Parsed scene description file '{}' in {} ms.", scene_path.string(), parse_time);

    auto desc = scene_desc.get();
    scene = Scene::create(*context, desc);
    LUISA_INFO("Scene created!");

    auto cameras = desc->root()->property_node_list_or_default("cameras");
    for (uint camera_id = 0; camera_id < cameras.size(); ++camera_id) {
        auto c = cameras[camera_id];
        auto camera = scene->cameras()[camera_id];
        auto resolution = camera->film()->resolution();
        uint pixel_count = resolution.x * resolution.y * 4;
        // camera_storage.emplace(c->identifier(), CameraStorage(camera_id, device.get(), pixel_count));
        camera_storage[c->identifier()] = luisa::make_unique<CameraStorage>(
            camera_id, device.get(), pixel_count
        );
    }

    pipeline = Pipeline::create(*device, *stream, *scene);
    LUISA_INFO("Pipeline created!");
}

void add_environment(
    std::string_view name, const PyTexture &texture, const PyTransform &transform
) noexcept {
    auto environment_info = RawEnvironmentInfo {
        luisa::string(name),
        RawTextureInfo {
            texture.empty,
            texture.image,
            texture.color
        },
        RawTransformInfo {
            transform.empty,
            transform.transform,
            transform.translate,
            transform.rotate,
            transform.scale
        },
    };
    auto environment = scene->add_environment(environment_info);
}

void add_light(std::string_view name, const PyTexture &texture) noexcept {
    auto light_info = RawLightInfo {
        luisa::string(name),
        RawTextureInfo {
            texture.empty,
            texture.image,
            texture.color
        }
    };
    auto light = scene->add_light(light_info);
}

void add_surface(
    std::string_view name, RawSurfaceInfo::RawMaterial material,
    const PyTexture &texture, float roughness
) noexcept {
    auto surface_info = RawSurfaceInfo {
        luisa::string(name),
        material,
        RawTextureInfo {
            texture.empty,
            texture.image,
            texture.color
        },
        roughness
    };
    surface_info.print_info();
    auto surface = scene->add_surface(surface_info);
}

void add_camera(
    std::string_view name,
    const PyFloatArr &position, const PyFloatArr &look_at, const PyFloatArr &up,
    float fov, int spp, const PyIntArr &resolution
) noexcept {
    auto camera_info = RawCameraInfo{
        luisa::string(name),
        pyarray_to_pack<float, 3>(position),
        pyarray_to_pack<float, 3>(look_at),
        pyarray_to_pack<float, 3>(up),
        fov, uint(spp), 
        pyarray_to_pack<uint, 2>(resolution),
        1.0
    };
    camera_info.print_info();
    uint camera_id = scene->cameras().size();
    auto camera = scene->add_camera(camera_info);

    uint pixel_count = camera_info.resolution.x * camera_info.resolution.y * 4;
    // camera_storage.emplace(camera_info.name, CameraStorage(camera_id, device.get(), pixel_count));
    camera_storage[camera_info.name] = luisa::make_unique<CameraStorage>(
        camera_id, device.get(), pixel_count
    );
}

void update_camera(std::string_view name, const PyTransform &transform) noexcept {
    auto real_transform = RawTransformInfo {
        transform.empty,
        transform.transform,
        transform.translate,
        transform.rotate,
        transform.scale
    };
    auto camera = scene->update_camera(name, real_transform);
}

// RawSurfaceInfo get_surface_info (
//     std::string_view name, RawSurfaceInfo::RawMaterial material,
//     const PyFloatArr &color, std::string_view image, float image_scale,
//     float roughness, float alpha
// ) noexcept {
//     bool is_color = (image.length() == 0);
//     // return std::move(RawSurfaceInfo {});
//     return std::move(RawSurfaceInfo {
//         luisa::string(name), material, is_color,
//         (is_color ? pyarray_to_pack<float, 3>(color) : make_float3(0.0f)),
//         (is_color ? "" : luisa::string(image)),
//         (is_color ? 0.0f : image_scale),
//         roughness, alpha
//     });
// }
void add_rigid(
    std::string_view name, std::string_view obj_path,
    const PyFloatArr &vertices, const PyIntArr &triangles, const PyFloatArr &normals, const PyFloatArr &uvs,
    std::string_view surface, std::string_view light
) noexcept {
    auto mesh_info = RawShapeInfo(
        luisa::string(name), RawTransformInfo(),
        luisa::string(surface), luisa::string(light), ""
    );
    if (!obj_path.empty()) {
        mesh_info.build_file_info(luisa::string(obj_path));
    } else {
        mesh_info.build_mesh_info(
            pyarray_to_vector<float>(vertices),
            pyarray_to_vector<uint>(triangles),
            pyarray_to_vector<float>(normals),
            pyarray_to_vector<float>(uvs)
        );
    }
    mesh_info.print_info();
    auto shape = scene->update_shape(mesh_info, "mesh", true);
}

void update_rigid(
    std::string_view name, const PyTransform &transform
) noexcept {
    auto mesh_info = RawShapeInfo(
        luisa::string(name),
        RawTransformInfo {
            transform.empty,
            transform.transform,
            transform.translate,
            transform.rotate,
            transform.scale
        }, "", "", ""
    );
    auto shape = scene->update_shape(mesh_info, "mesh", false);
}

void update_deformable(
    std::string_view name,
    const PyFloatArr &vertices, const PyIntArr &triangles, const PyFloatArr &normals, const PyFloatArr &uvs,
    std::string_view surface, std::string_view light
) noexcept {
    auto mesh_info = RawShapeInfo(
        luisa::string(name), RawTransformInfo(),
        luisa::string(surface), luisa::string(light), ""
    );
    mesh_info.build_mesh_info(
        pyarray_to_vector<float>(vertices),
        pyarray_to_vector<uint>(triangles),
        pyarray_to_vector<float>(normals),
        pyarray_to_vector<float>(uvs)
    );
    mesh_info.print_info();
    auto shape = scene->update_shape(mesh_info, "deformablemesh", false);
}

void update_particles(
    std::string_view name, const PyFloatArr &vertices, float radius,
    std::string_view surface, std::string_view light
) noexcept {
    auto spheres_info = RawShapeInfo(
        luisa::string(name), RawTransformInfo(),
        luisa::string(surface), luisa::string(light), ""
    );
    spheres_info.build_spheres_info(pyarray_to_vector<float>(vertices), std::move(radius), 0u);
    spheres_info.print_info();

    // for (auto i = 0u; i < vertex_count; ++i) {
    //     auto p0 = vert[i * 3u + 0u];
    //     auto p1 = vert[i * 3u + 1u];
    //     auto p2 = vert[i * 3u + 2u];
    //     sphere_infos.emplace_back(
    //         RawShapeInfo {
    //             luisa::format("{}_{}", name, i),
    //             RawTransformInfo(make_float3(p0, p1, p2), make_float4(0.0f), make_float3(radius)),
    //             luisa::string(surface), {}, {}
    //         }
    //     );
    // }
    auto shape = scene->update_shape(spheres_info, "spheregroup", false);
}

luisa::unique_ptr<luisa::vector<uint8_t>> convert_to_int_pixel(
    const float *buffer, uint2 resolution) noexcept {
    auto pixel_count = resolution.x * resolution.y;
    auto int_buffer_handle = luisa::make_unique<luisa::vector<uint8_t>>(pixel_count * 4);
    luisa::vector<uint8_t> &int_buffer = *int_buffer_handle;
    for (int i = 0; i < pixel_count * 4; ++i) {
        if ((i & 3) == 3) {
            int_buffer[i] = std::clamp(int(buffer[i] * 255 + 0.5), 0, 255);
        } else {
            int_buffer[i] = std::clamp(int(std::pow(buffer[i], 1.0f / gamma_factor) * 255 + 0.5), 0, 255);
        }
    }
    return std::move(int_buffer_handle);
}

PyFloatArr render_frame_exr(
    std::string_view name, std::string_view path,
    bool denoise, bool save_picture, bool render_png
) noexcept {
    LUISA_INFO("Start rendering camera {}, saving {}", name, save_picture);
    pipeline->scene_update(*stream, *scene, 0);

    auto camera_name = luisa::string(name);
    if (auto it = camera_storage.find(camera_name); it == camera_storage.end()) {
        LUISA_ERROR_WITH_LOCATION("Failed to find camera name '{}'.", camera_name);
    }
    auto camera_store = camera_storage[camera_name].get();
    auto idx = camera_store->index;
    auto resolution = scene->cameras()[idx]->film()->resolution();
    std::filesystem::path exr_path = path;
    auto picture = pipeline->render_to_buffer(*stream, idx);
    auto buffer = reinterpret_cast<float *>((*picture).data());
    stream->synchronize();

    /* denoise image */
    if (denoise) {
        LUISA_INFO("Start denoising...");
        if (save_picture) {
            std::filesystem::path origin_path(exr_path);
            origin_path.replace_filename(origin_path.stem().string() + "_ori" + origin_path.extension().string());
            save_image(origin_path, buffer, resolution);
        }

        Buffer<float> &hdr_buffer = camera_store->hdr_buffer;
        Buffer<float> &denoised_buffer = camera_store->denoised_buffer;
        (*stream) << hdr_buffer.copy_from(buffer);
        stream->synchronize();

        DenoiserExt::DenoiserInput data;
        data.beauty = &hdr_buffer;
        denoiser_ext->init(*stream, *mode, data, resolution);
        denoiser_ext->process(*stream, data);
        denoiser_ext->get_result(*stream, denoised_buffer);
        stream->synchronize();

        (*stream) << denoised_buffer.copy_to(buffer);
        stream->synchronize();
        denoiser_ext->destroy(*stream);
        stream->synchronize();
    }

    /* save image */
    if (save_picture) {
        save_image(exr_path, buffer, resolution);
        if (render_png) {
            std::filesystem::path png_path = path;
            png_path.replace_extension(".png");
            auto int_buffer = convert_to_int_pixel(buffer, resolution);
            save_image(png_path, (*int_buffer).data(), resolution);
        }
    }

    auto pixel_count = resolution.x * resolution.y;
    for (int i = 0; i < pixel_count * 4; ++i) {
        if ((i & 3) != 3) {
            buffer[i] = std::clamp(std::pow(buffer[i], 1.0f / gamma_factor), 0.0f, 1.0f);
        }
    }
    auto array_buffer = PyFloatArr(pixel_count * 4);
    std::memcpy(array_buffer.mutable_data(), buffer, array_buffer.size() * sizeof(float));
    return std::move(array_buffer);
}

void destroy() {}

PYBIND11_MODULE(LuisaRenderPy, m) {
    m.doc() = "Python binding of LuisaRender";

    py::enum_<RawSurfaceInfo::RawMaterial>(m, "Material")
        .value("METAL", RawSurfaceInfo::RawMaterial::RAW_METAL)
        .value("SUBSTRATE", RawSurfaceInfo::RawMaterial::RAW_SUBSTRATE)
        .value("MATTE", RawSurfaceInfo::RawMaterial::RAW_MATTE)
        .value("GLASS", RawSurfaceInfo::RawMaterial::RAW_GLASS)
        .value("NULL", RawSurfaceInfo::RawMaterial::RAW_NULL);
    py::enum_<LogLevel>(m, "LogLevel")
        .value("DEBUG", LogLevel::VERBOSE)
        .value("INFO", LogLevel::INFO)
        .value("WARNING", LogLevel::WARNING);
    py::class_<PyTransform>(m, "Transform")
        .def(py::init<PyFloatArr, PyFloatArr, PyFloatArr>(),
            py::arg("translate"), py::arg("rotate"), py::arg("scale")
        )
        .def(py::init<PyFloatArr>(), py::arg("transform"))
        .def(py::init<>());
    py::class_<PyTexture>(m, "Texture")
        .def(py::init<std::string_view, PyFloatArr>(), py::arg("image"), py::arg("image_scale"))
        .def(py::init<PyFloatArr>(), py::arg("color"))
        .def(py::init<>());

    m.def("init", &init,
        py::arg("context_path"),
        py::arg("cuda_device") = 0,
        py::arg("scene_index") = 0,
        py::arg("log_level") = LogLevel::WARNING
    );
    m.def("destroy", &destroy);

    m.def("add_environment", &add_environment,
        py::arg("name"),
        py::arg("texture") = PyTexture(),
        py::arg("transform") = PyTransform()
    );
    m.def("add_light", &add_light,
        py::arg("name"),
        py::arg("texture") = PyTexture()
    );
    m.def("add_surface", &add_surface,
        py::arg("name"),
        py::arg("material"),
        py::arg("texture") = PyTexture(),
        py::arg("roughness") = 0.f
    );
    m.def("add_camera", &add_camera,
        py::arg("name"),
        py::arg("position"), py::arg("look_at"), py::arg("up"),
        py::arg("fov"), py::arg("spp"), py::arg("resolution")
    );
    m.def("update_camera", &update_camera,
        py::arg("name"),
        py::arg("transform") = PyTransform()
    );
    m.def("add_rigid", &add_rigid,
        py::arg("name"),
        py::arg("obj_path") = "",
        py::arg("vertices") = PyFloatArr(),
        py::arg("triangles") = PyIntArr(),
        py::arg("normals") = PyFloatArr(),
        py::arg("uvs") = PyFloatArr(),
        py::arg("surface") = "",
        py::arg("light") = ""
    );
    m.def("update_rigid", &update_rigid,
        py::arg("name"),
        py::arg("transform")
    );
    m.def("update_particles", &update_particles,
        py::arg("name"),
        py::arg("vertices"),
        py::arg("radius"),
        py::arg("surface") = "",
        py::arg("light") = ""
    );
    m.def("update_deformable", &update_deformable,
        py::arg("name"),
        py::arg("vertices"),
        py::arg("triangles"),
        py::arg("normals") = PyFloatArr(),
        py::arg("uvs") = PyFloatArr(),
        py::arg("surface") = "",
        py::arg("light") = ""
    );
    m.def("render_frame_exr", &render_frame_exr,
        py::arg("name"),
        py::arg("path"),
        py::arg("denoise") = true,
        py::arg("save_picture") = false,
        py::arg("render_png") = true
    );
}