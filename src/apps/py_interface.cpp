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

template <typename T>
luisa::vector<T> pyarray_to_vector(const py::array_t<T> &array) noexcept {
    auto pd = array.data();
    luisa::vector<T> v(pd, pd + array.size());
    return v;
}

// luisa::string pystr_to_string(const py::str &s) noexcept {
//     return luisa::string(py::cast<std::string>(s));
// }

template <typename T>
py::array_t<T> get_default_array(const luisa::vector<T> &a) {
    auto buffer_info = py::buffer_info{
        (void *)a.data(), sizeof(T), py::format_descriptor<T>::format(),
        1, {a.size()}, {sizeof(T)}
    };
    return py::array_t<T>(buffer_info);
}

template <typename T, uint N>
Vector<T, N> pyarray_to_pack(const py::array_t<T> &array) noexcept {
    LUISA_ASSERT(array.size() == N, "Array (size = {}) does not match N = {}", array.size(), N);
    LUISA_ASSERT(N >= 2 && N <= 4, "Invalid N = {}", N);
    auto pd = (T *)array.data();
    Vector<T, N> v;
    for (int i = 0; i < N; ++i) v[i] = pd[i];
    return v;
}

struct PyTransform {
    PyTransform() noexcept = default;
    PyTransform(RawTransformInfo transform_info) noexcept : transform_info{std::move(transform_info)} {}
    static PyTransform empty() noexcept { return PyTransform(); }

    static PyTransform matrix(const PyFloatArr &matrix) noexcept {
        auto vec = pyarray_to_vector<float>(matrix);
        auto arr = make_float4x4(1.f);
        for (auto row = 0u; row < 4u; ++row) {
            for (auto col = 0u; col < 4u; ++col) {
                arr[col][row] = vec[row * 4u + col];
            }
        }
        return PyTransform(std::move(RawTransformInfo::matrix(std::move(arr))));
    }
    
    static PyTransform srt(const PyFloatArr &translate, const PyFloatArr &rotate, const PyFloatArr &scale) noexcept {
        PyTransform transform;
        transform.transform_info.build_srt(
            pyarray_to_pack<float, 3>(translate),
            pyarray_to_pack<float, 4>(rotate),
            pyarray_to_pack<float, 3>(scale)
        );
        return transform;
    }

    static PyTransform view(const PyFloatArr &position, const PyFloatArr &look_at, const PyFloatArr &up) noexcept {
        PyTransform transform;
        auto pos = pyarray_to_pack<float, 3>(position);
        transform.transform_info.build_view(
            std::move(pos),
            normalize(pyarray_to_pack<float, 3>(look_at) - pos),
            pyarray_to_pack<float, 3>(up)
        );
        return transform;
    }

    RawTransformInfo transform_info;
};

struct PyTexture {
    PyTexture() noexcept {}
    PyTexture(RawTextureInfo texture_info) noexcept: texture_info{std::move(texture_info)} {}
    static PyTexture empty() noexcept { return PyTexture(); }

    static PyTexture color(const PyFloatArr &color) noexcept {
        PyTexture texture;
        auto c = pyarray_to_vector<float>(color);
        LUISA_ASSERT(c.size() == 3, "Invalid color channel");
        texture.texture_info.build_constant(std::move(c));
        return texture;
    }

    static PyTexture inline_image(std::string_view image, const PyFloatArr &scale) noexcept {
        PyTexture texture;
        texture.texture_info.build_image(
            luisa::string(image), pyarray_to_pack<float, 3>(scale)
        );
        return texture;
    }

    static PyTexture image(std::string_view image, const PyFloatArr &scale) noexcept {
        PyTexture texture;
        texture.texture_info.build_image(
            luisa::string(image), pyarray_to_pack<float, 3>(scale));
        return texture;
    }

    static PyTexture checker(PyTexture &on, PyTexture &off, float scale) noexcept {
        PyTexture texture;
        texture.texture_info.build_checker(
            std::move(on.texture_info),
            std::move(off.texture_info), scale);
        return texture;
    }

    RawTextureInfo texture_info;
};

struct PySurface {
    PySurface(luisa::string name, float roughness, float opacity) noexcept
        : surface_info{name, roughness, opacity} {}
    static PySurface metal(
        std::string_view name, float roughness, float opacity,
        PyTexture &kd, std::string_view eta) noexcept {
        PySurface surface(luisa::string(name), roughness, opacity);
        surface.surface_info.build_metal(std::move(kd.texture_info), luisa::string(eta));
        return surface;
    }

    static PySurface plastic(
        std::string_view name, float roughness, float opacity,
        PyTexture &kd, PyTexture &ks, float eta) noexcept {
        PySurface surface(luisa::string(name), roughness, opacity);
        surface.surface_info.build_plastic(
            std::move(kd.texture_info), std::move(ks.texture_info), eta);
        return surface;
    }

    static PySurface glass(
        std::string_view name, float roughness, float opacity,
        PyTexture &ks, PyTexture &kt, float eta) noexcept {
        PySurface surface(luisa::string(name), roughness, opacity);
        surface.surface_info.build_glass(
            std::move(ks.texture_info), std::move(kt.texture_info), eta);
        return surface;
    }

    RawSurfaceInfo surface_info;
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

// RawMeshInfo get_mesh_from_dict(py::dict mesh_info, const SceneDesc *scene_desc) {
//     luisa::string surface_name = get_string_from_dict(mesh_info, "surface");
//     luisa::string light_name = get_string_from_dict(mesh_info, "light");
//     luisa::string medium_name = get_string_from_dict(mesh_info, "medium");

// }

void init(std::string_view context_path, uint cuda_device, uint scene_index, LogLevel log_level) noexcept {
    /* add device */
    context_storage = context_path;
    switch (log_level) {
        case VERBOSE: log_level_verbose(); break; 
        case INFO: log_level_info(); break;
        case WARNING: log_level_warning(); break;
    }
    context = luisa::make_unique<Context>(context_storage);
    
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
    scene = Scene::create(*context, desc, log_level != LogLevel::WARNING);
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
    std::string_view name, PyTexture &texture, PyTransform &transform
) noexcept {
    auto environment_info = RawEnvironmentInfo {
        luisa::string(name),
        std::move(texture.texture_info),
        std::move(transform.transform_info)
    };
    LUISA_INFO("Add: {}", environment_info.get_info());
    auto environment_node = scene->add_environment(environment_info);
}

void add_light(std::string_view name, PyTexture &texture) noexcept {
    auto light_info = RawLightInfo {
        luisa::string(name),
        std::move(texture.texture_info)
    };
    LUISA_INFO("Add: {}", light_info.get_info());
    auto light_node = scene->add_light(light_info);
}

void add_surface(const PySurface &surface) noexcept {
    // auto surface_info = RawSurfaceInfo {
    //     luisa::string(name),
    //     material,
    //     std::move(texture.texture_info),
    //     roughness
    // };
    LUISA_INFO("Add: {}", surface.surface_info.get_info());
    auto surface_node = scene->add_surface(surface.surface_info);
}

// void add_camera(
//     std::string_view name, PyTransform &origin_pose,
//     float fov, int spp, const PyIntArr &resolution
// ) noexcept {
//     auto camera_info = RawCameraInfo{
//         luisa::string(name),
//         std::move(origin_pose.transform_info),
//         fov, uint(spp), 
//         pyarray_to_pack<uint, 2>(resolution),
//         1.0
//     };
//     LUISA_INFO("Add: {}", camera_info.get_info());
//     uint camera_id = scene->cameras().size();
//     auto camera = scene->add_camera(camera_info);

//     uint pixel_count = camera_info.resolution.x * camera_info.resolution.y * 4;
//     // camera_storage.emplace(camera_info.name, CameraStorage(camera_id, device.get(), pixel_count));
//     camera_storage[camera_info.name] = luisa::make_unique<CameraStorage>(
//         camera_id, device.get(), pixel_count
//     );
// }

// void update_camera(std::string_view name, PyTransform &transform) noexcept {
//     // auto real_transform = RawTransformInfo {
//     //     transform.empty,
//     //     transform.transform,
//     //     transform.translate,
//     //     transform.rotate,
//     //     transform.scale
//     // };
//     LUISA_INFO("Update: Camera {}, {}", name, transform.transform_info.get_info());
//     auto camera = scene->update_camera(name, std::move(transform.transform_info));
// }


void update_camera(
    std::string_view name, PyTransform &origin_pose, PyTransform &transform,
    float fov, int spp, const PyIntArr &resolution
) noexcept {
    auto camera_info = RawCameraInfo{
        luisa::string(name),
        std::move(origin_pose.transform_info),
        std::move(transform.transform_info),
        fov, uint(spp),
        pyarray_to_pack<uint, 2>(resolution),
        1.0
    };

    // auto real_transform = RawTransformInfo {
    //     transform.empty,
    //     transform.transform,
    //     transform.translate,
    //     transform.rotate,
    //     transform.scale
    // };
    LUISA_INFO("Update: {}", camera_info.get_info());
    uint camera_id = scene->cameras().size();
    auto camera = scene->update_camera(camera_info);
    if (auto it = camera_storage.find(camera_info.name); it == camera_storage.end()) {
        uint pixel_count = camera_info.resolution.x * camera_info.resolution.y * 4;
        camera_storage[camera_info.name] = luisa::make_unique<CameraStorage>(camera_id, device.get(), pixel_count);
    }
}

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
        mesh_info.build_file(luisa::string(obj_path));
    } else {
        mesh_info.build_mesh(
            pyarray_to_vector<float>(vertices),
            pyarray_to_vector<uint>(triangles),
            pyarray_to_vector<float>(normals),
            pyarray_to_vector<float>(uvs)
        );
    }
    LUISA_INFO("Add: {}", mesh_info.get_info());
    auto shape = scene->update_shape(mesh_info, "mesh", true);
}

void update_rigid(
    std::string_view name, PyTransform &transform
) noexcept {
    auto mesh_info = RawShapeInfo(
        luisa::string(name),
        std::move(transform.transform_info),
        "", "", ""
    );
    LUISA_INFO("Update: {}", mesh_info.get_info());
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
    mesh_info.build_mesh(
        pyarray_to_vector<float>(vertices),
        pyarray_to_vector<uint>(triangles),
        pyarray_to_vector<float>(normals),
        pyarray_to_vector<float>(uvs)
    );
    LUISA_INFO("Update: {}", mesh_info.get_info());
    auto shape = scene->update_shape(mesh_info, "deformablemesh", false);
}

void add_ground(
    std::string_view name, float height, float range, const PyFloatArr &up_direction,
    std::string_view surface
) noexcept {
    static auto z = make_float3(0.f, 0.f, 1.f);
    auto up = normalize(pyarray_to_pack<float, 3>(up_direction));
    auto plane_info = RawShapeInfo(
        luisa::string(name),
        RawTransformInfo::srt(
            height * up,
            make_float4(cross(z, up), degrees(acos(dot(z, up)))),
            make_float3(range)
        ),
        luisa::string(surface), "", ""
    );
    plane_info.build_plane();
    LUISA_INFO("Add: {}", plane_info.get_info());
    auto shape = scene->update_shape(plane_info, "plane", true);
}

void update_particles(
    std::string_view name, const PyFloatArr &vertices, float radius,
    std::string_view surface, std::string_view light
) noexcept {
    auto spheres_info = RawShapeInfo(
        luisa::string(name), RawTransformInfo(),
        luisa::string(surface), luisa::string(light), ""
    );
    spheres_info.build_spheres(pyarray_to_vector<float>(vertices), std::move(radius), 0u);
    LUISA_INFO("Update: {}", spheres_info.get_info());
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
    return int_buffer_handle;
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
    return array_buffer;
}

void destroy() {}

PYBIND11_MODULE(LuisaRenderPy, m) {
    m.doc() = "Python binding of LuisaRender";
    // py::enum_<RawSurfaceInfo::RawMaterial>(m, "Material")
    //     .value("METAL", RawSurfaceInfo::RawMaterial::RAW_METAL)
    //     .value("SUBSTRATE", RawSurfaceInfo::RawMaterial::RAW_SUBSTRATE)
    //     .value("MATTE", RawSurfaceInfo::RawMaterial::RAW_MATTE)
    //     .value("GLASS", RawSurfaceInfo::RawMaterial::RAW_GLASS)
    //     .value("NULL", RawSurfaceInfo::RawMaterial::RAW_NULL);
    py::enum_<LogLevel>(m, "LogLevel")
        .value("DEBUG", LogLevel::VERBOSE)
        .value("INFO", LogLevel::INFO)
        .value("WARNING", LogLevel::WARNING);
    py::class_<PyTransform>(m, "Transform")
        .def_static("empty", &PyTransform::empty)
        .def_static("matrix", &PyTransform::matrix, py::arg("matrix"))
        .def_static("srt", &PyTransform::srt,
            py::arg("translate"),
            py::arg("rotate"),
            py::arg("scale")
        )
        .def_static("view", &PyTransform::view,
            py::arg("position"),
            py::arg("look_at"),
            py::arg("up")
        );
    py::class_<PyTexture>(m, "Texture")
        .def_static("empty", &PyTexture::empty)
        .def_static("image", &PyTexture::image, py::arg("image"), py::arg("scale"))
        .def_static("color", &PyTexture::color, py::arg("color"))
        .def_static("checker", &PyTexture::checker,
            py::arg("on"), py::arg("off"), py::arg("scale")
        );        
    py::class_<PySurface>(m, "Surface")
        .def_static("metal", &PySurface::metal,
            py::arg("name"), py::arg("roughness"), py::arg("opacity") = 1.f,
            py::arg("kd"),
            py::arg("eta") = "Al"
        )
        .def_static("plastic", &PySurface::plastic,
            py::arg("name"), py::arg("roughness"), py::arg("opacity") = 1.f,
            py::arg("kd"),
            py::arg("ks"),
            py::arg("eta") = 1.5
        )
        .def_static("glass", &PySurface::glass,
            py::arg("name"), py::arg("roughness"), py::arg("opacity") = 1.f,
            py::arg("ks"),
            py::arg("kt"),
            py::arg("eta") = 1.5
        );

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
        py::arg("surface")
    );
    // m.def("add_camera", &add_camera,
    //     py::arg("name"),
    //     py::arg("origin_pose"),
    //     py::arg("fov"), py::arg("spp"), py::arg("resolution")
    // );
    m.def("update_camera", &update_camera,
        py::arg("name"),
        py::arg("origin_pose"),
        py::arg("transform"),
        py::arg("fov"),
        py::arg("spp"),
        py::arg("resolution")
    );
    m.def("add_ground", &add_ground,
        py::arg("name"),
        py::arg("height"),
        py::arg("range"),
        py::arg("up_direction"),
        py::arg("surface") = ""
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