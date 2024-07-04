#include <span>
#include <iostream>
#include <vector>
#include <string>

#include <luisa/backends/ext/denoiser_ext.h>
#include <base/scene.h>
#include <base/pipeline.h>
#include <apps/app_base.h>
#include <apps/py_class.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

struct CameraStorage {
    CameraStorage(uint index, Device* device, uint pixel_count) noexcept:
        index{index},
        color_buffer{device->create_buffer<float4>(pixel_count)},
        denoised_buffer{device->create_buffer<float4>(pixel_count)} {} 
    uint index;
    Buffer<float4> color_buffer;
    Buffer<float4> denoised_buffer;
};

luisa::unique_ptr<Stream> stream;
luisa::unique_ptr<Device> device;
luisa::unique_ptr<Context> context;
DenoiserExt *denoiser_ext = nullptr;

luisa::unique_ptr<Pipeline> pipeline;
luisa::unique_ptr<Scene> scene;
luisa::unordered_map<luisa::string, luisa::unique_ptr<CameraStorage>> camera_storage;
luisa::string context_storage;

void init(
    std::string_view context_path, uint cuda_device, LogLevel log_level,
    const PyIntegrator &integrator_options, const PySpectrum &spectrum_options,
    float clamp_normal
) noexcept {
    /* add device */
    context_storage = context_path;
    Clock clock;
    switch (log_level) {
        case VERBOSE: log_level_verbose(); break; 
        case INFO: log_level_info(); break;
        case WARNING: log_level_warning(); break;
    }
    context = luisa::make_unique<Context>(context_storage);
    
    luisa::string backend = "CUDA";
    compute::DeviceConfig config;
    config.device_index = cuda_device;
    /* Please ensure that cuda:cuda_device has enough space */
    device = luisa::make_unique<Device>(context->create_device(backend, &config));

    /* build denoiser */
    stream = luisa::make_unique<Stream>(device->create_stream(StreamTag::COMPUTE));
    denoiser_ext = device->extension<DenoiserExt>();

    /* build scene and pipeline */
    auto scene_info = RawSceneInfo {
        integrator_options.integrator_info,
        spectrum_options.spectrum_info,
        clamp_normal
    };
    scene = Scene::create(*context, scene_info);
    auto scene_create_time = clock.toc();
    LUISA_INFO("Scene created in {} ms.", scene_create_time);

    pipeline = Pipeline::create(*device, *stream, *scene);
    auto pipeline_create_time = clock.toc();
    LUISA_INFO("Pipeline created in {} ms.", pipeline_create_time - scene_create_time);
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

void add_emission(std::string_view name, PyTexture &texture) noexcept {
    auto light_info = RawLightInfo {
        luisa::string(name),
        std::move(texture.texture_info)
    };
    LUISA_INFO("Add: {}", light_info.get_info());
    auto light_node = scene->add_light(light_info);
}

void add_surface(const PySurface &surface) noexcept {
    LUISA_INFO("Add: {}", surface.surface_info.get_info());
    auto surface_node = scene->add_surface(surface.surface_info);
}

void update_camera(const PyCamera &camera) noexcept {
    const auto &camera_info = camera.camera_info;
    LUISA_INFO("Update: {}", camera_info.get_info());
    uint camera_id = scene->cameras().size();
    auto camera_node = scene->update_camera(camera_info);
    if (auto it = camera_storage.find(camera_info.name); it == camera_storage.end()) {
        uint pixel_count = camera_info.film_info.resolution.x * camera_info.film_info.resolution.y;
        camera_storage[camera_info.name] = luisa::make_unique<CameraStorage>(camera_id, device.get(), pixel_count);
    }
}

void update_shape(const PyShape &shape) noexcept {
    LUISA_INFO("Update: {}", shape.shape_info.get_info());
    auto shape_node = scene->update_shape(shape.shape_info);
}

PyFloatArr render_frame(
    std::string_view name, std::string_view path,
    bool denoise, bool save_picture, bool render_png
) noexcept {
    Clock clock;
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
    luisa::vector<float4> buffer;
    pipeline->render_to_buffer(*stream, idx, buffer);
    auto buffer_p = reinterpret_cast<float *>(buffer.data());
    stream->synchronize();

    /* denoise image */
    if (denoise) {
        if (save_picture) {
            std::filesystem::path origin_path(exr_path);
            origin_path.replace_filename(origin_path.stem().string() + "_ori" + origin_path.extension().string());
            save_image(origin_path, buffer_p, resolution);
        }

        // auto color_buffer = device.create_buffer<float4>(pixel_count);
        // auto output_buffer = device.create_buffer<float4>(pixel_count);
        auto &color_buffer = camera_store->color_buffer;
        auto &denoised_buffer = camera_store->denoised_buffer;
        auto denoiser = denoiser_ext->create(*stream);
        {
            auto input = DenoiserExt::DenoiserInput{resolution.x, resolution.y};
            input.push_noisy_image(
                color_buffer.view(), denoised_buffer.view(),
                DenoiserExt::ImageFormat::FLOAT3,
                DenoiserExt::ImageColorSpace::HDR
            );
            input.noisy_features = false;
            input.filter_quality = DenoiserExt::FilterQuality::DEFAULT;
            input.prefilter_mode = DenoiserExt::PrefilterMode::NONE;
            denoiser->init(input);
        }
        (*stream) << color_buffer.copy_from(buffer_p) << synchronize();
        denoiser->execute(true);
        (*stream) << denoised_buffer.copy_to(buffer_p) << synchronize();
        LUISA_INFO("Start denoising...");
    }

    /* save image */
    if (save_picture) save_image(exr_path, buffer_p, resolution);

    apply_gamma(buffer_p, resolution);

    if (save_picture && render_png) {
        std::filesystem::path png_path = path;
        png_path.replace_extension(".png");
        auto int_buffer = convert_to_int_pixel(buffer_p, resolution);
        save_image(png_path, (*int_buffer).data(), resolution);
    }

    auto array_buffer = PyFloatArr(resolution.x * resolution.y * 4);
    std::memcpy(array_buffer.mutable_data(), buffer_p, array_buffer.size() * sizeof(float));
    return array_buffer;
}

void destroy() {}

PYBIND11_MODULE(LuisaRenderPy, m) {
    m.doc() = "Python binding of LuisaRender";
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
        .def_static("image", &PyTexture::image,
            py::arg("image") = "",
            py::arg("scale") = PyFloatArr(),
            py::arg("image_data") = PyFloatArr()
        )
        .def_static("color", &PyTexture::color, py::arg("color"))
        .def_static("checker", &PyTexture::checker,
            py::arg("on"), py::arg("off"), py::arg("scale")
        );        
    py::class_<PySurface>(m, "Surface")
        .def_static("metal", &PySurface::metal,
            py::arg("name"),
            py::arg("roughness"),
            py::arg("opacity") = 1.f,
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
    py::class_<PyShape>(m, "Shape")
        .def_static("rigid_from_file", &PyShape::rigid_from_file,
            py::arg("name"),
            py::arg("obj_path"),
            py::arg("surface") = "",
            py::arg("emission") = "",
            py::arg("clamp_normal") = -1.f
        )
        .def_static("rigid_from_mesh", &PyShape::rigid_from_mesh,
            py::arg("name"),
            py::arg("vertices"),
            py::arg("triangles"),
            py::arg("normals") = PyFloatArr(),
            py::arg("uvs") = PyFloatArr(),
            py::arg("surface") = "",
            py::arg("emission") = "",
            py::arg("clamp_normal") = -1.f
        )
        .def_static("deformable", &PyShape::deformable,
            py::arg("name"),
            py::arg("surface") = "",
            py::arg("emission") = "",
            py::arg("clamp_normal") = -1.f
        )
        .def_static("particles", &PyShape::particles,
            py::arg("name"),
            py::arg("radius"),
            py::arg("subdivision") = 0u,
            py::arg("surface") = "",
            py::arg("emission") = ""
        )
        .def_static("plane", &PyShape::plane,
            py::arg("name"),
            py::arg("height"),
            py::arg("range"),
            py::arg("up_direction"),
            py::arg("subdivision") = 0u,
            py::arg("surface") = "",
            py::arg("emission") = ""
        )
        .def("update_rigid", &PyShape::update_rigid,
            py::arg("transform")
        )
        .def("update_deformable", &PyShape::update_deformable,
            py::arg("vertices"),
            py::arg("triangles"),
            py::arg("normals") = PyFloatArr(),
            py::arg("uvs") = PyFloatArr()
        )
        .def("update_particles", &PyShape::update_particles,
            py::arg("vertices")
        );
    py::class_<PyCamera>(m, "Camera")
        .def_static("pinhole", &PyCamera::pinhole,
            py::arg("name"), py::arg("spp"), py::arg("resolution")
        )
        .def_static("thinlens", &PyCamera::thinlens,
            py::arg("name"), py::arg("spp"), py::arg("resolution")
        )
        .def("update_pinhole", &PyCamera::update_pinhole,
            py::arg("pose"),
            py::arg("fov")
        )
        .def("update_thinlens", &PyCamera::update_thinlens,
            py::arg("pose"),
            py::arg("aperture"),
            py::arg("focal_length"),
            py::arg("focus_distance")
        );
    py::class_<PyIntegrator>(m, "Integrator")
        .def_static("wave_path", &PyIntegrator::wave_path,
            py::arg("log_level"), 
            py::arg("wave_path_version") = 2u,
            py::arg("max_depth") = 32u,
            py::arg("state_limit") = 512u * 512u * 32u
        );
    py::class_<PySpectrum>(m, "Spectrum")
        .def_static("srgb", &PySpectrum::srgb)
        .def_static("hero", &PySpectrum::hero, py::arg("dimension") = 4u);

    m.def("init", &init,
        py::arg("context_path"),
        py::arg("cuda_device") = 0u,
        py::arg("log_level") = LogLevel::WARNING,
        py::arg("integrator_options") = PyIntegrator::wave_path(
            LogLevel::WARNING, 2u, 32u, 512u * 512u * 32u
        ),
        py::arg("spectrum_options") = PySpectrum::hero(4u),
        py::arg("clamp_normal") = 0.f
    );
    m.def("destroy", &destroy);

    m.def("add_environment", &add_environment,
        py::arg("name"),
        py::arg("texture") = PyTexture::empty(),
        py::arg("transform") = PyTransform::empty()
    );
    m.def("add_emission", &add_emission,
        py::arg("name"),
        py::arg("texture") = PyTexture::empty()
    );
    m.def("add_surface", &add_surface,
        py::arg("surface")
    );
    m.def("update_camera", &update_camera,
        py::arg("camera")
    );
    m.def("update_shape", &update_shape,
        py::arg("shape")
    );
    m.def("render_frame", &render_frame,
        py::arg("name"),
        py::arg("path") = "",
        py::arg("denoise") = true,
        py::arg("save_picture") = false,
        py::arg("render_png") = true
    );
}