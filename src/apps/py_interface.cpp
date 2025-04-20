#include <apps/app_base.h>
#include <apps/py_scene.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

luisa::unique_ptr<Stream> stream_ptr = nullptr;
luisa::unique_ptr<Device> device_ptr = nullptr;
luisa::unique_ptr<Context> context_ptr = nullptr;
luisa::vector<PyScene> scenes;
std::mutex mutex;

template <typename T>
py::array_t<T> get_default_array(const luisa::vector<T> &a) {
    auto buffer_info = py::buffer_info{
        (void *)a.data(), sizeof(T), py::format_descriptor<T>::format(),
        1, {a.size()}, {sizeof(T)}
    };
    return py::array_t<T>(buffer_info);
}

void init(
    std::string_view context_path, std::string_view context_id,
    std::string_view backend, uint device_index, LogLevel log_level
) noexcept {
    /* add device */
    switch (log_level) {
        case VERBOSE: log_level_verbose(); break; 
        case INFO: log_level_info(); break;
        case WARNING: log_level_warning(); break;
    }
    context_ptr = luisa::make_unique<Context>(luisa::string(context_path), context_id);
    LUISA_INFO("Hardware concurrency: {}", std::thread::hardware_concurrency());
    compute::DeviceConfig config;
    config.device_index = device_index;     // Please ensure that cuda:cuda_device has enough space
    device_ptr = luisa::make_unique<Device>(context_ptr->create_device(backend, &config));
    stream_ptr = luisa::make_unique<Stream>(device_ptr->create_stream(StreamTag::COMPUTE));
}

PyScene *create_scene() noexcept {
    std::scoped_lock lock{mutex};
    scenes.emplace_back(*device_ptr, *context_ptr, *stream_ptr);
    LUISA_INFO("Luisa Scene created: {}", scenes.size());
    return &scenes.back();
}

void destroy() noexcept {
    std::scoped_lock lock{mutex};
    if (stream_ptr) {
        stream_ptr->synchronize();
    }
    scenes.clear();
    stream_ptr = nullptr;
    device_ptr = nullptr;
    context_ptr = nullptr;
    LUISA_INFO("Luisa environment destroyed.");
}

PYBIND11_MODULE(LuisaRenderPy, m) {
    m.doc() = "Python binding of LuisaRender";

    py::enum_<LogLevel>(m, "LogLevel")
        .value("DEBUG", LogLevel::VERBOSE)
        .value("INFO", LogLevel::INFO)
        .value("WARNING", LogLevel::WARNING);

    // Transform
    py::class_<PyTransform>(m, "Transform");
    py::class_<PyMatrix, PyTransform>(m, "MatrixTransform")
        .def(py::init<const PyDoubleArr&>(),
            py::arg("matrix"))
        .def("update", &PyMatrix::update,
            py::arg("matrix"));
    py::class_<PySRT, PyTransform>(m, "SRTTransform")
        .def(py::init<const PyDoubleArr&, const PyDoubleArr&, const PyDoubleArr&>(),
            py::arg("translate"),
            py::arg("rotate"),
            py::arg("scale"))
        .def("update", &PySRT::update,
            py::arg("translate"),
            py::arg("rotate"),
            py::arg("scale"));
    py::class_<PyView, PyTransform>(m, "ViewTransform")
        .def(py::init<const PyDoubleArr&, const PyDoubleArr&, const PyDoubleArr&>(),
            py::arg("position"),
            py::arg("front"),
            py::arg("up"))
        .def("update", &PyView::update,
            py::arg("position"),
            py::arg("front"),
            py::arg("up"));

    // Texture
    py::class_<PyTexture>(m, "Texture");
    py::class_<PyColor, PyTexture>(m, "ColorTexture")
        .def(py::init<const PyDoubleArr&>(),
            py::arg("color"));
    py::class_<PyImage, PyTexture>(m, "ImageTexture")
        .def(py::init<std::string_view, std::string_view, uint, uint, uint,
                      const PyDoubleArr&, std::string_view>(),
            py::arg("file") = "",
            py::arg("image_data") = "",
            py::arg("width") = 0u,
            py::arg("height") = 0u,
            py::arg("channel") = 0u,
            py::arg("scale") = PyDoubleArr(),
            py::arg("encoding").none(true) = py::none());
    py::class_<PyChecker, PyTexture>(m, "CheckerTexture")
        .def(py::init<PyTexture*, PyTexture*, float>(),
            py::arg("on").none(true) = py::none(),
            py::arg("off").none(true) = py::none(),
            py::arg("scale") = 1.0f);

    // Light
    py::class_<PyLight>(m, "Light")
        .def(py::init<std::string_view, PyTexture*, float, bool, float>(),
            py::arg("name"),
            py::arg("emission").none(true) = py::none(),
            py::arg("intensity") = 1.0f,
            py::arg("two_sided") = false,
            py::arg("beam_angle") = 180.0f);

    // Subsurface
    py::class_<PySubsurface>(m, "Subsurface");
    py::class_<PyUniformSubsurface, PySubsurface>(m, "UniformSubsurface")
        .def(py::init<std::string_view, PyTexture*>(),
            py::arg("name"),
            py::arg("thickness").none(true) = py::none());

    // Surface
    py::class_<PySurface>(m, "Surface");
    py::class_<PyMetalSurface, PySurface>(m, "MetalSurface")
        .def(py::init<std::string_view, PyTexture*, PyTexture*, PyTexture*, PyTexture*, std::string_view>(),
            py::arg("name"),
            py::arg("roughness").none(true) = py::none(),
            py::arg("opacity").none(true) = py::none(),
            py::arg("normal_map").none(true) = py::none(),
            py::arg("kd").none(true) = py::none(),
            py::arg("eta").none(true) = py::none());
    py::class_<PyPlasticSurface, PySurface>(m, "PlasticSurface")
        .def(py::init<std::string_view, PyTexture*, PyTexture*, PyTexture*, PyTexture*, PyTexture*, PyTexture*>(),
            py::arg("name"),
            py::arg("roughness").none(true) = py::none(),
            py::arg("opacity").none(true) = py::none(),
            py::arg("normal_map").none(true) = py::none(),
            py::arg("kd").none(true) = py::none(),
            py::arg("ks").none(true) = py::none(),
            py::arg("eta").none(true) = py::none());
    py::class_<PyGlassSurface, PySurface>(m, "GlassSurface")
        .def(py::init<std::string_view, PyTexture*, PyTexture*, PyTexture*, PyTexture*, PyTexture*, PyTexture*>(),
            py::arg("name"),
            py::arg("roughness").none(true) = py::none(),
            py::arg("opacity").none(true) = py::none(),
            py::arg("normal_map").none(true) = py::none(),
            py::arg("ks").none(true) = py::none(),
            py::arg("kt").none(true) = py::none(),
            py::arg("eta").none(true) = py::none());
    py::class_<PyDisneySurface, PySurface>(m, "DisneySurface")
        .def(py::init<std::string_view, PyTexture*, PyTexture*, PyTexture*,
                     PyTexture*, PyTexture*, PyTexture*, PyTexture*, PyTexture*, PyTexture*>(),
            py::arg("name"),
            py::arg("roughness").none(true) = py::none(),
            py::arg("opacity").none(true) = py::none(),
            py::arg("normal_map").none(true) = py::none(),
            py::arg("kd").none(true) = py::none(),
            py::arg("eta").none(true) = py::none(),
            py::arg("metallic").none(true) = py::none(),
            py::arg("specular_tint").none(true) = py::none(),
            py::arg("specular_trans").none(true) = py::none(),
            py::arg("diffuse_trans").none(true) = py::none());
    py::class_<PyLayeredSurface, PySurface>(m, "LayeredSurface")
        .def(py::init<std::string_view, PySurface*, PySurface*, PyTexture*>(),
            py::arg("name"),
            py::arg("top").none(true) = py::none(),
            py::arg("bottom").none(true) = py::none(),
            py::arg("thickness").none(true) = py::none());

    // Shape
    py::class_<PyShape>(m, "Shape");
    py::class_<PyRigid, PyShape>(m, "RigidShape")
        .def(py::init<std::string_view, std::string_view,
            const PyDoubleArr&, const PyUIntArr&, const PyDoubleArr&, const PyDoubleArr&,
            PyTransform*, PySurface*, PyLight*, PySubsurface*, float>(),
            py::arg("name"),
            py::arg("obj_path") = "",
            py::arg("vertices") = PyDoubleArr(),
            py::arg("triangles") = PyUIntArr(),
            py::arg("normals") = PyDoubleArr(),
            py::arg("uvs") = PyDoubleArr(),
            py::arg("transform").none(true) = py::none(),
            py::arg("surface").none(true) = py::none(),
            py::arg("emission").none(true) = py::none(),
            py::arg("subsurface").none(true) = py::none(),
            py::arg("clamp_normal") = 180.f)
        .def("update", &PyRigid::update,
            py::arg("transform").none(false) = py::none());
    py::class_<PyDeformable, PyShape>(m, "DeformableShape")
        .def(py::init<std::string_view,
            const PyDoubleArr&, const PyUIntArr&, const PyDoubleArr&, const PyDoubleArr&,
            PySurface*, PyLight*, PySubsurface*, float>(),
            py::arg("name"),
            py::arg("vertices") = PyDoubleArr(),
            py::arg("triangles") = PyUIntArr(),
            py::arg("normals") = PyDoubleArr(),
            py::arg("uvs") = PyDoubleArr(),
            py::arg("surface").none(true) = py::none(),
            py::arg("emission").none(true) = py::none(),
            py::arg("subsurface").none(true) = py::none(),
            py::arg("clamp_normal") = 180.f)
        .def("update", &PyDeformable::update,
            py::arg("vertices"),
            py::arg("triangles"),
            py::arg("normals") = PyDoubleArr(),
            py::arg("uvs") = PyDoubleArr());
    py::class_<PyParticles, PyShape>(m, "ParticlesShape")
        .def(py::init<std::string_view,
            const PyDoubleArr&, const PyDoubleArr&, uint,
            PySurface*, PyLight*, PySubsurface*, float>(),
            py::arg("name"),
            py::arg("centers") = PyDoubleArr(),
            py::arg("radii") = PyDoubleArr(),
            py::arg("subdivision") = 0u,
            py::arg("surface").none(true) = py::none(),
            py::arg("emission").none(true) = py::none(),
            py::arg("subsurface").none(true) = py::none(),
            py::arg("clamp_normal") = 180.f)
        .def("update", &PyParticles::update,
            py::arg("centers") = PyDoubleArr(),
            py::arg("radii") = PyDoubleArr());

    // Film
    py::class_<PyFilm>(m, "Film")
        .def(py::init<const PyUIntArr&>(),
            py::arg("resolution"));

    // Filter
    py::class_<PyFilter>(m, "Filter")
        .def(py::init<float>(),
            py::arg("radius") = 1.0f)
        .def("update", &PyFilter::update,
            py::arg("radius") = 1.0f);

    // Camera
    py::class_<PyCamera>(m, "Camera");
    py::class_<PyPinhole, PyCamera>(m, "PinholeCamera")
        .def(py::init<std::string_view,
            PyTransform*, PyFilm *, PyFilter*, uint, float>(),
            py::arg("name"),
            py::arg("pose").none(true) = py::none(),
            py::arg("film").none(false),
            py::arg("filter").none(true) = py::none(),
            py::arg("spp"),
            py::arg("fov"))
        .def("update", &PyPinhole::update,
            py::arg("pose").none(true) = py::none(),
            py::arg("fov"));
    py::class_<PyThinLens, PyCamera>(m, "ThinLensCamera")
        .def(py::init<std::string_view,
            PyTransform*, PyFilm *, PyFilter*, uint, float, float, float>(),
            py::arg("name"),
            py::arg("pose").none(true) = py::none(),
            py::arg("film").none(false),
            py::arg("filter").none(true) = py::none(),
            py::arg("spp"),
            py::arg("aperture"),
            py::arg("focal_len"),
            py::arg("focus_dis"))
        .def("update", &PyThinLens::update,
            py::arg("pose").none(true) = py::none(),
            py::arg("aperture"),
            py::arg("focal_len"),
            py::arg("focus_dis"));

    // Environment
    py::class_<PyEnvironment>(m, "Environment")
        .def(py::init<std::string_view, PyTexture*, PyTransform*>(),
            py::arg("name"),
            py::arg("emission").none(true) = py::none(),
            py::arg("transform").none(true) = py::none());

    // Integrator
    py::class_<PyIntegrator>(m, "Integrator");
    py::class_<PyWavePath, PyIntegrator>(m, "WavePathIntegrator")
        .def(py::init<LogLevel, bool, uint, uint, float>(),
            py::arg("log_level") = LogLevel::WARNING,
            py::arg("enable_cache") = true,
            py::arg("max_depth") = 32u,
            py::arg("rr_depth") = 0u,
            py::arg("rr_threshold") = 0.95f);
    py::class_<PyWavePathV2, PyIntegrator>(m, "WavePathV2Integrator")
        .def(py::init<LogLevel, bool, uint, uint, float, uint>(),
            py::arg("log_level") = LogLevel::WARNING,
            py::arg("enable_cache") = true,
            py::arg("max_depth") = 32u,
            py::arg("rr_depth") = 0u,
            py::arg("rr_threshold") = 0.95f,
            py::arg("state_limit") =  512u * 512u * 32u);

    // Spectrum
    py::class_<PySpectrum>(m, "Spectrum");
    py::class_<PyHero, PySpectrum>(m, "HeroSpectrum")
        .def(py::init<uint>(), py::arg("dimension") = 4u);
    py::class_<PySRGB, PySpectrum>(m, "SRGBSpectrum")
        .def(py::init<>());

    // Render
    py::class_<PyRender>(m, "Render")
        .def(py::init<std::string_view, PySpectrum*, PyIntegrator*, float>(),
            py::arg("name"),
            py::arg("spectrum"),
            py::arg("integrator"),
            py::arg("clamp_normal") = 180.f);

    py::class_<PyScene>(m, "Scene")
        .def("init", &PyScene::init,
            py::arg("render").none(false))
        .def("update_environment", &PyScene::update_environment,
            py::arg("environment").none(false))
        .def("update_emission", &PyScene::update_emission,
            py::arg("light").none(false))
        .def("update_subsurface", &PyScene::update_subsurface,
            py::arg("subsurface").none(false))
        .def("update_surface", &PyScene::update_surface,
            py::arg("surface").none(false))
        .def("update_shape", &PyScene::update_shape,
            py::arg("shape").none(false))
        .def("update_camera", &PyScene::update_camera,
            py::arg("camera").none(false),
            py::arg("denoise"))
        .def("update_scene", &PyScene::update_scene,
            py::arg("time"))
        .def("render_frame", &PyScene::render_frame,
            py::arg("camera").none(false));

    m.def("init", &init,
        py::arg("context_path"),
        py::arg("context_id") = "",
        py::arg("backend") = "cuda",
        py::arg("device_index") = 0u,
        py::arg("log_level") = LogLevel::WARNING
    );
    m.def("create_scene", &create_scene, py::return_value_policy::reference);
    m.def("destroy", &destroy);
}