#include <span>
#include <iostream>
#include <vector>
#include <string>

#include <luisa/core/stl/format.h>
#include <luisa/core/basic_types.h>
#include <sdl/scene_desc.h>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>


using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

namespace py = pybind11;
using namespace py::literals;
using PyFloatArr = py::array_t<float>;
using PyDoubleArr = py::array_t<double>;
using PyUIntArr = py::array_t<uint>;

template <typename T1, typename T2>
luisa::vector<T2> pyarray_to_vector(const py::array_t<T1> &array) noexcept {
    auto pd = array.data();
    luisa::vector<T2> v(pd, pd + array.size());
    return v;
}
template <typename T>
luisa::vector<T> pyarray_to_vector(const py::array_t<T> &array) noexcept {
    auto pd = array.data();
    luisa::vector<T> v(pd, pd + array.size());
    return v;
}

enum LogLevel: uint { VERBOSE, INFO, WARNING };

class PyDesc {
public:
    using ref_pair = std::pair<luisa::string, luisa::string>;
    struct DefineCache {
        DefineCache(luisa::string_view name, SceneNodeTag tag, luisa::string_view impl_type) noexcept:
            node{luisa::make_unique<SceneNodeDesc>(luisa::string(name), tag)},
            name{luisa::string(name)},
            impl_type{luisa::string(impl_type)} {
        }
        luisa::unique_ptr<SceneNodeDesc> node;
        luisa::string name, impl_type;
    };
    struct ReferCache {
        ReferCache(const SceneNodeDesc *node, luisa::string_view property_name, const SceneNodeDesc *property_node) noexcept:
            node{node}, property_node{property_node},
            property_name{luisa::string(property_name)} {    
        }
        const SceneNodeDesc *node, *property_node;
        luisa::string property_name;
    };

    PyDesc(std::string_view name, SceneNodeTag tag, std::string_view impl_type) noexcept {
        _define_cache.emplace_back(name, tag, impl_type);
        _node = _define_cache.back().node.get();
    }
    
    [[nodiscard]] auto node() const noexcept { return _node; }
    void clear_cache() noexcept { _define_cache.clear(); }
    void move_property_cache(PyDesc *property, luisa::string_view property_name) noexcept {
        bool has_name = !_node->identifier().empty();
        for (auto &c: property->_define_cache) {
            if (c.node->identifier().empty()) {
                c.name = luisa::format("{}.{}", property_name, c.name);
                if (has_name) {
                    c.name = luisa::format("{}.{}:{}", _node->identifier(), c.name, c.impl_type);
                    c.node->set_identifier(c.name);
                }
            }
            _define_cache.emplace_back(std::move(c));
        }
        property->_define_cache.clear();

        for (auto &c: property->_refer_cache) {
            _refer_cache.emplace_back(std::move(c));
        }
        property->_refer_cache.clear();
    }
    void add_property_node(luisa::string_view name, PyDesc *property) noexcept {
        if (property) {
            add_reference(name, property);
            move_property_cache(property, name);
        }
    }
    void add_reference(luisa::string_view name, PyDesc *property) noexcept {
        if (property) [[likely]] {
            _refer_cache.emplace_back(_node, name, property->node());
        }
    }
    void define_in_scene(SceneDesc *scene_desc) noexcept {
        luisa::vector<luisa::string> property_names;
        for (auto &c: _refer_cache) {
            property_names.emplace_back(luisa::string(c.property_node->identifier()));
        }

        for (auto i = _define_cache.size(); i-- > 0; ) {
            auto &c = _define_cache[i];
            auto node = scene_desc->define(std::move(c.node), c.impl_type);
        }
        _define_cache.clear();

        for (auto i = _refer_cache.size(); i-- > 0; ) {
            auto &c = _refer_cache[i];
            auto node = scene_desc->node(c.node->identifier());
            node->add_property(c.property_name, scene_desc->reference(property_names[i]));
        }
        _refer_cache.clear();
    }
    
protected:
    SceneNodeDesc *_node;
    luisa::vector<DefineCache> _define_cache;
    luisa::vector<ReferCache> _refer_cache;
};


// Transform
class PyTransform: public PyDesc {
public:
    PyTransform(std::string_view impl_type) noexcept:
        PyDesc{"", SceneNodeTag::TRANSFORM, impl_type} { }
};

class PyMatrix: public PyTransform {
public:
    PyMatrix(const PyDoubleArr &matrix) noexcept:
        PyTransform{"matrix"} {
        _node->add_property("m", pyarray_to_vector<double>(matrix));
    }
    void update(const PyDoubleArr &matrix) noexcept {
        _node->add_property("m", pyarray_to_vector<double>(matrix));
    }
};

class PySRT: public PyTransform {
public:
    PySRT(const PyDoubleArr &translate, const PyDoubleArr &rotate, const PyDoubleArr &scale) noexcept:
        PyTransform{"srt"} {
        _node->add_property("translate", pyarray_to_vector<double>(translate));
        _node->add_property("rotate", pyarray_to_vector<double>(rotate));
        _node->add_property("scale", pyarray_to_vector<double>(scale));
    }
    void update(const PyDoubleArr &translate, const PyDoubleArr &rotate, const PyDoubleArr &scale) noexcept {
        _node->add_property("translate", pyarray_to_vector<double>(translate));
        _node->add_property("rotate", pyarray_to_vector<double>(rotate));
        _node->add_property("scale", pyarray_to_vector<double>(scale));
    }
};

class PyView: public PyTransform {
public:
    PyView(const PyDoubleArr &position, const PyDoubleArr &front, const PyDoubleArr &up) noexcept:
        PyTransform{"view"} {
        _node->add_property("origin", pyarray_to_vector<double>(position));
        _node->add_property("front", pyarray_to_vector<double>(front));
        _node->add_property("up", pyarray_to_vector<double>(up));
    }
    void update(const PyDoubleArr &position, const PyDoubleArr &front, const PyDoubleArr &up) noexcept {
        _node->add_property("origin", pyarray_to_vector<double>(position));
        _node->add_property("front", pyarray_to_vector<double>(front));
        _node->add_property("up", pyarray_to_vector<double>(up));
    }
};


// Texture
class PyTexture: public PyDesc {
public:
    PyTexture(std::string_view impl_type) noexcept:
        PyDesc{"", SceneNodeTag::TEXTURE, impl_type} { }
};

class PyColor: public PyTexture {
public:
    PyColor(const PyDoubleArr &color) noexcept:
        PyTexture("constant") {
        _node->add_property("v", pyarray_to_vector<double>(color));
    }
};

class PyImage: public PyTexture {
public:
    PyImage(
        std::string_view file, const PyDoubleArr &image_data, const PyDoubleArr &scale
    ) noexcept: PyTexture{"image"} {
        if (file.empty() && !image_data.size() == 0) {
            uint channel;
            if (image_data.ndim() == 2) channel = 1;
            else if (image_data.ndim() == 3) channel = image_data.shape(2);
            else LUISA_ERROR_WITH_LOCATION("Invalid image dim!");
            _node->add_property("resolution", luisa::vector<double>{
                static_cast<double>(image_data.shape(1)),
                static_cast<double>(image_data.shape(0))
            });
            _node->add_property("channel", double(channel));
            _node->add_property("image_data", pyarray_to_vector<double>(image_data));
            _node->add_property("scale", pyarray_to_vector<double>(scale));
        } else if (!file.empty() && image_data.size() == 0) {
            _node->add_property("file", luisa::string(file));
            _node->add_property("scale", pyarray_to_vector<double>(scale));
        } else [[unlikely]] if (file.empty() && image_data.size() == 0)  {
            LUISA_ERROR_WITH_LOCATION("Cannot set both file image and inline image empty.");
        } else {
            LUISA_ERROR_WITH_LOCATION("Cannot set both file image and inline image.");
        }
    }
};

class PyChecker: public PyTexture {
public:
    PyChecker(PyTexture *on, PyTexture *off, float scale) noexcept:
        PyTexture{"checkerboard"} {
        add_property_node("on", on);
        add_property_node("off", off);
        _node->add_property("scale", scale);
    }
};


// Lights
class PyLight: public PyDesc {
public:
    PyLight(std::string_view name, PyTexture *emission) noexcept:
        PyDesc{name, SceneNodeTag::LIGHT, "diffuse"} {
        add_property_node("emission", emission);
    }
};


// Surface
class PySurface: public PyDesc {
public:
    PySurface(
        std::string_view name, std::string_view impl_type,
        PyTexture *roughness, PyTexture *opacity, PyTexture *normal_map
    ) noexcept: PyDesc{name, SceneNodeTag::SURFACE, impl_type} {
        add_property_node("roughness", roughness);
        add_property_node("opacity", opacity);
        add_property_node("normal_map", normal_map);
    }
};

class PyMetal: public PySurface {
public:
    PyMetal(
        std::string_view name,
        PyTexture *roughness, PyTexture *opacity, PyTexture *normal_map,
        PyTexture *kd, std::string_view eta
    ) noexcept: PySurface{name, "metal", roughness, opacity, normal_map} {
        add_property_node("Kd", kd);
        _node->add_property("eta", luisa::string(eta));
    }
};

class PyPlastic: public PySurface {
public:
    PyPlastic(
        std::string_view name,
        PyTexture *roughness, PyTexture *opacity, PyTexture *normal_map,
        PyTexture *kd, PyTexture *ks, PyTexture *eta
    ) noexcept: PySurface{name, "substrate", roughness, opacity, normal_map} {
        add_property_node("Kd", kd);
        add_property_node("Ks", ks);
        add_property_node("eta", eta);
    }
};

class PyGlass: public PySurface {
public:
    PyGlass(
        std::string_view name,
        PyTexture *roughness, PyTexture *opacity, PyTexture *normal_map,
        PyTexture *ks, PyTexture *kt, PyTexture *eta
    ) noexcept: PySurface{name, "glass", roughness, opacity, normal_map} {
        add_property_node("Ks", ks);
        add_property_node("Kt", kt);
        add_property_node("eta", eta);
    }
};


// Shape
class PyShape: public PyDesc {
public:
    PyShape(
        std::string_view name, std::string_view impl_type,
        PySurface *surface, PyLight *emission, float clamp_normal
    ) noexcept: PyDesc{name, SceneNodeTag::SHAPE, impl_type} {
        add_property_node("surface", surface);
        add_property_node("light", emission);
        _node->add_property("clamp_normal", clamp_normal);
    }
    bool loaded{false};
};

class PyRigid: public PyShape {
public:
    PyRigid(
        std::string_view name,
        std::string_view obj_path,
        const PyDoubleArr &vertices, const PyUIntArr &triangles,
        const PyDoubleArr &normals, const PyDoubleArr &uvs,
        PyTransform *transform,
        PySurface *surface, PyLight *emission, float clamp_normal
    ) noexcept: PyShape(name, "mesh", surface, emission, clamp_normal) {
        if (!obj_path.size() == 0 && vertices.size() == 0 && triangles.size() == 0) {   // file
            _node->add_property("file", luisa::string(obj_path));
        } else if (obj_path.size() == 0 && !vertices.size() == 0 && !triangles.size() == 0) {   // inline
            _node->add_property("positions", pyarray_to_vector<double>(vertices));
            _node->add_property("indices", pyarray_to_vector<uint, double>(triangles));
            _node->add_property("normals", pyarray_to_vector<double>(normals));
            _node->add_property("uvs", pyarray_to_vector<double>(uvs));
        } else [[unlikely]] if (obj_path.size() == 0 && vertices.size() == 0 && triangles.size() == 0) {
            LUISA_ERROR_WITH_LOCATION("Cannot set both file mesh and inline mesh empty.");
        } else {
            LUISA_ERROR_WITH_LOCATION("Cannot set both file mesh and inline mesh.");
        }
        add_property_node("transform", transform);
    }
    void update(PyTransform *transform) noexcept {
        add_property_node("transform", transform);
    }
};

class PyDeformable: public PyShape {
public:
    PyDeformable(
        std::string_view name,
        const PyDoubleArr &vertices, const PyUIntArr &triangles,
        const PyDoubleArr &normals, const PyDoubleArr &uvs,
        PySurface *surface, PyLight *emission, float clamp_normal
    ) noexcept: PyShape(name, "deformablemesh", surface, emission, clamp_normal) {
        _node->add_property("positions", pyarray_to_vector<double>(vertices));
        _node->add_property("indices", pyarray_to_vector<uint, double>(triangles));
        _node->add_property("normals", pyarray_to_vector<double>(normals));
        _node->add_property("uvs", pyarray_to_vector<double>(uvs));
    }
    void update(
        const PyDoubleArr &vertices, const PyUIntArr &triangles,
        const PyDoubleArr &normals, const PyDoubleArr &uvs
    ) noexcept {
        _node->add_property("positions", pyarray_to_vector<double>(vertices));
        _node->add_property("indices", pyarray_to_vector<uint, double>(triangles));
        _node->add_property("normals", pyarray_to_vector<double>(normals));
        _node->add_property("uvs", pyarray_to_vector<double>(uvs));
    }
};

class PyParticles: public PyShape {
public:
    PyParticles(
        std::string_view name,
        const PyDoubleArr &centers, const PyDoubleArr &radii, uint subdivision,
        PySurface *surface, PyLight *emission, float clamp_normal
    ) noexcept: PyShape(name, "spheregroup", surface, emission, clamp_normal) {
        _node->add_property("centers", pyarray_to_vector<double>(centers));
        _node->add_property("radii", pyarray_to_vector<double>(radii));
        _node->add_property("subdivision", double(subdivision));
    }
    void update(const PyDoubleArr &centers, const PyDoubleArr &radii) noexcept {
        _node->add_property("centers", pyarray_to_vector<double>(centers));
        _node->add_property("radii", pyarray_to_vector<double>(radii));
    }
};


// Film
class PyFilm: public PyDesc {
public:
    PyFilm(const PyUIntArr &resolution) noexcept:
        PyDesc{"", SceneNodeTag::FILM, "color"} {
        _node->add_property("resolution", pyarray_to_vector<uint, double>(resolution));
    }
};


// Filter
class PyFilter: public PyDesc {
public:
    PyFilter(float radius) noexcept:
        PyDesc{"", SceneNodeTag::FILTER, "gaussian"} {
        _node->add_property("radius", radius);
    }
    void update(float radius) noexcept {
        _node->add_property("radius", radius);
    }
};


// Camera
class PyCamera: public PyDesc {
public:
    PyCamera(
        std::string_view name, std::string_view impl_type,
        PyTransform *pose, PyFilm *film, PyFilter *filter, uint spp
    ) noexcept: PyDesc{name, SceneNodeTag::CAMERA, impl_type} {
        add_property_node("transform", pose);
        add_property_node("film", film);
        add_property_node("filter", filter);
        _node->add_property("spp", double(spp));
    }
    void update(PyTransform *pose) noexcept {
        add_property_node("transform", pose);
    }
    bool loaded{false};
    uint index{0};
    bool denoise{false};
    luisa::unique_ptr<Buffer<float4>> color_buffer;
    luisa::unique_ptr<Buffer<float4>> denoised_buffer;
};

class PyPinhole: public PyCamera {
public:
    PyPinhole(    
        std::string_view name,
        PyTransform *pose, PyFilm *film, PyFilter *filter, uint spp,
        float fov
    ) noexcept: PyCamera{name, "pinhole", pose, film, filter, spp} {
        _node->add_property("fov", fov);
    }
    void update(PyTransform *pose, float fov) noexcept {
        PyCamera::update(pose);
        _node->add_property("fov", fov);
    }
};

class PyThinLens: public PyCamera {
public:
    PyThinLens(
        std::string_view name,
        PyTransform *pose, PyFilm *film, PyFilter *filter, uint spp,
        float aperture, float focal_length, float focus_distance
    ) noexcept: PyCamera{name, "thinlens", pose, film, filter, spp} {
        _node->add_property("aperture", aperture);
        _node->add_property("focal_length", focal_length);
        _node->add_property("focus_distance", focus_distance);
    }
    void update(
        PyTransform *pose,
        float aperture, float focal_length, float focus_distance
    ) noexcept {
        PyCamera::update(pose);
        _node->add_property("aperture", aperture);
        _node->add_property("focal_length", focal_length);
        _node->add_property("focus_distance", focus_distance);
    }
};


// Environment
class PyEnvironment: public PyDesc {
public:
    PyEnvironment(
        std::string_view name,
        PyTexture *emission, PyTransform *transform
    ) noexcept: PyDesc{name, SceneNodeTag::ENVIRONMENT, "spherical"} {
        add_property_node("emission", emission);
        add_property_node("transform", transform);
    }    
};


// Light Sampler
class PyLightSampler: public PyDesc {
public:
    PyLightSampler() noexcept:
        PyDesc{"", SceneNodeTag::LIGHT_SAMPLER, "uniform"} { }
};


// Sampler
class PySampler: public PyDesc {
public:
    PySampler(std::string_view impl_type) noexcept:
        PyDesc{"", SceneNodeTag::SAMPLER, impl_type} { }
};

class PyIndependent: public PySampler {
public:
    PyIndependent() noexcept: PySampler{"independent"} { }
};

class PyPMJ02BN: public PySampler {
public:
    PyPMJ02BN() noexcept: PySampler{"pmj02bn"} { }
};


// Integrator
class PyIntegrator: public PyDesc {
public:
    // rr: Russian Roulette, a technique to control the average depth of ray tracing.
    PyIntegrator(std::string_view impl_type, LogLevel log_level, uint max_depth, uint rr_depth, float rr_threshold) noexcept:
        PyDesc{"", SceneNodeTag::INTEGRATOR, impl_type} {
        _node->add_property("use_progress", log_level != LogLevel::WARNING);
        _node->add_property("depth", double(max_depth));
        _node->add_property("rr_depth", double(rr_depth));
        _node->add_property("rr_threshold", rr_threshold);
    }
};

class PyWavePath: public PyIntegrator {
public:
    PyWavePath(LogLevel log_level, uint max_depth, uint rr_depth, float rr_threshold) noexcept:
        PyIntegrator{"wavepath", log_level, max_depth, rr_depth, rr_threshold} { }
};

class PyWavePathV2: public PyIntegrator {
public:
    PyWavePathV2(LogLevel log_level, uint max_depth, uint rr_depth, float rr_threshold, uint state_limit) noexcept:
        PyIntegrator{"wavepath_v2", log_level, max_depth, rr_depth, rr_threshold} {
        _node->add_property("state_limit", double(state_limit));
    }
};


// Spectrum
class PySpectrum: public PyDesc {
public:
    PySpectrum(std::string_view impl_type) noexcept:
        PyDesc{"", SceneNodeTag::SPECTRUM, impl_type} { }
};

class PyHero: public PySpectrum {
public:
    PyHero(uint dimension) noexcept:
        PySpectrum{"hero"} {
        _node->add_property("dimension", double(dimension));
    }
};

class PySRGB: public PySpectrum {
public:
    PySRGB() noexcept: PySpectrum{"srgb"} { }
};


// Root
class PyRender: public PyDesc {
public:
    PyRender(
        std::string_view name,
        PySpectrum *spectrum, PyIntegrator *integrator, float clamp_normal
    ) noexcept: PyDesc{name, SceneNodeTag::ROOT, SceneDesc::root_node_identifier} {
        add_property_node("spectrum", spectrum);
        add_property_node("integrator", integrator);
        _node->add_property("clamp_normal", clamp_normal);
    }
};