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
    virtual ~PyDesc() = default;
    
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
    virtual ~PyTransform() = default;
};

class PyMatrix final: public PyTransform {
public:
    PyMatrix(const PyDoubleArr &matrix) noexcept:
        PyTransform{"matrix"} {
        _node->add_property("m", pyarray_to_vector<double>(matrix));
    }
    ~PyMatrix() = default;
    void update(const PyDoubleArr &matrix) noexcept {
        _node->add_property("m", pyarray_to_vector<double>(matrix));
    }
};

class PySRT final: public PyTransform {
public:
    PySRT(const PyDoubleArr &translate, const PyDoubleArr &rotate, const PyDoubleArr &scale) noexcept:
        PyTransform{"srt"} {
        _node->add_property("translate", pyarray_to_vector<double>(translate));
        _node->add_property("rotate", pyarray_to_vector<double>(rotate));
        _node->add_property("scale", pyarray_to_vector<double>(scale));
    }
    ~PySRT() = default;
    void update(const PyDoubleArr &translate, const PyDoubleArr &rotate, const PyDoubleArr &scale) noexcept {
        _node->add_property("translate", pyarray_to_vector<double>(translate));
        _node->add_property("rotate", pyarray_to_vector<double>(rotate));
        _node->add_property("scale", pyarray_to_vector<double>(scale));
    }
};

class PyView final: public PyTransform {
public:
    PyView(const PyDoubleArr &position, const PyDoubleArr &front, const PyDoubleArr &up) noexcept:
        PyTransform{"view"} {
        _node->add_property("origin", pyarray_to_vector<double>(position));
        _node->add_property("front", pyarray_to_vector<double>(front));
        _node->add_property("up", pyarray_to_vector<double>(up));
    }
    ~PyView() = default;
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
    virtual ~PyTexture() = default;
};

class PyColor final: public PyTexture {
public:
    PyColor(const PyDoubleArr &color) noexcept:
        PyTexture("constant") {
        _node->add_property("v", pyarray_to_vector<double>(color));
    }
    ~PyColor() = default;
};

class PyImage final: public PyTexture {
public:
    PyImage(
        std::string_view file,
        std::string_view image_data, uint width, uint height, uint channel, 
        const PyDoubleArr &scale, std::string_view encoding
    ) noexcept: PyTexture{"image"} {
        if (file.empty() && !image_data.empty()) {
            _node->add_property("resolution", luisa::vector<double>{
                static_cast<double>(width), static_cast<double>(height)
            });
            _node->add_property("channel", double(channel));
            _node->add_property("image_data", luisa::string(image_data));
        } else if (!file.empty() && image_data.empty()) {
            _node->add_property("file", luisa::string(file));
        } else [[unlikely]] if (file.empty() && image_data.empty())  {
            LUISA_ERROR_WITH_LOCATION("Cannot set both file image and inline image empty.");
        } else {
            LUISA_ERROR_WITH_LOCATION("Cannot set both file image and inline image.");
        }
        auto scale_array = pyarray_to_vector<double>(scale);
        while (scale_array.size() < 3) scale_array.push_back(0.0f);
        _node->add_property("scale", scale_array);
        _node->add_property("encoding", luisa::string(encoding));
    }
    ~PyImage() = default;
};

class PyChecker final: public PyTexture {
public:
    PyChecker(PyTexture *on, PyTexture *off, float scale) noexcept:
        PyTexture{"checkerboard"} {
        add_property_node("on", on);
        add_property_node("off", off);
        _node->add_property("scale", scale);
    }
    ~PyChecker() = default;
};


// Lights
class PyLight: public PyDesc {
public:
    PyLight(
        std::string_view name, PyTexture *emission, float intensity,
        bool two_sided, float beam_angle
    ) noexcept: PyDesc{name, SceneNodeTag::LIGHT, "diffuse"} {
        add_property_node("emission", emission);
        _node->add_property("scale", intensity);
        _node->add_property("two_sided", JustBool(two_sided));
        _node->add_property("angle", beam_angle);
    }
    virtual ~PyLight() = default;
};


// Subsurface
class PySubsurface: public PyDesc {
public:
    PySubsurface(
        std::string_view name, std::string_view impl_type
    ) noexcept: PyDesc{name, SceneNodeTag::SUBSURFACE, impl_type} { }
    virtual ~PySubsurface() = default;
};

class PyUniformSubsurface final: public PySubsurface {
public:
    PyUniformSubsurface(
        std::string_view name, PyTexture *thickness
    ) noexcept: PySubsurface{name, "uniform"} {
        add_property_node("thickness", thickness);
    }
    ~PyUniformSubsurface() = default;
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
    virtual ~PySurface() = default;
};

class PyMetalSurface final: public PySurface {
public:
    PyMetalSurface(
        std::string_view name,
        PyTexture *roughness, PyTexture *opacity, PyTexture *normal_map,
        PyTexture *kd, std::string_view eta
    ) noexcept: PySurface{name, "metal", roughness, opacity, normal_map} {
        add_property_node("Kd", kd);
        _node->add_property("eta", luisa::string(eta));
    }
    ~PyMetalSurface() = default;
};

class PyPlasticSurface final: public PySurface {
public:
    PyPlasticSurface(
        std::string_view name,
        PyTexture *roughness, PyTexture *opacity, PyTexture *normal_map,
        PyTexture *kd, PyTexture *ks, PyTexture *eta
    ) noexcept: PySurface{name, "substrate", roughness, opacity, normal_map} {
        add_property_node("Kd", kd);
        add_property_node("Ks", ks);
        add_property_node("eta", eta);
    }
    ~PyPlasticSurface() = default;
};

class PyGlassSurface final: public PySurface {
public:
    PyGlassSurface(
        std::string_view name,
        PyTexture *roughness, PyTexture *opacity, PyTexture *normal_map,
        PyTexture *ks, PyTexture *kt, PyTexture *eta
    ) noexcept: PySurface{name, "glass", roughness, opacity, normal_map} {
        add_property_node("Ks", ks);
        add_property_node("Kt", kt);
        add_property_node("eta", eta);
    }
    ~PyGlassSurface() = default;
};

class PyDisneySurface final: public PySurface {
public:
    PyDisneySurface(
        std::string_view name,
        PyTexture *roughness, PyTexture *opacity, PyTexture *normal_map,
        PyTexture *kd, PyTexture *eta,
        PyTexture *metallic, PyTexture *specular_tint,
        PyTexture *specular_trans, PyTexture *diffuse_trans
    ) noexcept: PySurface{name, "disney", roughness, opacity, normal_map} {
        add_property_node("Kd", kd);
        add_property_node("eta", eta);
        add_property_node("metallic", metallic);
        add_property_node("specular_tint", specular_tint);
        add_property_node("specular_trans", specular_trans);
        add_property_node("diffuse_trans", diffuse_trans);
        if (diffuse_trans) {
            _node->add_property("thin", JustBool(true));
        }
    }
    ~PyDisneySurface() = default;
};

class PyLayeredSurface final: public PySurface {
public:
    PyLayeredSurface(
        std::string_view name,
        PySurface *top, PySurface *bottom, PyTexture *thickness
    ) noexcept: PySurface{name, "layered", nullptr, nullptr, nullptr} {
        add_property_node("top", top);
        add_property_node("bottom", bottom);
        add_property_node("thickness", thickness);
    }
    ~PyLayeredSurface() = default;
};


// Shape
class PyShape: public PyDesc {
public:
    PyShape(
        std::string_view name, std::string_view impl_type,
        PySurface *surface, PyLight *emission, PySubsurface *subsurface,
        float clamp_normal
    ) noexcept: PyDesc{name, SceneNodeTag::SHAPE, impl_type} {
        add_property_node("surface", surface);
        add_property_node("light", emission);
        add_property_node("subsurface", subsurface);
        _node->add_property("clamp_normal", clamp_normal);
    }
    virtual ~PyShape() = default;
};

class PyRigid final: public PyShape {
public:
    PyRigid(
        std::string_view name,
        std::string_view obj_path,
        const PyDoubleArr &vertices, const PyUIntArr &triangles,
        const PyDoubleArr &normals, const PyDoubleArr &uvs,
        PyTransform *transform,
        PySurface *surface, PyLight *emission, PySubsurface *subsurface,
        float clamp_normal
    ) noexcept: PyShape(name, "mesh", surface, emission, subsurface, clamp_normal) {
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
    ~PyRigid() = default;
    void update(PyTransform *transform) noexcept {
        add_property_node("transform", transform);
    }
};

class PyDeformable final: public PyShape {
public:
    PyDeformable(
        std::string_view name,
        const PyDoubleArr &vertices, const PyUIntArr &triangles,
        const PyDoubleArr &normals, const PyDoubleArr &uvs,
        PySurface *surface, PyLight *emission, PySubsurface *subsurface,
        float clamp_normal
    ) noexcept: PyShape(name, "deformablemesh", surface, emission, subsurface, clamp_normal) {
        _node->add_property("positions", pyarray_to_vector<double>(vertices));
        _node->add_property("indices", pyarray_to_vector<uint, double>(triangles));
        _node->add_property("normals", pyarray_to_vector<double>(normals));
        _node->add_property("uvs", pyarray_to_vector<double>(uvs));
    }
    ~PyDeformable() = default;
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

class PyParticles final: public PyShape {
public:
    PyParticles(
        std::string_view name,
        const PyDoubleArr &centers, const PyDoubleArr &radii, uint subdivision,
        PySurface *surface, PyLight *emission, PySubsurface *subsurface,
        float clamp_normal
    ) noexcept: PyShape(name, "spheregroup", surface, emission, subsurface, clamp_normal) {
        _node->add_property("centers", pyarray_to_vector<double>(centers));
        _node->add_property("radii", pyarray_to_vector<double>(radii));
        _node->add_property("subdivision", double(subdivision));
    }
    ~PyParticles() = default;
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
    virtual ~PyFilm() = default;
};


// Filter
class PyFilter: public PyDesc {
public:
    PyFilter(float radius) noexcept:
        PyDesc{"", SceneNodeTag::FILTER, "gaussian"} {
        _node->add_property("radius", radius);
    }
    virtual ~PyFilter() = default;
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
    virtual ~PyCamera() = default;
    void update(PyTransform *pose) noexcept {
        add_property_node("transform", pose);
    }
    void *camera{nullptr};
    bool denoise{false};
    luisa::unique_ptr<Buffer<float4>> color_buffer = nullptr;
    luisa::unique_ptr<Buffer<float4>> denoised_buffer = nullptr;
    luisa::shared_ptr<DenoiserExt::Denoiser> denoiser = nullptr;
};

class PyPinhole final: public PyCamera {
public:
    PyPinhole(    
        std::string_view name,
        PyTransform *pose, PyFilm *film, PyFilter *filter, uint spp,
        float fov
    ) noexcept: PyCamera{name, "pinhole", pose, film, filter, spp} {
        _node->add_property("fov", fov);
    }
    ~PyPinhole() = default;
    void update(PyTransform *pose, float fov) noexcept {
        PyCamera::update(pose);
        _node->add_property("fov", fov);
    }
};

class PyThinLens final: public PyCamera {
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
    ~PyThinLens() = default;
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
    virtual ~PyEnvironment() = default;
};


// Light Sampler
class PyLightSampler: public PyDesc {
public:
    PyLightSampler() noexcept:
        PyDesc{"", SceneNodeTag::LIGHT_SAMPLER, "uniform"} { }
    virtual ~PyLightSampler() = default;
};


// Sampler
class PySampler: public PyDesc {
public:
    PySampler(std::string_view impl_type) noexcept:
        PyDesc{"", SceneNodeTag::SAMPLER, impl_type} { }
    virtual ~PySampler() = default;
};

class PyIndependent final: public PySampler {
public:
    PyIndependent() noexcept: PySampler{"independent"} { }
    ~PyIndependent() = default;
};

class PyPMJ02BN final: public PySampler {
public:
    PyPMJ02BN() noexcept: PySampler{"pmj02bn"} { }
    ~PyPMJ02BN() = default;
};


// Integrator
class PyIntegrator: public PyDesc {
public:
    // rr: Russian Roulette, a technique to control the average depth of ray tracing.
    PyIntegrator(
        std::string_view impl_type, LogLevel log_level, bool enable_cache,
        uint max_depth, uint rr_depth, float rr_threshold
    ) noexcept:
        PyDesc{"", SceneNodeTag::INTEGRATOR, impl_type} {
        _node->add_property("silent", JustBool(log_level == LogLevel::WARNING));
        _node->add_property("enable_cache", JustBool(enable_cache));
        _node->add_property("depth", double(max_depth));
        _node->add_property("rr_depth", double(rr_depth));
        _node->add_property("rr_threshold", rr_threshold);
    }
    virtual ~PyIntegrator() = default;
};

class PyWavePath final: public PyIntegrator {
public:
    PyWavePath(
        LogLevel log_level, bool enable_cache,
        uint max_depth, uint rr_depth, float rr_threshold
    ) noexcept:
        PyIntegrator{"wavepath", log_level, enable_cache, max_depth, rr_depth, rr_threshold} { }
    ~PyWavePath() = default;
};

class PyWavePathV2 final: public PyIntegrator {
public:
    PyWavePathV2(
        LogLevel log_level, bool enable_cache,
        uint max_depth, uint rr_depth, float rr_threshold, uint state_limit
    ) noexcept:
        PyIntegrator{"wavepath_v2", log_level, enable_cache, max_depth, rr_depth, rr_threshold} {
        _node->add_property("state_limit", double(state_limit));
    }
    ~PyWavePathV2() = default;
};


// Spectrum
class PySpectrum: public PyDesc {
public:
    PySpectrum(std::string_view impl_type) noexcept:
        PyDesc{"", SceneNodeTag::SPECTRUM, impl_type} { }
    virtual ~PySpectrum() = default;
};

class PyHero final: public PySpectrum {
public:
    PyHero(uint dimension) noexcept:
        PySpectrum{"hero"} {
        _node->add_property("dimension", double(dimension));
    }
    ~PyHero() = default;
};

class PySRGB final: public PySpectrum {
public:
    PySRGB() noexcept: PySpectrum{"srgb"} { }
    ~PySRGB() = default;
};


// Root
class PyRender final: public PyDesc {
public:
    PyRender(
        std::string_view name,
        PySpectrum *spectrum, PyIntegrator *integrator, float clamp_normal
    ) noexcept: PyDesc{name, SceneNodeTag::ROOT, SceneDesc::root_node_identifier} {
        add_property_node("spectrum", spectrum);
        add_property_node("integrator", integrator);
        _node->add_property("clamp_normal", clamp_normal);
    }
    ~PyRender() = default;
};