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

// template <typename T>
// py::array_t<T> get_default_array(const luisa::vector<T> &a) {
//     auto buffer_info = py::buffer_info{
//         (void *)a.data(), sizeof(T), py::format_descriptor<T>::format(),
//         1, {a.size()}, {sizeof(T)}
//     };
//     return py::array_t<T>(buffer_info);
// }

// template <typename T, uint N>
// Vector<T, N> pyarray_to_pack(const py::array_t<T> &array) noexcept {
//     LUISA_ASSERT(array.size() == N, "Array (size = {}) does not match N = {}", array.size(), N);
//     auto pd = (T *)array.data();
//     Vector<T, N> v;
//     for (int i = 0; i < N; ++i) v[i] = pd[i];
//     return v;
// }

enum LogLevel: uint { VERBOSE, INFO, WARNING };

class PyDesc {
public:
    using ref_pair = std::pair<luisa::string, luisa::string>;
    struct cache {
        cache(luisa::string_view name, SceneNodeTag tag, luisa::string_view impl_type) noexcept:
            node{luisa::make_unique<SceneNodeDesc>(luisa::string(name), tag)},
            name{luisa::string(name)},
            impl_type{luisa::string(impl_type)} {
        }
        luisa::unique_ptr<SceneNodeDesc> node;
        luisa::string name, impl_type;
        // luisa::vector<ref_pair> references;
        luisa::unordered_map<luisa::string, const SceneNodeDesc *> references;
    };

    PyDesc(std::string_view name, SceneNodeTag tag, std::string_view impl_type) noexcept {
        _node_cache.emplace_back(name, tag, impl_type);
        _node = _node_cache.back().node.get();
    }
    
    // [[nodiscard]] virtual SceneNodeTag tag() const noexcept = 0;
    // [[nodiscard]] virtual luisa::string_view impl_type() const noexcept = 0;
    [[nodiscard]] auto node() const noexcept { return _node; }
    auto &node_cache() const noexcept { return _node_cache; }
    void clear_cache() noexcept { _node_cache.clear(); }
    void move_property_cache(PyDesc *property, luisa::string_view property_name) noexcept {
        bool has_name = !_node->identifier().empty();
        for (auto &c: property->_node_cache) {
            if (c.node->identifier().empty()) {
                c.name = luisa::format("{}.{}", property_name, c.name);
                if (has_name) {
                    c.name = luisa::format("{}.{}:{}", _node->identifier(), c.name, c.impl_type);
                    c.node->set_identifier(c.name);
                }
            }
            _node_cache.emplace_back(std::move(c));
        }
        property->_node_cache.clear();
    }
    void add_property_node(luisa::string_view name, PyDesc *property) noexcept {
        if (property) {
            // _node->add_property(name, property->_node);
            add_reference(name, property);
            move_property_cache(property, name);
        }
    }
    // void add_reference(luisa::string name, luisa::string_view reference_name) noexcept {
    void add_reference(luisa::string_view name, PyDesc *property) noexcept {
        if (property) {
            _node_cache[0].references[luisa::string(name)] = property->node();
        }   
        // if (!reference_name.empty()) {
        //     _node_cache[0].references.emplace_back(
        //         std::make_pair(name, luisa::string(reference_name))
        //     );
        // }   
    }
    void define_in_scene(SceneDesc *scene_desc) noexcept {
        for (auto i = _node_cache.size(); i-- > 0; ) {
            auto &c = _node_cache[i];
            auto node = scene_desc->define(std::move(c.node), c.impl_type);
            for (auto &[k, v]: c.references) {
                node->add_property(k, scene_desc->reference(v->identifier()));
            }
            // for (auto &p: c.references) {
            //     node->add_property(p.first, scene_desc->reference(p.second));
            // }
        }
        _node_cache.clear();
    }
    
    // _name{name}, _impl_type{impl_type} { }
    // virtual void update_desc_node(SceneNodeDesc *node) const noexcept = 0;
    // [[nodiscard]] luisa::string_view get_name() const noexcept { return _name; }
    // [[nodiscard]] luisa::string_view get_name(luisa::string_view name) const noexcept {
        // if (_name.empty()) return name; else return _name;
    // }
    // [[nodiscard]] luisa::string_view get_impl_type() const noexcept { return _impl_type; }
    // [[nodiscard]] luisa::string_view get_property_name(luisa::string_view property_name) const noexcept {
        // return luisa::format("{}.{}", _name, property_name);
    // }

protected:
    SceneNodeDesc *_node;
    luisa::vector<cache> _node_cache;
};


// Transform
class PyTransform: public PyDesc {
public:
    PyTransform(std::string_view impl_type) noexcept:
        PyDesc{"", SceneNodeTag::TRANSFORM, impl_type} { }
    // [[nodiscard]] SceneNodeTag tag() const noexcept override { return ; }
};

class PyMatrix: public PyTransform {
public:
    PyMatrix(const PyDoubleArr &matrix) noexcept:
        PyTransform{"matrix"} {
        _node->add_property("m", pyarray_to_vector<double>(matrix));
        // for (auto row = 0u; row < 4u; ++row) {
        //     for (auto col = 0u; col < 4u; ++col) {
        //         arr[col][row] = vec[row * 4u + col];
        //     }
        // }
    }
    void update(const PyDoubleArr &matrix) noexcept {
        _node->add_property("m", pyarray_to_vector<double>(matrix));
    }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return ; }
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
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "srt"; }
};

class PyView: public PyTransform {
public:
    // PyView(const PyDoubleArr &position, const PyDoubleArr &look_at, const PyDoubleArr &up) noexcept {
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
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "view"; }
};


// Texture
class PyTexture: public PyDesc {
public:
    PyTexture(std::string_view impl_type) noexcept:
        PyDesc{"", SceneNodeTag::TEXTURE, impl_type} { }
    // [[nodiscard]] SceneNodeTag tag() const noexcept override { return SceneNodeTag::TEXTURE; }
};

class PyColor: public PyTexture {
public:
    PyColor(const PyDoubleArr &color) noexcept:
        PyTexture("constant") {
        _node->add_property("v", pyarray_to_vector<double>(color));
    }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "constant"; }
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
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "image"; }
};

class PyChecker: public PyTexture {
public:
    PyChecker(PyTexture *on, PyTexture *off, float scale) noexcept:
        PyTexture{"checkerboard"} {
        add_property_node("on", on);
        add_property_node("off", off);
        _node->add_property("scale", scale);
    }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "checkerboard"; }
};


// Lights
class PyLight: public PyDesc {
public:
    PyLight(std::string_view name, PyTexture *emission) noexcept:
        PyDesc{name, SceneNodeTag::LIGHT, "diffuse"} {
        add_property_node("emission", emission);
    }
    // [[nodiscard]] SceneNodeTag tag() const noexcept override { return SceneNodeTag::LIGHT; }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "diffuse"; }
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
    // [[nodiscard]] SceneNodeTag tag() const noexcept override { return SceneNodeTag::SURFACE; }
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
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "metal"; }
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
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "substrate"; }
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
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "glass"; }
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
    // [[nodiscard]] SceneNodeTag tag() const noexcept override { return SceneNodeTag::SHAPE; }
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
            // _node->add_property("indices", pyarray_to_vector<uint>(triangles));
            _node->add_property("indices", pyarray_to_vector<uint, double>(triangles));
            _node->add_property("normals", pyarray_to_vector<double>(normals));
            _node->add_property("uvs", pyarray_to_vector<double>(uvs));

            LUISA_INFO("v{}, f{}, n{}, uv{}",
                _node->property_float_list("positions").size(),
                _node->property_float_list("indices").size(),
                _node->property_float_list("normals").size(),
                _node->property_float_list("uvs").size());
                
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
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "mesh"; }
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
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "deformablemesh"; }
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
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "spheregroup"; }
};


// Film
class PyFilm: public PyDesc {
public:
    PyFilm(const PyUIntArr &resolution) noexcept:
        PyDesc{"", SceneNodeTag::FILM, "color"} {
        _node->add_property("resolution", pyarray_to_vector<uint, double>(resolution));
    }
    // [[nodiscard]] SceneNodeTag tag() const noexcept override { return SceneNodeTag::FILM; }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "color"; }
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
    // [[nodiscard]] SceneNodeTag tag() const noexcept override { return SceneNodeTag::FILTER; }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "gaussian"; }
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
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "pinhole"; }
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
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "thinlens"; }
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
    // [[nodiscard]] SceneNodeTag tag() const noexcept override { return SceneNodeTag::ENVIRONMENT; }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "spherical"; }
};


// Light Sampler
class PyLightSampler: public PyDesc {
public:
    PyLightSampler() noexcept:
        PyDesc{"", SceneNodeTag::LIGHT_SAMPLER, "uniform"} { }
    // [[nodiscard]] SceneNodeTag tag() const noexcept override { return SceneNodeTag::LIGHT_SAMPLER; }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "uniform"; }
};


// Sampler
class PySampler: public PyDesc {
public:
    PySampler(std::string_view impl_type) noexcept:
        PyDesc{"", SceneNodeTag::SAMPLER, impl_type} { }
    // [[nodiscard]] SceneNodeTag tag() const noexcept override { return SceneNodeTag::SAMPLER; }
};

class PyIndependent: public PySampler {
public:
    PyIndependent() noexcept: PySampler{"independent"} { }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "independent"; }
};

class PyPMJ02BN: public PySampler {
public:
    PyPMJ02BN() noexcept: PySampler{"pmj02bn"} { }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "pmj02bn"; }
};


// Integrator
class PyIntegrator: public PyDesc {
public:
    // rr: Russian Roulette, a technique to control the average depth of ray tracing.
    PyIntegrator(std::string_view impl_type, LogLevel log_level, uint max_depth) noexcept:
        PyDesc{"", SceneNodeTag::INTEGRATOR, impl_type} {

        std::cout << log_level << ' ' << max_depth << std::endl;
        
        _node->add_property("use_progress", log_level != LogLevel::WARNING);
        _node->add_property("depth", double(max_depth));
    }
    // [[nodiscard]] SceneNodeTag tag() const noexcept override { return SceneNodeTag::INTEGRATOR; }
};

class PyWavePath: public PyIntegrator {
public:
    PyWavePath(LogLevel log_level, uint max_depth) noexcept:
        PyIntegrator{"wavepath", log_level, max_depth} { }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "wavepath"; }
};

class PyWavePathV2: public PyIntegrator {
public:
    PyWavePathV2(LogLevel log_level, uint max_depth, uint state_limit) noexcept:
        PyIntegrator{"wavepath_v2", log_level, max_depth} {
        _node->add_property("state_limit", double(state_limit));
    }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "wavepath_v2"; }
};


// Spectrum
class PySpectrum: public PyDesc {
public:
    PySpectrum(std::string_view impl_type) noexcept:
        PyDesc{"", SceneNodeTag::SPECTRUM, impl_type} { }
    // [[nodiscard]] SceneNodeTag tag() const noexcept override { return SceneNodeTag::SPECTRUM; }
};

class PyHero: public PySpectrum {
public:
    PyHero(uint dimension) noexcept:
        PySpectrum{"hero"} {
        _node->add_property("dimension", double(dimension));
    }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "hero"; }
};

class PySRGB: public PySpectrum {
public:
    PySRGB() noexcept: PySpectrum{"srgb"} { }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "srgb"; }
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
    // [[nodiscard]] SceneNodeTag tag() const noexcept override { return SceneNodeTag::ROOT; }
    // [[nodiscard]] luisa::string_view impl_type() const noexcept override { return SceneDesc::root_node_identifier; }
};