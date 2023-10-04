// Modified from cli.cpp

#include <span>
#include <iostream>
#include <vector>
#include <string>

#include <luisa/core/stl/format.h>
#include <luisa/core/basic_types.h>

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

enum LogLevel: uint { VERBOSE, INFO, WARNING };

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

    static PyTexture image(std::string_view image, const PyFloatArr &scale, const PyFloatArr &image_data) noexcept {
        PyTexture texture;

        bool has_data = image_data.size() > 0;
        if (has_data) {
            uint channel;
            if (image_data.ndim() == 2) channel = 1;
            else if (image_data.ndim() == 3) channel = image_data.shape(2);
            else LUISA_ERROR_WITH_LOCATION("Invalid image dim!");
            // if (scale.shape(0) != channel)
            //     LUISA_ERROR_WITH_LOCATION("Channels of scale and image do not match!");
            texture.texture_info.build_image(
                luisa::string(image), 
                pyarray_to_vector<float>(scale),
                pyarray_to_vector<float>(image_data),
                make_uint2(image_data.shape(1), image_data.shape(0)), channel
            );
        } else {
            texture.texture_info.build_image(
                luisa::string(image), pyarray_to_vector<float>(scale)
            );
        }
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
    // static PySurface empty() noexcept { return PySurface("Null_Surface", 1.0, 1.0); }

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

struct PyIntegrator {
    static PyIntegrator wave_path(
        LogLevel log_level, uint wave_path_version,
        uint max_depth, uint state_limit
    ) noexcept {
        return PyIntegrator {
            RawIntegratorInfo {
                wave_path_version,
                RawSamplerInfo::independent(),
                log_level != LogLevel::WARNING,
                max_depth,
                0, 0.95, state_limit
            }
        };
    }

    RawIntegratorInfo integrator_info;
};

struct PySpectrum {
    PySpectrum() noexcept {}
    
    static PySpectrum hero(uint dimension) noexcept {
        PySpectrum spectrum;
        spectrum.spectrum_info.build_hero(dimension);
        return spectrum;
    }

    static PySpectrum srgb() noexcept {
        PySpectrum spectrum;
        spectrum.spectrum_info.build_srgb();
        return spectrum;
    }

    RawSpectrumInfo spectrum_info;
};

struct PyShape {
    PyShape(
        luisa::string name, PyTransform transform,
        luisa::string surface, luisa::string emission, luisa::string medium,
        float clamp_normal
    ) noexcept : shape_info{
        name, std::move(transform.transform_info), clamp_normal,
        surface, emission, medium
    } {}

    static PyShape rigid_from_file(
        std::string_view name, std::string_view obj_path,
        std::string_view surface, std::string_view emission, float clamp_normal
    ) noexcept {
        PyShape shape(
            luisa::string(name), PyTransform::empty(),
            luisa::string(surface), luisa::string(emission), "", clamp_normal
        );
        shape.shape_info.build_file(luisa::string(obj_path));
        return shape;
    }

    static PyShape rigid_from_mesh(
        std::string_view name,
        const PyFloatArr &vertices, const PyIntArr &triangles,
        const PyFloatArr &normals, const PyFloatArr &uvs,
        std::string_view surface, std::string_view emission, float clamp_normal
    ) noexcept {
        PyShape shape(
            luisa::string(name), PyTransform::empty(),
            luisa::string(surface), luisa::string(emission), "", clamp_normal
        );
        shape.shape_info.build_mesh(
            pyarray_to_vector<float>(vertices),
            pyarray_to_vector<uint>(triangles),
            pyarray_to_vector<float>(normals),
            pyarray_to_vector<float>(uvs),
            false
        );
        return shape;
    }

    static PyShape deformable(
        std::string_view name,
        std::string_view surface, std::string_view emission, float clamp_normal
    ) noexcept {
        PyShape shape(
            luisa::string(name), PyTransform::empty(),
            luisa::string(surface), luisa::string(emission), "", clamp_normal
        );
        shape.shape_info.build_mesh(
            luisa::vector<float>(),
            luisa::vector<uint>(),
            luisa::vector<float>(),
            luisa::vector<float>(),
            true
        );
        return shape;
    }

    static PyShape particles(
        std::string_view name,
        float radius, uint subdivision,
        std::string_view surface, std::string_view emission
    ) noexcept {
        PyShape shape(
            luisa::string(name), PyTransform::empty(),
            luisa::string(surface), luisa::string(emission), "", -1.f
        );
        shape.shape_info.build_spheres(
            luisa::vector<float>(),
            std::move(radius), subdivision
        );
        return shape;
    }

    static PyShape plane(
        std::string_view name,
        float height, float range, const PyFloatArr &up_direction, uint subdivision,
        std::string_view surface, std::string_view emission
    ) noexcept {
        static auto z = make_float3(0.f, 0.f, 1.f);
        auto up = normalize(pyarray_to_pack<float, 3>(up_direction));
        PyShape shape(
            luisa::string(name), PyTransform(RawTransformInfo::srt(
                height * up,
                make_float4(cross(z, up), degrees(acos(dot(z, up)))),
                make_float3(range)
            )),
            luisa::string(surface), luisa::string(emission), "", -1.f
        );
        shape.shape_info.build_plane(subdivision);
        return shape;
    }

    void update_rigid(PyTransform &transform) noexcept {
        if (shape_info.get_type() != "mesh") 
            LUISA_ERROR_WITH_LOCATION("This object is not a rigid mesh!");
        shape_info.transform_info = std::move(transform.transform_info);
    }

    void update_deformable(
        const PyFloatArr &vertices, const PyIntArr &triangles,
        const PyFloatArr &normals, const PyFloatArr &uvs
    ) noexcept {
        if (shape_info.get_type() != "deformablemesh") 
            LUISA_ERROR_WITH_LOCATION("This object is not a deformable mesh!");
        auto mesh_info = shape_info.mesh_info.get();
        mesh_info->vertices = pyarray_to_vector<float>(vertices);
        mesh_info->triangles = pyarray_to_vector<uint>(triangles);
        mesh_info->normals = pyarray_to_vector<float>(normals);
        mesh_info->uvs = pyarray_to_vector<float>(uvs);
    }

    void update_particles(
        const PyFloatArr &vertices
    ) noexcept {
        if (shape_info.get_type() != "spheregroup") 
            LUISA_ERROR_WITH_LOCATION("This object is not particles!");
        auto spheres_info = shape_info.spheres_info.get();
        spheres_info->centers = pyarray_to_vector<float>(vertices);
    }

    RawShapeInfo shape_info;
};