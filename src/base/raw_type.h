#pragma once

#include <luisa/dsl/syntax.h>
#include <luisa/core/logging.h>

namespace luisa::render {

namespace omit{
    using FloatArr = luisa::vector<float>;
    using IntArr = luisa::vector<int>;
    using UintArr = luisa::vector<uint>;
    using StringArr = luisa::string;
    template<typename T>
    using UniquePtr = luisa::unique_ptr<T>;

    template <typename T, uint N>
    luisa::string format_pack(const Vector<T, N> &v) noexcept {
        luisa::string a(std::to_string(v[0]));
        for (int i = 1; i < N; ++i) a += ", " + std::to_string(v[i]);
        return "(" + a + ")";
    }
}
using namespace omit;

struct RawSRTInfo;
struct RawMatrixInfo;
struct RawViewInfo;

/* Keep the constructing methods (SRT or matrix) on same semantics */
struct RawTransformInfo {
    RawTransformInfo() noexcept = default;
    [[nodiscard]] static RawTransformInfo matrix(float4x4 matrix) noexcept {
        RawTransformInfo transform_info;
        transform_info.build_matrix(std::move(matrix));
        return transform_info;
    }
    [[nodiscard]] static RawTransformInfo srt(float3 translate, float4 rotate, float3 scale) noexcept {
        RawTransformInfo transform_info;
        transform_info.build_srt(std::move(translate), std::move(rotate), std::move(scale));
        return transform_info;
    }
    [[nodiscard]] static RawTransformInfo view(float3 position, float3 front, float3 up) noexcept {
        RawTransformInfo transform_info;
        transform_info.build_view(std::move(position), std::move(front), std::move(up));
        return transform_info;
    }

    void build_matrix(float4x4 matrix) noexcept {
        matrix_info = luisa::make_unique<RawMatrixInfo>(std::move(matrix));
    }
    void build_srt(float3 translate, float4 rotate, float3 scale) noexcept {
        srt_info = luisa::make_unique<RawSRTInfo>(std::move(translate), std::move(rotate), std::move(scale));
    }
    void build_view(float3 position, float3 front, float3 up) noexcept {
        view_info = luisa::make_unique<RawViewInfo>(std::move(position), std::move(front), std::move(up));
    }
    
    [[nodiscard]] StringArr get_type() const noexcept {
        return matrix_info != nullptr ? "matrix":
               srt_info != nullptr ? "srt" :
               view_info != nullptr ? "view" : "None";
    }

    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format("Tranform <{}>", get_type());
    }
    
    UniquePtr<RawMatrixInfo> matrix_info;
    UniquePtr<RawSRTInfo> srt_info;
    UniquePtr<RawViewInfo> view_info;
};

struct RawSRTInfo {
    float3 translate;
    float4 rotate;
    float3 scale;
};

struct RawMatrixInfo {
    float4x4 matrix;
};

struct RawViewInfo {
    float3 position;
    float3 front;
    float3 up;
};

struct RawConstantInfo; 
struct RawImageInfo;
struct RawCheckerInfo;

struct RawTextureInfo {
    RawTextureInfo() noexcept = default;
    [[nodiscard]] static RawTextureInfo constant(FloatArr constant) noexcept {
        RawTextureInfo texture_info;
        texture_info.build_constant(std::move(constant));
        return texture_info;
    }
    
    void build_constant(FloatArr constant) noexcept;
    void build_image(
        StringArr image, FloatArr scale,
        FloatArr image_data = FloatArr(), uint2 resolution = make_uint2(0u), uint channel = 0u
    ) noexcept;
    void build_checker(RawTextureInfo on, RawTextureInfo off, float scale) noexcept;

    [[nodiscard]] StringArr get_type() const noexcept {
        return constant_info != nullptr ? "constant" :
               image_info != nullptr ? "image" :
               checker_info != nullptr ? "checkerboard": "None";
    }

    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format("Texture <{}>", get_type());
    }

    UniquePtr<RawConstantInfo> constant_info;
    UniquePtr<RawImageInfo> image_info;
    UniquePtr<RawCheckerInfo> checker_info;
};

struct RawConstantInfo {
    FloatArr constant;
}; 

struct RawImageInfo {
    StringArr image;
    FloatArr scale;
    FloatArr image_data;
    uint2 resolution;
    uint channel;
};

struct RawCheckerInfo {
    RawTextureInfo on, off;
    float scale;
};

struct RawLightInfo {
    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format("Light {} <{}>", name, texture_info.get_info());
    }
    StringArr name;
    RawTextureInfo texture_info;
};

struct RawEnvironmentInfo {
    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format("Environment {} <{}, {}>",
            name, texture_info.get_info(), transform_info.get_info());
    }
    StringArr name;
    RawTextureInfo texture_info;
    RawTransformInfo transform_info;
};

struct RawCameraInfo {
    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format("Camera {} <pos={}, fov={}, spp={}, res={}x{}>",
            name, pose.get_info(), fov, spp, resolution[0], resolution[1]
        );
        // append_pose.get_info(),
    }

    StringArr name;
    RawTransformInfo pose;
    float fov;
    uint spp;
    uint2 resolution;
    float radius;
};

struct RawSpheresInfo;
struct RawMeshInfo;
struct RawFileInfo;
struct RawPlaneInfo;

struct RawShapeInfo {
    RawShapeInfo(StringArr name, RawTransformInfo transform_info, float clamp_normal,
                 StringArr surface, StringArr light, StringArr medium) noexcept:
        name{std::move(name)}, transform_info{std::move(transform_info)}, clamp_normal{clamp_normal},
        surface{std::move(surface)}, light{std::move(light)}, medium{std::move(medium)} {}

    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format("Shape {} <type <{}>, transform <{}>, clamp_normal={}, surface={}, light={}>",
            name, get_type_info(), transform_info.get_info(), clamp_normal, surface, light);
    }
    [[nodiscard]] StringArr get_type_info() const noexcept;
    [[nodiscard]] StringArr get_type() const noexcept;

    void build_spheres(
        FloatArr centers, float radius, uint subdivision,
        bool reconstruction, float voxel_scale, float smoothing_scale
    ) noexcept {
        spheres_info = luisa::make_unique<RawSpheresInfo>(
            std::move(centers), radius, subdivision,
            reconstruction, voxel_scale, smoothing_scale
        );
    }
    void build_mesh(
        FloatArr vertices, UintArr triangles,
        FloatArr normals, FloatArr uvs, bool is_deformable
    ) noexcept {
        mesh_info = luisa::make_unique<RawMeshInfo>(
            std::move(vertices), std::move(triangles),
            std::move(normals), std::move(uvs), is_deformable
        );
    }
    void build_file(StringArr file) noexcept {
        file_info = luisa::make_unique<RawFileInfo>(std::move(file));
    }
    void build_plane(uint subdivision) noexcept {
        plane_info = luisa::make_unique<RawPlaneInfo>(subdivision);
    }

    StringArr name;
    RawTransformInfo transform_info;
    float clamp_normal;
    StringArr surface;
    StringArr light;
    StringArr medium;
    UniquePtr<RawSpheresInfo> spheres_info;
    UniquePtr<RawMeshInfo> mesh_info;
    UniquePtr<RawFileInfo> file_info;
    UniquePtr<RawPlaneInfo> plane_info;
};

struct RawSpheresInfo {
    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format(
            "centers={}, subdiv={}, reconstruction={}, voxel_scale={}, smoothing_scale={}",
            centers.size(), subdivision, reconstruction, voxel_scale, smoothing_scale
        );
    }

    FloatArr centers;
    float radius;
    uint subdivision;
    bool reconstruction;
    float voxel_scale;
    float smoothing_scale;
};

struct RawMeshInfo {
    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format(
            "vertices={}, triangles={}, normals={}, uvs={}, is_deformable={}",
            vertices.size(), triangles.size(), normals.size(), uvs.size(), is_deformable
        );
    }
    
    [[nodiscard]] StringArr get_type() const noexcept {
        return is_deformable ? "deformablemesh" : "mesh";
    }

    FloatArr vertices;
    UintArr triangles;
    FloatArr normals;
    FloatArr uvs;
    bool is_deformable;
};

struct RawFileInfo {
    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format("file={}", file);
    }
    
    StringArr file;
};

struct RawPlaneInfo {
    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format("subdiv={}", subdivision);
    }

    uint subdivision;
};

struct RawMetalInfo;
struct RawPlasticInfo;
struct RawGlassInfo;

struct RawSurfaceInfo {
    RawSurfaceInfo(StringArr name, float roughness, float opacity) noexcept:
        name{name}, roughness{roughness}, opacity{opacity} {}

    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format("Surface {} <material={}, roughness={}, opacity={}>",
            name, get_type(), roughness, opacity);
    }

    [[nodiscard]] StringArr get_type() const noexcept {
        return metal_info != nullptr ? "metal" :
               plastic_info != nullptr ? "substrate" :
               glass_info != nullptr ? "glass": "None";
    }

    void build_metal(RawTextureInfo kd, StringArr eta) noexcept {
        metal_info = luisa::make_unique<RawMetalInfo>(std::move(kd), std::move(eta));
    }

    void build_plastic(RawTextureInfo kd, RawTextureInfo ks, float eta) noexcept {
        plastic_info = luisa::make_unique<RawPlasticInfo>(std::move(kd), std::move(ks), eta);
    }

    void build_glass(RawTextureInfo ks, RawTextureInfo kt, float eta) noexcept {
        glass_info = luisa::make_unique<RawGlassInfo>(std::move(ks), std::move(kt), eta);
    }

    StringArr name;
    float roughness;
    float opacity;
    UniquePtr<RawMetalInfo> metal_info;
    UniquePtr<RawPlasticInfo> plastic_info;
    UniquePtr<RawGlassInfo> glass_info;
};

struct RawMetalInfo {
    RawTextureInfo kd;
    StringArr eta;
};

struct RawPlasticInfo {
    RawTextureInfo kd, ks;
    float eta;
};

struct RawGlassInfo {
    RawTextureInfo ks, kt;
    float eta;
};

struct RawSamplerInfo {
    RawSamplerInfo() noexcept = default;

    [[nodiscard]] static RawSamplerInfo independent() noexcept {
        RawSamplerInfo sampler_info;
        sampler_info.build_independent();
        return sampler_info;
    }
    [[nodiscard]] static RawSamplerInfo pmj02bn() noexcept {
        RawSamplerInfo sampler_info;
        sampler_info.build_pmj02bn();
        return sampler_info;
    }

    void build_independent() noexcept {
        sampler_index = 1;
    }
    void build_pmj02bn() noexcept {
        sampler_index = 2;
    }

    [[nodiscard]] StringArr get_type() const noexcept {
        return sampler_index == 1 ? "independent" :
               sampler_index == 2 ? "pmj02bn" : "None";
    }

    uint sampler_index;
};

struct RawIntegratorInfo {
    [[nodiscard]] StringArr get_type() const noexcept {
        return version == 1 ? "wavepath" :
               version == 2 ? "wavepath_v2" : "None";
    }
    
    uint version;
    RawSamplerInfo sampler_info;
    bool use_progress;

    uint max_depth;
    uint rr_depth;
    float rr_threshold;
    uint state_limit;
};

struct RawSpectrumInfo {
    RawSpectrumInfo() noexcept = default;

    [[nodiscard]] static RawSpectrumInfo hero(uint dimension) noexcept {
        RawSpectrumInfo spectrum_info;
        spectrum_info.build_hero(dimension);
        return spectrum_info;
    }
    [[nodiscard]] static RawSpectrumInfo srgb() noexcept {
        RawSpectrumInfo spectrum_info;
        spectrum_info.build_srgb();
        return spectrum_info;
    }

    void build_hero(uint dim) noexcept {
        spectrum_index = 1u;
        dimension = dim;
    }
    void build_srgb() noexcept {
        spectrum_index = 2u;
    }

    [[nodiscard]] StringArr get_type() const noexcept {
        return spectrum_index == 1u ? "hero" :
               spectrum_index == 2u ? "srgb" : "None";
    }

    uint spectrum_index{0u};
    uint dimension{0u};
};

struct RawSceneInfo {
    RawIntegratorInfo integrator_info;
    RawSpectrumInfo spectrum_info;
    float clamp_normal;
};

} // namespace luisa::render