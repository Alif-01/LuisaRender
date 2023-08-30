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
        return std::move(transform_info);
    }
    [[nodiscard]] static RawTransformInfo srt(float3 translate, float4 rotate, float3 scale) noexcept {
        RawTransformInfo transform_info;
        transform_info.build_srt(std::move(translate), std::move(rotate), std::move(scale));
        return std::move(transform_info);
    }
    [[nodiscard]] static RawTransformInfo view(float3 position, float3 front, float3 up) noexcept {
        RawTransformInfo transform_info;
        transform_info.build_view(std::move(position), std::move(front), std::move(up));
        return std::move(transform_info);
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
    static RawTextureInfo constant(FloatArr constant) noexcept {
        RawTextureInfo texture_info;
        texture_info.build_constant(std::move(constant));
        return std::move(texture_info);
    }
    
    void build_constant(FloatArr &&constant) noexcept;
    void build_image(StringArr &&image, float3 &&scale) noexcept;
    void build_checker(RawTextureInfo on, RawTextureInfo off, float scale) noexcept;

    [[nodiscard]] StringArr get_type() const noexcept {
        return constant_info != nullptr ? "constant" :
               image_info != nullptr ? "image" :
               checker_info != nullptr ? "checkerboard": "None";
    }

    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format("Texture <{}>", get_type());
    }

    UniquePtr<RawConstantInfo> constant_info{nullptr};
    UniquePtr<RawImageInfo> image_info{nullptr};
    UniquePtr<RawCheckerInfo> checker_info{nullptr};
};

struct RawConstantInfo {
    FloatArr constant;
}; 

struct RawImageInfo {
    StringArr image;
    float3 scale;
};

struct RawCheckerInfo {
    RawTextureInfo on, off;
    float scale;
};

struct RawLightInfo {
    StringArr get_info() const noexcept {
        return luisa::format("Light {} <{}>", name, texture_info.get_info());
    }
    StringArr name;
    RawTextureInfo texture_info;
};

struct RawEnvironmentInfo {
    StringArr get_info() const noexcept {
        return luisa::format("Environment {} <{}, {}>",
            name, texture_info.get_info(), transform_info.get_info());
    }
    StringArr name;
    RawTextureInfo texture_info;
    RawTransformInfo transform_info;
};

struct RawCameraInfo {
    StringArr get_info() const noexcept {
        return luisa::format("Camera {} <{}, fov={}, spp={}, res={}x{}>",
            name, base_pose.get_info(), fov, spp, resolution[0], resolution[1]);
    }

    StringArr name;
    RawTransformInfo base_pose;
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
    RawShapeInfo(StringArr &&name, RawTransformInfo &&transform_info,
                 StringArr &&surface, StringArr &&light, StringArr &&medium) noexcept:
        name{name}, transform_info{std::move(transform_info)}, surface{surface}, light{light}, medium{medium} {}

    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format("Shape {} <{}, {}, surface={}, light={}>",
            name, get_type_info(), transform_info.get_info(), surface, light);
    }
    [[nodiscard]] StringArr get_type_info() const noexcept;

    void build_spheres(FloatArr &&centers, float &&radius, uint &&subdivision) noexcept {
        spheres_info = luisa::make_unique<RawSpheresInfo>(std::move(centers), radius, subdivision);
    }
    void build_mesh(FloatArr &&vertices, UintArr &&triangles, FloatArr &&normals, FloatArr &&uvs) noexcept {
        mesh_info = luisa::make_unique<RawMeshInfo>(
            std::move(vertices), std::move(triangles), std::move(normals), std::move(uvs)
        );
    }
    void build_file(StringArr &&file) noexcept {
        file_info = luisa::make_unique<RawFileInfo>(std::move(file));
    }
    void build_plane() noexcept {
        plane_info = luisa::make_unique<RawPlaneInfo>();
    }

    StringArr name;
    RawTransformInfo transform_info;
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
        return luisa::format("centers={}, subdiv={}", centers.size(), subdivision);
    }

    FloatArr centers;
    float radius;
    uint subdivision;
};

struct RawMeshInfo {
    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format(
            "vertices={}, triangles={}, normals={}, uvs={}",
            vertices.size(), triangles.size(), normals.size(), uvs.size()
        );
    }
    
    FloatArr vertices;
    UintArr triangles;
    FloatArr normals;
    FloatArr uvs;
};

struct RawFileInfo {
    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format("file={}", file);
    }
    
    StringArr file;
};

struct RawPlaneInfo {};

struct RawMetalInfo;
struct RawPlasticInfo;
struct RawGlassInfo;

struct RawSurfaceInfo {

    // static const luisa::string mat_string[5];
    // enum RawMaterial: uint { 
    //     RAW_NULL, RAW_METAL, RAW_SUBSTRATE, RAW_MATTE, RAW_GLASS
    // };
    [[nodiscard]] StringArr get_info() const noexcept {
        return luisa::format("Surface {} <material={}, roughness={}, opacity={}>",
            name, get_type(), roughness, opacity);
    }

    [[nodiscard]] StringArr get_type() const noexcept {
        return metal_info != nullptr ? "metal" :
               plastic_info != nullptr ? "substrate" :
               glass_info != nullptr ? "glass": "None";
    }

    void build_metal(RawTextureInfo &&kd, StringArr &&eta) noexcept {
        metal_info = luisa::make_unique<RawMetalInfo>(std::move(kd), std::move(eta));
    }

    void build_plastic(RawTextureInfo kd, RawTextureInfo ks, float eta) noexcept {
        plastic_info = luisa::make_unique<RawPlasticInfo>(std::move(kd), std::move(ks), eta);
    }

    void build_glass(RawTextureInfo ks, RawTextureInfo kt, float &&eta) noexcept {
        glass_info = luisa::make_unique<RawGlassInfo>(std::move(ks), std::move(kt), eta);
    }

    RawSurfaceInfo(StringArr &&name, float &&roughness, float &&opacity) noexcept:
        name{name}, roughness{roughness}, opacity{opacity} {}

    // void build_file(StringArr &&file) noexcept {
    //     file_info = luisa::make_unique<RawFileInfo>(std::move(file));
    // }
    // void build_plane() noexcept {
    //     plane_info = luisa::make_unique<RawPlaneInfo>();
    // }

    StringArr name;
    // RawMaterial material;
    // RawTextureInfo reflect;

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

} // namespace luisa::render