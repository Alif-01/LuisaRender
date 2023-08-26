#pragma once

#include <luisa/dsl/syntax.h>
#include <luisa/core/logging.h>

namespace luisa::render {

/* Keep the constructing methods (SRT or matrix) on same semantics */
struct RawTransformInfo {
    RawTransformInfo() noexcept = default;
    RawTransformInfo(bool empty, float4x4 transform, float3 translate, float4 rotate, float3 scale) noexcept:
        empty{empty}, transform{transform}, translate{translate}, rotate{rotate}, scale{scale} {}
    RawTransformInfo(float4x4 transform) noexcept: transform{transform}, empty{false} {}
    RawTransformInfo(float3 translate, float4 rotate, float3 scale) noexcept:
        translate{translate}, rotate{rotate}, scale{scale}, empty{false} {}
    
    luisa::string get_info() const noexcept {
        return is_srt() ? "tranform=SRT" :
               is_matrix() ? "transform=matrix" : "No tranform";
    }

    [[nodiscard]] bool is_matrix() const noexcept {
        return any(transform[0] != make_float4(1.0f, 0.0f, 0.0f, 0.0f)) ||
               any(transform[1] != make_float4(0.0f, 1.0f, 0.0f, 0.0f)) ||
               any(transform[2] != make_float4(0.0f, 0.0f, 1.0f, 0.0f)) ||
               any(transform[3] != make_float4(0.0f, 0.0f, 0.0f, 1.0f));
    }
    [[nodiscard]] bool is_srt() const noexcept {
        return any(translate != make_float3(0.f)) ||
               any(rotate != make_float4(0.f)) ||
               any(scale != make_float3(1.f));
    }
    
    bool empty{true};
    float4x4 transform{make_float4x4(1.f)};
    float3 translate{make_float3(0.f)};
    float4 rotate{make_float4(0.f)};
    float3 scale{make_float3(1.f)};
};

struct RawTextureInfo {
    RawTextureInfo() noexcept = default;
    RawTextureInfo(bool empty, luisa::string image, float4 color) noexcept:
        empty{empty}, image{image}, color{color} {}
    RawTextureInfo(float4 color) noexcept:
        color{color}, empty{false} {}
    RawTextureInfo(luisa::string image, float4 image_scale) noexcept:
        image{image}, color{image_scale}, empty{false} {}

    luisa::string get_info() const noexcept {
        return empty ? "No Texture" :
               is_image() ? luisa::format("texture: image={}", image) :
               "texture: color";
    }

    [[nodiscard]] bool is_image() const noexcept {
        return !image.empty();
    }

    bool empty{true};
    luisa::string image{};
    float4 color{make_float4(1.f)};
};

struct RawLightInfo {
    luisa::string name;
    RawTextureInfo texture_info;
};

struct RawEnvironmentInfo {
    luisa::string name;
    RawTextureInfo texture_info;
    RawTransformInfo transform_info;
};

struct RawCameraInfo {
    void print_info() {
        LUISA_INFO(
            "Adding camera {}: from: ({}, {}, {}), to: ({}, {}, {}), up: ({}, {}, {}),"
            "fov: {}, spp: {}, res: ({} x {})",
            name, position[0], position[1], position[2],
            look_at[0], look_at[1], look_at[2],
            up[0], up[1], up[2],
            fov, spp, resolution[0], resolution[1]
        );
    }

    luisa::string name;
    float3 position, look_at, up;
    float fov;
    uint spp;
    uint2 resolution;
    float radius;
};

struct RawShapeInfo {
    using FloatArr = luisa::vector<float>;
    using IntArr = luisa::vector<int>;
    using UintArr = luisa::vector<uint>;
    using StringArr = luisa::string;

    struct RawSpheresInfo {
        StringArr get_info() const noexcept {
            return luisa::format("centers={}, subdiv={}", centers.size(), subdivision);
        }

        FloatArr centers;
        float radius;
        uint subdivision;
    };

    struct RawMeshInfo {
        StringArr get_info() const noexcept {
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
        StringArr get_info() const noexcept {
            return luisa::format("file={}", file);
        }
        
        StringArr file;
    };

    RawShapeInfo(StringArr &&name, RawTransformInfo &&transform_info,
                 StringArr &&surface, StringArr &&light, StringArr &&medium) noexcept:
        name{name}, transform_info{std::move(transform_info)}, surface{surface}, light{light}, medium{medium} {}

    void print_info() const noexcept {
        auto info_str = spheres_info != nullptr ? spheres_info->get_info() :
                        mesh_info != nullptr ? mesh_info->get_info() : 
                        file_info != nullptr ? file_info->get_info() : "";
        LUISA_INFO("Updating shape {}: {}, {}, surface={}",
                   name, info_str, transform_info.get_info(), surface);
    }
    void build_spheres_info(FloatArr &&centers, float &&radius, uint &&subdivision) noexcept {
        spheres_info = luisa::make_unique<RawSpheresInfo>(std::move(centers), radius, subdivision);
    }
    void build_mesh_info(FloatArr &&vertices, UintArr &&triangles, FloatArr &&normals, FloatArr &&uvs) noexcept {
        mesh_info = luisa::make_unique<RawMeshInfo>(
            std::move(vertices), std::move(triangles), std::move(normals), std::move(uvs)
        );
    }
    void build_file_info(StringArr &&file) noexcept {
        file_info = luisa::make_unique<RawFileInfo>(std::move(file));
    }

    StringArr name;
    RawTransformInfo transform_info;
    luisa::unique_ptr<RawSpheresInfo> spheres_info;
    luisa::unique_ptr<RawMeshInfo> mesh_info;
    luisa::unique_ptr<RawFileInfo> file_info;
    StringArr surface;
    StringArr light;
    StringArr medium;
};

struct RawSurfaceInfo {
    static const luisa::string mat_string[5];
    enum RawMaterial: uint { 
        RAW_NULL, RAW_METAL, RAW_SUBSTRATE, RAW_MATTE, RAW_GLASS
    };
    
    void print_info() const noexcept {
        LUISA_INFO("Adding surface {}: {}, {}",
                   name, mat_string[material], texture_info.get_info());
    }

    luisa::string name;
    RawMaterial material;
    RawTextureInfo texture_info;
    float roughness;
};

} // namespace luisa::render