#pragma once

#include <luisa/dsl/syntax.h>
#include <luisa/core/logging.h>

namespace luisa::render {

namespace {
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

/* Keep the constructing methods (SRT or matrix) on same semantics */
struct RawTransformInfo {
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

    RawTransformInfo() noexcept = default;
    void build_matrix(float4x4 &&matrix) noexcept {
        matrix_info = luisa::make_unique<RawMatrixInfo>(std::move(matrix));
    }
    void build_srt(float3 &&translate, float4 &&rotate, float3 &&scale) noexcept {
        srt_info = luisa::make_unique<RawSRTInfo>(std::move(translate), std::move(rotate), std::move(scale));
    }
    void build_view(float3 &&position, float3 &&front, float3 &&up) noexcept {
        view_info = luisa::make_unique<RawViewInfo>(std::move(position), std::move(front), std::move(up));
    }
    
    luisa::string get_type() const noexcept {
        return matrix_info != nullptr ? "matrix":
               srt_info != nullptr ? "srt" :
               view_info != nullptr ? "view" : "None";
    }

    luisa::string get_info() const noexcept {
        return luisa::format("Tranform <{}>", get_type());
    }
    
    UniquePtr<RawMatrixInfo> matrix_info;
    UniquePtr<RawSRTInfo> srt_info;
    UniquePtr<RawViewInfo> view_info;
};

struct RawTextureInfo {
    struct RawConstantInfo {
        // float3 color;
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

    RawTextureInfo() noexcept = default;
    static RawTextureInfo constant(FloatArr &&constant) noexcept {
        RawTextureInfo texture_info;
        texture_info.build_constant(std::move(constant));
        return std::move(texture_info);
    }
    
    void build_constant(FloatArr &&constant) noexcept {
        constant_info = luisa::make_unique<RawConstantInfo>(std::move(constant));
    }
    void build_image(StringArr &&image, float3 &&scale) noexcept {
        image_info = luisa::make_unique<RawImageInfo>(std::move(image), std::move(scale));
    }
    void build_checker(RawTextureInfo &&on, RawTextureInfo &&off, float &&scale) noexcept {
        checker_info = luisa::make_unique<RawCheckerInfo>(std::move(on), std::move(off), scale);
    }

    StringArr get_type() const noexcept {
        return constant_info != nullptr ? "constant" :
               image_info != nullptr ? "image" :
               checker_info != nullptr ? "checkerboard": "None";
    }

    StringArr get_info() const noexcept {
        return luisa::format("Texture <{}>", get_type());
    }

    UniquePtr<RawConstantInfo> constant_info;
    UniquePtr<RawImageInfo> image_info;
    UniquePtr<RawCheckerInfo> checker_info;
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

struct RawShapeInfo {
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

    struct RawPlaneInfo {};

    RawShapeInfo(StringArr &&name, RawTransformInfo &&transform_info,
                 StringArr &&surface, StringArr &&light, StringArr &&medium) noexcept:
        name{name}, transform_info{std::move(transform_info)}, surface{surface}, light{light}, medium{medium} {}

    StringArr get_info() const noexcept {
        return luisa::format("Shape {} <{}, {}, surface={}, light={}>",
            name, get_type_info(), transform_info.get_info(), surface, light);
    }

    StringArr get_type_info() const noexcept {
        return  spheres_info != nullptr ? spheres_info->get_info() :
                mesh_info != nullptr ? mesh_info->get_info() : 
                file_info != nullptr ? file_info->get_info() : "";
    }

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

struct RawSurfaceInfo {
    struct RawMatelInfo {
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

    // static const luisa::string mat_string[5];
    // enum RawMaterial: uint { 
    //     RAW_NULL, RAW_METAL, RAW_SUBSTRATE, RAW_MATTE, RAW_GLASS
    // };
    
    // StringArr get_info() const noexcept {
    //     return luisa::format("Surface {} <material={}, {}, roughness={}>",
    //         name, mat_string[material], texture_info.get_info(), roughness);
    // }
    StringArr get_info() const noexcept {
        return luisa::format("Surface {} <material={}, roughness={}, opacity={}>",
            name, get_type(), roughness, opacity);
    }

    StringArr get_type() const noexcept {
        return metal_info != nullptr ? "metal" :
               plastic_info != nullptr ? "substrate" :
               glass_info != nullptr ? "glass": "None";
    }

    void build_metal(RawTextureInfo &&kd, StringArr &&eta) noexcept {
        metal_info = luisa::make_unique<RawMetalInfo>(std::move(kd), std::move(eta));
    }

    void build_plastic(RawTextureInfo &&kd, RawTextureInfo &&ks, float &&eta) noexcept {
        plastic_info = luisa::make_unique<RawPlasticInfo>(std::move(kd), std::move(ks), eta);
    }

    void build_glass(RawTextureInfo &&ks, RawTextureInfo &&kt, float &&eta) noexcept {
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
    UniquePtr<RawMatelInfo> metal_info;
    UniquePtr<RawPlasticInfo> plastic_info;
    UniquePtr<RawGlassInfo> glass_info;
};

} // namespace luisa::render