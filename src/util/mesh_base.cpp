//
// Created by Mike Smith on 2022/11/8.
//
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/mesh.h>
#include <assimp/scene.h>
#include <assimp/Subdivision.h>

#include <luisa/core/logging.h>
#include <luisa/core/clock.h>
#include <util/loop_subdiv.h>
#include <util/thread_pool.h>
#include <util/mesh_base.h>

namespace luisa::render {

PlaneGeometry::PlaneGeometry(uint subdiv) noexcept {
    uint num_points = PlaneGeometry::base_points.size();
    luisa::vector<Vertex> base_vertices(num_points);
    for (auto i = 0u; i < num_points; ++i)
        base_vertices[i] = Vertex::encode(
            PlaneGeometry::base_points[i], make_float3(0.f, 0.f, 1.f), make_float2()
        );
    auto result = loop_subdivide(base_vertices, PlaneGeometry::base_triangles, subdiv);
    _vertices = result.vertices;
    _triangles = result.triangles;
    for (auto &v : _vertices) {
        auto p = v.position();
        auto uv = position_to_uv(p);
        v = Vertex::encode(p, make_float3(0.f, 0.f, 1.f), uv);
    }
}

std::shared_future<PlaneGeometry> PlaneGeometry::create(uint subdiv) noexcept {
    if (subdiv > PlaneGeometry::max_subdivision_level) [[unlikely]]
        LUISA_ERROR_WITH_LOCATION("Plane subdivision level {} is too high.", subdiv);

    static std::array<std::shared_future<PlaneGeometry>,
        PlaneGeometry::max_subdivision_level + 1u> cache;
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    if (auto g = cache.at(subdiv); g.valid()) { return g; }

    auto future = global_thread_pool().async(
        [subdiv] { return PlaneGeometry(subdiv); }
    );
    cache[subdiv] = future;
    return future;
}

SphereGeometry::SphereGeometry(uint subdiv) noexcept {
    uint num_points = SphereGeometry::base_points.size();
    luisa::vector<Vertex> base_vertices(num_points);
    
    for (auto i = 0u; i < num_points; ++i) {
        const float3 &p = SphereGeometry::base_points[i];
        base_vertices[i] = Vertex::encode(p, p, make_float2());
    }
    
    auto result = loop_subdivide(base_vertices, SphereGeometry::base_triangles, subdiv);
    _vertices = result.vertices;
    _triangles = result.triangles;
    for (auto &v : _vertices) {
        auto p = normalize(v.position());
        auto uv = direction_to_uv(v.position());
        v = Vertex::encode(p, p, uv);
    }
}

std::shared_future<SphereGeometry> SphereGeometry::create(uint subdiv) noexcept {
    if (subdiv > SphereGeometry::max_subdivision_level) [[unlikely]]
        LUISA_ERROR_WITH_LOCATION("Sphere subdivision level {} is too high.", subdiv);

    static std::array<std::shared_future<SphereGeometry>,
        SphereGeometry::max_subdivision_level + 1u> cache;
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    if (auto g = cache.at(subdiv); g.valid()) { return g; }
    
    auto future = global_thread_pool().async(
        [subdiv] { return SphereGeometry(subdiv); }
    );
    cache[subdiv] = future;
    return future;
}

SpheresMeshGeometry::SpheresMeshGeometry(
    const luisa::vector<float> &centers, const luisa::vector<float> &radii, uint subdiv
) noexcept {
    if (centers.size() % 3u != 0u ||
        radii.size() * 3u != centers.size() && radii.size() != 1u) {
        LUISA_ERROR_WITH_LOCATION("Invalid center or radius count.");
    }

    auto sphere_geom = SphereGeometry::create(subdiv).get();
    auto single_vertices = sphere_geom.vertices();
    auto single_triangles = sphere_geom.triangles();
    uint vertex_count = single_vertices.size();
    uint triangle_count = single_triangles.size();
    uint center_count = centers.size() / 3u;
    bool global_radius = radii.size() == 1u;
    
    _num_spheres = center_count;
    _vertices.reserve(center_count * vertex_count);
    _triangles.reserve(center_count * triangle_count);
    
    for (uint i = 0u; i < center_count; ++i) {
        auto center = make_float3(
            centers[i * 3u + 0u], centers[i * 3u + 1u], centers[i * 3u + 2u]
        );
        auto radius = global_radius ? radii[0] : radii[i];
        for (auto v: single_vertices) {
            _vertices.emplace_back(Vertex::encode(
                v.position() * radius + center, v.normal(), v.uv())
            );
        }
        for (auto t: single_triangles) {
            _triangles.emplace_back(
                t.i0 + vertex_count * i,
                t.i1 + vertex_count * i,
                t.i2 + vertex_count * i
            );
        }
    }
}

std::shared_future<SpheresMeshGeometry> SpheresMeshGeometry::create(
    luisa::vector<float> centers, luisa::vector<float> radii, uint subdiv
) noexcept {
    return global_thread_pool().async(
        [centers = std::move(centers), radii = std::move(radii), subdiv = subdiv] { 
            return SpheresMeshGeometry(centers, radii, subdiv);
        }
    );
}

MeshGeometry::MeshGeometry(
    std::filesystem::path path, uint subdiv,
    bool flip_uv, bool drop_normal, bool drop_uv
) noexcept {
    Clock clock;
    auto path_string = path.string();
    Assimp::Importer importer;
    importer.SetPropertyInteger(
        AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
    importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 45.f);
    auto import_flags = aiProcess_RemoveComponent | aiProcess_SortByPType |
                        aiProcess_ValidateDataStructure | aiProcess_ImproveCacheLocality |
                        aiProcess_PreTransformVertices | aiProcess_FindInvalidData |
                        aiProcess_JoinIdenticalVertices;
    auto remove_flags = aiComponent_ANIMATIONS | aiComponent_BONEWEIGHTS |
                        aiComponent_CAMERAS | aiComponent_LIGHTS |
                        aiComponent_MATERIALS | aiComponent_TEXTURES |
                        aiComponent_COLORS | aiComponent_TANGENTS_AND_BITANGENTS;
    if (drop_uv) {
        remove_flags |= aiComponent_TEXCOORDS;
    } else {
        if (!flip_uv) { import_flags |= aiProcess_FlipUVs; }
        import_flags |= aiProcess_GenUVCoords | aiProcess_TransformUVCoords;
    }
    if (drop_normal) {
        import_flags |= aiProcess_DropNormals;
        remove_flags |= aiComponent_NORMALS;
    } else {
        import_flags |= aiProcess_GenSmoothNormals;
    }
    if (subdiv == 0u) { import_flags |= aiProcess_Triangulate; }
    importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, static_cast<int>(remove_flags));
    auto model = importer.ReadFile(path_string.c_str(), import_flags);
    if (model == nullptr || (model->mFlags & AI_SCENE_FLAGS_INCOMPLETE) ||
        model->mRootNode == nullptr || model->mRootNode->mNumMeshes == 0) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load mesh '{}': {}.",
            path_string, importer.GetErrorString());
    }
    if (auto err = importer.GetErrorString();
        err != nullptr && err[0] != '\0') [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Mesh '{}' has warnings: {}.",
            path_string, err);
    }
    LUISA_ASSERT(model->mNumMeshes == 1u, "Only single mesh is supported.");
    auto mesh = model->mMeshes[0];
    if (subdiv > 0u) {
        auto subdivider = Assimp::Subdivider::Create(Assimp::Subdivider::CATMULL_CLARKE);
        aiMesh *subdiv_mesh = nullptr;
        subdivider->Subdivide(mesh, subdiv_mesh, subdiv, true);
        model->mMeshes[0] = nullptr;
        mesh = subdiv_mesh;
        delete subdivider;
    }

    if (mesh->mTextureCoords[0] == nullptr ||
        mesh->mNumUVComponents[0] != 2) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid texture coordinates in mesh: address = {}, components = {}.",
            static_cast<void *>(mesh->mTextureCoords[0]),
            mesh->mNumUVComponents[0]);
    }

    auto vertex_count = mesh->mNumVertices;
    auto ai_positions = mesh->mVertices;
    auto ai_normals = mesh->mNormals;
    auto ai_uvs = mesh->mTextureCoords[0];

    _vertices.resize(vertex_count);
    _has_normal = ai_normals != nullptr;
    _has_uv = ai_uvs != nullptr;
    for (auto i = 0; i < vertex_count; i++) {
        auto p = make_float3(ai_positions[i].x, ai_positions[i].y, ai_positions[i].z);
        auto n = ai_normals ?
                 normalize(make_float3(ai_normals[i].x, ai_normals[i].y, ai_normals[i].z)) :
                 make_float3(0.f, 0.f, 1.f);
        auto uv = ai_uvs ? make_float2(ai_uvs[i].x, ai_uvs[i].y) : make_float2(0.f, 0.f);
        _vertices[i] = Vertex::encode(p, n, uv);
    }
    if (subdiv == 0u) {
        auto ai_triangles = mesh->mFaces;
        _triangles.resize(mesh->mNumFaces);
        for (auto i = 0; i < mesh->mNumFaces; i++) {
            auto &&face = ai_triangles[i];
            assert(face.mNumIndices == 3u);
            _triangles[i] = {face.mIndices[0], face.mIndices[1], face.mIndices[2]};
        }
    } else {
        auto ai_quads = mesh->mFaces;
        _triangles.resize(mesh->mNumFaces * 2u);
        for (auto i = 0u; i < mesh->mNumFaces; i++) {
            auto &&face = ai_quads[i];
            assert(face.mNumIndices == 4u);
            _triangles[i * 2u + 0u] = {face.mIndices[0], face.mIndices[1], face.mIndices[2]};
            _triangles[i * 2u + 1u] = {face.mIndices[2], face.mIndices[3], face.mIndices[0]};
        }
    }
    LUISA_INFO("Loaded triangle mesh '{}' in {} ms.", path_string, clock.toc());
}

// Load the mesh from a file.
std::shared_future<MeshGeometry> MeshGeometry::create(
    std::filesystem::path path, uint subdiv,
    bool flip_uv, bool drop_normal, bool drop_uv
) noexcept {
    static luisa::lru_cache<uint64_t, std::shared_future<MeshGeometry>> loaded_meshes{256u};
    static std::mutex mutex;

    auto abs_path = std::filesystem::canonical(path).string();
    auto key = luisa::hash_value(abs_path, luisa::hash_value(subdiv));

    std::scoped_lock lock{mutex};
    if (auto m = loaded_meshes.at(key)) { return *m; }

    auto future = global_thread_pool().async(
        [path = std::move(path), subdiv = subdiv, flip_uv = flip_uv, drop_normal = drop_normal, drop_uv = drop_uv] { 
            return MeshGeometry(path, subdiv, flip_uv, drop_normal, drop_uv); 
    });
    loaded_meshes.emplace(key, future);
    return future;
}

MeshGeometry::MeshGeometry(
    const luisa::vector<float> &positions,
    const luisa::vector<uint> &triangles,
    const luisa::vector<float> &normals,
    const luisa::vector<float> &uvs
) noexcept {
    if (triangles.size() % 3u != 0u ||
        positions.size() % 3u != 0u ||
        normals.size() % 3u != 0u ||
        uvs.size() % 2u != 0u ||
        (!normals.empty() && normals.size() != positions.size()) ||
        (!uvs.empty() && uvs.size() / 2u != positions.size() / 3u)) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid vertex or triangle count: "
            "vertices={}, triangles={}, normals={}, uvs={}",
            positions.size(), triangles.size(), normals.size(), uvs.size()
        );
    }
    _has_normal = !normals.empty();
    _has_uv = !uvs.empty();

    auto triangle_count = triangles.size() / 3u;
    auto vertex_count = positions.size() / 3u;
    _triangles.resize(triangle_count);
    for (auto i = 0u; i < triangle_count; ++i) {
        auto t0 = triangles[i * 3u + 0u];
        auto t1 = triangles[i * 3u + 1u];
        auto t2 = triangles[i * 3u + 2u];
        // if (t0 >= vertex_count || t1 >= vertex_count || t2 >= vertex_count) [[unlikely]] {
        //     LUISA_ERROR("Triangle ({}, {}, {}) indices exceed {}", t0, t1, t2, vertex_count);
        // }
        _triangles[i] = Triangle{t0, t1, t2};
    }
    _vertices.resize(vertex_count);
    for (auto i = 0u; i < vertex_count; ++i) {
        auto p0 = positions[i * 3u + 0u];
        auto p1 = positions[i * 3u + 1u];
        auto p2 = positions[i * 3u + 2u];
        auto p = make_float3(p0, p1, p2);
        auto n = normals.empty() ?
                 make_float3(0.f, 0.f, 1.f) :
                 make_float3(normals[i * 3u + 0u], normals[i * 3u + 1u], normals[i * 3u + 2u]);
        auto uv = uvs.empty() ? make_float2(0.f) : make_float2(uvs[i * 2u + 0u], uvs[i * 2u + 1u]);
        _vertices[i] = Vertex::encode(p, n, uv);
    }
}

std::shared_future<MeshGeometry> MeshGeometry::create(
    luisa::vector<float> positions,
    luisa::vector<uint> triangles,
    luisa::vector<float> normals,
    luisa::vector<float> uvs
) noexcept {
    auto future = global_thread_pool().async(
        [positions = std::move(positions), triangles = std::move(triangles),
         normals = std::move(normals), uvs = std::move(uvs)] {
            return MeshGeometry(positions, triangles, normals, uvs);
    });
    return future;
}

}   // namespace luisa::render