#include <luisa/core/logging.h>
#include <luisa/core/clock.h>
#include <util/mesh_construct.h>

namespace luisa::render {

#ifdef USE_OPENVDB
using namespace openvdb::tools;
using openvdb::FloatGrid;
using openvdb::Vec3I;
using openvdb::Vec3s;
using openvdb::Vec4I;
using openvdb::createLevelSet;
using openvdb::initialize;

OpenVDBMeshConstructor::OpenVDBMeshConstructor(
    float particle_radius, float voxel_scale, float isovalue, float adaptivity
) noexcept : 
    MeshConstructor{particle_radius, voxel_scale, isovalue},
    _adaptivity{adaptivity} {
    initialize();
}

ConstructMesh OpenVDBMeshConstructor::construct(
    const luisa::vector<float> &positions
) noexcept {
    if (positions.size() % 3u != 0u)
        LUISA_ERROR_WITH_LOCATION("Invalid particle count.");

    Clock clock;

    OpenVDBParticleList pa(_particle_radius);
    auto particle_count = positions.size() / 3u;
    for (auto i = 0; i < particle_count; i++) {
        pa.addPos(Vec3R(positions[i * 3u + 0u], positions[i * 3u + 1u], positions[i * 3u + 2u]));
    }
    LUISA_INFO(
        "Particles count = {}, radius = {}, voxel_scale = {}",
        particle_count, _particle_radius, _voxel_scale
    );
    LUISA_INFO("Add particles in {} ms: ", clock.toc());

    float voxel_size = _particle_radius * _voxel_scale;
    float particle_sep = _particle_radius * 2;
    float index_sep = particle_sep / voxel_size;

    auto sdf = createLevelSet<FloatGrid>(voxel_size);
    ParticlesToLevelSet<FloatGrid> p2ls(*sdf);
    p2ls.setRmin(index_sep / 1.1);
    p2ls.setRmax(index_sep * 2);
    p2ls.rasterizeSpheres(pa, particle_sep);
    pruneLevelSet(sdf->tree());
    // LUISA_INFO(
    //     "P2LS: voxel_size = {}, half_width = {}, rmin = {}, rmax = {}, "
    //     "min_count = {}, max_count = {}, grain_size = {}",
    //     p2ls.getVoxelSize(), p2ls.getHalfWidth(), p2ls.getRmin(), p2ls.getRmax(),
    //     p2ls.getMinCount(), p2ls.getMaxCount(), p2ls.getGrainSize()
    // );
    LUISA_INFO("To SDF in {} ms: ", clock.toc());

    std::vector<Vec3s> points;
    std::vector<Vec3I> tris;
    std::vector<Vec4I> quads;
    volumeToMesh(*sdf, points, tris, quads, _isovalue, _adaptivity, true);
    LUISA_INFO("To Mesh in {} ms: ", clock.toc());

    ConstructMesh mesh;
    mesh.vertices.resize(points.size());
    mesh.triangles.resize(tris.size() + quads.size() * 2);

    for (auto i = 0u; i < points.size(); i++) {
        auto point = points[i];
        mesh.vertices[i] = Vertex::encode(
            make_float3(point[0], point[1], point[2]),
            make_float3(0.f, 0.f, 1.f), make_float2(0.f)
        );
    }

    for (auto i = 0u; i < tris.size(); i++) {
        auto tri = tris[i];
        mesh.triangles[i] = {tri[0], tri[1], tri[2]};
    }
    for (auto i = 0u; i < quads.size(); i++) {
        auto quad = quads[i];
        auto base_index = tris.size() + i * 2u;
        mesh.triangles[base_index + 0u] = {quad[0], quad[1], quad[2]};
        mesh.triangles[base_index + 1u] = {quad[0], quad[2], quad[3]};
    }
    
    LUISA_INFO(
        "Reconstruct mesh surface from particles using OpenVDB in {} ms: "
        "vertices_count = {}, triangles_count = {}.",
        clock.toc(), mesh.vertices.size(), mesh.triangles.size()
    );
    return mesh;
}
#endif

luisa::unique_ptr<MeshConstructor> get_mesh_constructor(
    std::string_view type, float particle_radius, float voxel_scale,
    float isovalue, float adaptivity
) noexcept {
    if (type == "OpenVDB") {
#ifdef USE_OPENVDB
        return luisa::make_unique<OpenVDBMeshConstructor>(
            particle_radius, voxel_scale, isovalue, adaptivity
        );
#else
        LUISA_ERROR_WITH_LOCATION("Invalid OpenVDB library.");
#endif
    }
    return nullptr;
}

}  // namespace luisa::render