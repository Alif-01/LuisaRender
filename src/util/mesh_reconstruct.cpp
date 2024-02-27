#include <util/mesh_reconstruct.h>

namespace luisa::render {

#ifdef USE_OPENVDB
using namespace openvdb::tools;
using openvdb::FloatGrid;
using openvdb::Vec3I;
using openvdb::Vec3s;
using openvdb::Vec4I;
using openvdb::createLevelSet;
using openvdb::initialize;

// OpenVDBParticleList::OpenVDBParticleList(Real r) noexcept
//     : _radius{r} {}

// size_t OpenVDBParticleList::size() const noexcept {
//     return _particle_list.size();
// }

// Vec3R OpenVDBParticleList::getPos(int n) const noexcept {
//     return _particle_list[n];
// }

// Real OpenVDBParticleList::getRadius(int n) const noexcept {
//     return _radius;
// }

// void OpenVDBParticleList::clear() noexcept {
//     _particle_list.clear();
// }

// void OpenVDBParticleList::addPos(const Vec3R &p) noexcept {
//     _particle_list.push_back(p);
// }

// void OpenVDBParticleList::getPos(size_t n, Vec3R &pos) const noexcept {
//     pos = _particle_list[n];
// }

// void OpenVDBParticleList::getPosRad(size_t n, Vec3R &pos, Real &rad) const noexcept {
//     pos = _particle_list[n];
//     rad = _radius;
// }

OpenVDBMeshReconstructor::OpenVDBMeshReconstructor() noexcept {
    initialize();
}

ReconstructMesh OpenVDBMeshReconstructor::reconstruct(
    luisa::span<float> positions, float particle_radius,
    float voxel_scale, float smoothing_scale
) noexcept {
    if (positions.size() % 3u != 0u)
        LUISA_ERROR_WITH_LOCATION("Invalid particle count.");

    OpenVDBParticleList pa(particle_radius);
    auto particle_count = positions.size() / 3u;
    for (auto i = 0; i < particle_count; i++) {
        pa.addPos(Vec3R(
            positions[i * 3u + 0u], positions[i * 3u + 1u], positions[i * 3u + 2u]
        ));
    }
    float voxel_size = particle_radius * voxel_scale;
    float smoothing_radius = particle_radius * smoothing_scale;
    auto sdf = createLevelSet<FloatGrid>(voxel_size, voxel_size * 2);
    particlesToSdf(pa, *sdf, smoothing_radius);

    std::vector<Vec3s> points;
    std::vector<Vec3I> tris;
    std::vector<Vec4I> quads;
    volumeToMesh(*sdf, points, tris, quads);

    ReconstructMesh mesh;
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
        auto base_index = quads.size() + i * 2u;
        mesh.triangles[base_index + 0u] = {quad[0], quad[1], quad[2]};
        mesh.triangles[base_index + 1u] = {quad[0], quad[2], quad[3]};
    }
    return mesh;
}
#endif


MeshReconstructor getConstructor() noexcept {
#ifdef USE_OPENVDB
    return OpenVDBMeshReconstructor();
#else
    LUISA_ERROR_WITH_LOCATION("Invalid reconstruction library.");
#endif
}

}  // namespace luisa::render

/// @return coordinate bbox in the space of the specified transfrom
// openvdb::CoordBBox getBBox(const openvdb::GridBase& grid) {
//     openvdb::CoordBBox bbox;
//     openvdb::Coord &min = bbox.min(), &max = bbox.max();
//     openvdb::Vec3R pos;
//     openvdb::Real rad, invDx = 1 / grid.voxelSize()[0];
//     for (size_t n = 0, e = this->size(); n < e; ++n) {
//         this->getPosRad(n, pos, rad);
//         const openvdb::Vec3d xyz = grid.worldToIndex(pos);
//         const openvdb::Real r = rad * invDx;
//         for (int i = 0; i < 3; ++i) {
//             min[i] = openvdb::math::Min(min[i], openvdb::math::Floor(xyz[i] - r));
//             max[i] = openvdb::math::Max(max[i], openvdb::math::Ceil(xyz[i] + r));
//         }
//     }
//     return bbox;
// }

// class SurfaceReconstruction {
// private:
//     MyParticleList particles;
//     float influence_radius;
//     openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid> raster;

//     SurfaceReconstruction(float particle_radius, float voxel_size, float influence_radius) :
//         particles{particle_radius},
//         influence_radius{influence_radius},
//         raster{openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid>{
//             *openvdb::createLevelSet<openvdb::FloatGrid>(voxel_size, voxel_size * 2)
//         }} {
//         raster.setGrainSize(1);
//     }

// template<typename GridT, typename ParticleListT, typename InterrupterT>
// inline void
// particlesToSdf(const ParticleListT& plist, GridT& grid, Real radius, InterrupterT* interrupt)
// {
//     static_assert(std::is_floating_point<typename GridT::ValueType>::value,
//         "particlesToSdf requires an SDF grid with floating-point values");
//     if (grid.getGridClass() != GRID_LEVEL_SET) {
//         OPENVDB_LOG_WARN("particlesToSdf requires a level set grid;"
//             " try Grid::setGridClass(openvdb::GRID_LEVEL_SET)");
//     }
 
//     ParticlesToLevelSet<GridT> p2ls(grid, interrupt);
//     p2ls.rasterizeSpheres(plist, radius);
//     tools::pruneLevelSet(grid.tree());
// }
// }


// int main()
// {
//     // Initialize the OpenVDB library.
//     openvdb::initialize();

//     // Create a vector to hold your particles.
//     std::vector<openvdb::Vec3R> particles;

//     // TODO: Fill the 'particles' vector with your particle data.

//     // Convert the particles to a level set VDB.
//     float voxelSize = 0.1f; // Set this to an appropriate value.
//     float halfWidth = 3.0f; // Set this to an appropriate value.

//     // Create a transform using the voxel size.
//     openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(voxelSize);

//     // Create an instance of the ParticlesToLevelSet class.
//     openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid> p2ls(*transform, halfWidth);

//     // Rasterize the particles into a level set grid.
//     typename openvdb::FloatGrid::Ptr grid = p2ls.rasterize(particles.begin(), particles.end());

//     // Convert the level set VDB to a polygon mesh.
//     std::vector<openvdb::Vec3s> points;
//     std::vector<openvdb::Vec3I> triangles;
//     openvdb::tools::volumeToMesh(*grid, points, triangles);

//     // TODO: Use the 'points' and 'triangles' data.

//     return 0;
// }
