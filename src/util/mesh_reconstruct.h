#pragma once

#include <luisa/core/stl.h>
#include <luisa/runtime/rtx/triangle.h>
#include <util/vertex.h>

#ifdef USE_OPENVDB
#include <openvdb/openvdb.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>
#endif

namespace luisa::render {

using compute::Triangle;
struct ReconstructMesh {
    luisa::vector<Vertex> vertices;
    luisa::vector<Triangle> triangles;
};

class MeshReconstructor {
public:
    MeshReconstructor() noexcept = default;
    [[nodiscard]] virtual ReconstructMesh reconstruct(
        luisa::span<float> positions, float particle_radius,
        float voxel_scale, float smoothing_scale
    ) noexcept;
};


#ifdef USE_OPENVDB
using openvdb::Real;
using openvdb::Vec3R;

class OpenVDBMeshReconstructor : public MeshReconstructor {

    class OpenVDBParticleList {

    private:
        Real _radius;
        luisa::vector<Vec3R> _particle_list;

    public:
        OpenVDBParticleList(Real r) noexcept: _radius{r} {}
        [[nodiscard]] size_t size() const noexcept {
            return _particle_list.size();
        }
        [[nodiscard]] Vec3R getPos(int n) const noexcept {
            return _particle_list[n];
        }
        [[nodiscard]] Real getRadius(int n) const noexcept {
            return _radius;
        }
        void clear() noexcept {
            _particle_list.clear();
        }
        void addPos(const Vec3R &p) noexcept {
            _particle_list.push_back(p);
        }
        void getPos(size_t n, Vec3R &pos) const noexcept {
            pos = _particle_list[n];
        }
        void getPosRad(size_t n, Vec3R &pos, Real &rad) const noexcept {
            pos = _particle_list[n];
            rad = _radius;
        }
    };

public:
    OpenVDBMeshReconstructor() noexcept;
    [[nodiscard]] ReconstructMesh reconstruct(
        luisa::span<float> positions, float particle_radius,
        float voxel_scale, float smoothing_scale
    ) override noexcept;
}
#endif

MeshReconstructor getConstructor() noexcept;

}  // namespace luisa::render