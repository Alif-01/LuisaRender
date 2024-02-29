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
struct ConstructMesh {
    luisa::vector<Vertex> vertices;
    luisa::vector<Triangle> triangles;
};

class MeshConstructor {
private:
    float _particle_radius;
    float _voxel_scale;
    float _isovalue;

public:
    MeshConstructor(
        float particle_radius, float voxel_scale, float isovalue
    ) noexcept
        : _particle_radius{particle_radius}, _voxel_scale{voxel_scale}, _isovalue{isovalue} {}
    [[nodiscard]] virtual ConstructMesh construct(
        const luisa::vector<float> &positions
    ) noexcept = 0;
};


#ifdef USE_OPENVDB
using openvdb::Real;
using openvdb::Vec3R;

class OpenVDBMeshConstructor : public MeshConstructor {

    class OpenVDBParticleList {

    private:
        Real _radius;
        luisa::vector<Vec3R> _particle_list;

    public:
        typedef Vec3R PosType;
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

private:
    float _adaptivity;

public:
    OpenVDBMeshConstructor(
        float particle_radius, float voxel_scale = 2.f,
        float isovalue = 0.f, float adaptivity = 0.01f
    ) noexcept;
    [[nodiscard]] ConstructMesh construct(
        const luisa::vector<float> &positions //, float particle_radius,
        // float voxel_scale, float smooth_scale, float isovalue
    ) noexcept override;
};
#endif

luisa::unique_ptr<MeshReconstructor> get_mesh_constructor(
    std::string_view type, float particle_radius, float voxel_scale,
    float isovalue = 0.f, float adaptivity = 0.f
) noexcept;

}  // namespace luisa::render