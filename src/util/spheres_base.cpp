//
// Created by Mike Smith on 2022/11/8.
//
#include <luisa/core/logging.h>
#include <luisa/core/clock.h>
#include <util/thread_pool.h>
#include <util/spheres_base.h>

namespace luisa::render {

SpheresProceduralGeometry::SpheresProceduralGeometry(
    const luisa::vector<float> &centers, const luisa::vector<float> &radii
) noexcept {
    if (centers.size() % 3u != 0u ||
        radii.size() * 3u != centers.size() && radii.size() != 1u) {
        LUISA_ERROR_WITH_LOCATION("Invalid center or radius count.");
    }

    uint center_count = centers.size() / 3u;
    bool global_radius = radii.size() == 1u;
    
    _num_spheres = center_count;
    _aabbs.reserve(center_count);
    
    for (uint i = 0u; i < center_count; ++i) {
        auto center = make_float3(
            centers[i * 3u + 0u], centers[i * 3u + 1u], centers[i * 3u + 2u]
        );
        auto radius = global_radius ? radii[0] : radii[i];
        auto max = center + radius;
        auto min = center - radius;
        _aabbs.emplace_back(AABB{
            .packed_min = {min.x, min.y, min.z},
            .packed_max = {max.x, max.y, max.z}
        });
    }
}

std::shared_future<SpheresProceduralGeometry> SpheresProceduralGeometry::create(
    luisa::vector<float> centers, luisa::vector<float> radii
) noexcept {
    return global_thread_pool().async(
        [centers = std::move(centers), radii = std::move(radii)] { 
            return SpheresProceduralGeometry(centers, radii);
        }
    );
}

} // namespace luisa::render