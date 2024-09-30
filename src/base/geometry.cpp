//
// Created by Mike Smith on 2022/9/14.
//

#include <util/sampling.h>
#include <util/thread_pool.h>
#include <base/geometry.h>
#include <base/pipeline.h>

namespace luisa::render {

Geometry::~Geometry() noexcept {
    for (auto index: _resource_store) {
        _pipeline.remove_resource(index);
    }
}

void Geometry::build(
    CommandBuffer &command_buffer, const luisa::unordered_set<Shape *> &shapes, float time
) noexcept {
    // TODO: AccelOption
    _accel = _pipeline.device().create_accel({});
    // for (auto i = 0u; i < 3u; ++i) {
    //     _world_max[i] = -std::numeric_limits<float>::max();
    //     _world_min[i] = std::numeric_limits<float>::max();
    // }
    for (auto shape : shapes) { _process_shape(command_buffer, time, shape); }
    _instance_buffer = _pipeline.device().create_buffer<uint4>(_instances.size());
    command_buffer << _instance_buffer.copy_from(_instances.data())
                   << _accel.build();
}

// bool Geometry::update(
//     CommandBuffer &command_buffer, float time
// ) noexcept {
//     auto updated = false;
//     if (!_dynamic_transforms.empty()) {
//         updated = true;
//         if (_dynamic_transforms.size() < 128u) {
//             for (auto t : _dynamic_transforms) {
//                 _accel.set_transform_on_update(t.instance_id(), t.matrix(time));
//             }
//         } else {
//             global_thread_pool().parallel(
//                 _dynamic_transforms.size(),
//                 [this, time](auto i) noexcept {
//                     auto t = _dynamic_transforms[i];
//                     _accel.set_transform_on_update(t.instance_id(), t.matrix(time));
//                 });
//             global_thread_pool().synchronize();
//         }
//         command_buffer << _accel.build();
//     }
//     return updated;
// }

void Geometry::_process_shape(
    CommandBuffer &command_buffer, float time,
    const Shape *shape,
    const Surface *overridden_surface,
    const Light *overridden_light,
    const Medium *overridden_medium,
    bool overridden_visible) noexcept {

    auto surface = overridden_surface == nullptr ? shape->surface() : overridden_surface;
    auto light = overridden_light == nullptr ? shape->light() : overridden_light;
    auto medium = overridden_medium == nullptr ? shape->medium() : overridden_medium;
    auto visible = overridden_visible && shape->visible();
    
    if (shape->empty()) { return; }
    if (shape->is_mesh() || shape->is_spheres()) {
        luisa::vector<float> primitive_areas;
        auto shape_geom = [&] {
            auto buffer_id_base = 0u;
            void *geom_resource = nullptr;

            if (shape->is_mesh()) {
                auto [vertices, triangles] = shape->mesh();
                // auto hash = luisa::hash64(
                //     triangles.data(), triangles.size_bytes(), luisa::hash64(
                //     vertices.data(), vertices.size_bytes(), luisa::hash64_default_seed
                // ));
                // if (auto mesh_iter = _mesh_cache.find(hash);
                //     mesh_iter != _mesh_cache.end()) {
                //     return mesh_iter->second;
                // }

                // create mesh
                auto [vertex_buffer, vertex_index] = _pipeline.create_with_index<Buffer<Vertex>>(vertices.size());
                auto [triangle_buffer, triangle_index] = _pipeline.create_with_index<Buffer<Triangle>>(triangles.size());
                auto [mesh, mesh_index] = _pipeline.create_with_index<Mesh>(
                    *vertex_buffer, *triangle_buffer, shape->build_option());
                command_buffer << vertex_buffer->copy_from(vertices.data())
                                << triangle_buffer->copy_from(triangles.data())
                                << mesh->build()
                                << compute::commit();
                auto vertex_buffer_id = _pipeline.register_bindless(vertex_buffer->view());
                auto triangle_buffer_id = _pipeline.register_bindless(triangle_buffer->view());
                _resource_store.insert(_resource_store.end(), {vertex_index, triangle_index, mesh_index});
                LUISA_ASSERT(triangle_buffer_id - vertex_buffer_id == 1u, "Invalid.");

                // compute alias table
                primitive_areas.resize(triangles.size());
                for (auto i = 0u; i < triangles.size(); ++i) {
                    auto t = triangles[i];
                    auto p0 = vertices[t.i0].position();
                    auto p1 = vertices[t.i1].position();
                    auto p2 = vertices[t.i2].position();
                    primitive_areas[i] = std::abs(length(cross(p1 - p0, p2 - p0)));
                }

                buffer_id_base = vertex_buffer_id;
                geom_resource = (void *)mesh;
                // _mesh_cache.emplace(hash, geom);
            } else {
                auto [aabbs] = shape->spheres();
                auto [aabb_buffer, aabb_index] = _pipeline.create_with_index<Buffer<AABB>>(aabbs.size());
                auto [procedural, procedural_index] = _pipeline.create_with_index<ProceduralPrimitive>(aabb_buffer->view(), shape->build_option());
                command_buffer << aabb_buffer->copy_from(aabbs.data())
                                << procedural->build()
                                << compute::commit();
                auto aabbs_buffer_id = _pipeline.register_bindless(aabb_buffer->view());
                _resource_store.insert(_resource_store.end(), {aabb_index, procedural_index});

                // compute alias table
                primitive_areas.resize(aabbs.size());
                for (auto i = 0u; i < aabbs.size(); ++i) {
                    auto diameter = aabbs[i].packed_max[0] - aabbs[i].packed_min[0];
                    primitive_areas[i] = diameter * diameter;
                }
                
                buffer_id_base = aabbs_buffer_id;
                geom_resource = (void *)procedural;
            }

            auto [alias_table, pdf] = create_alias_table(primitive_areas);
            auto [alias_table_buffer_view, alias_table_index, alias_buffer_id] = _pipeline.bindless_buffer<AliasEntry>(alias_table.size());
            auto [pdf_buffer_view, pdf_index, pdf_buffer_id] = _pipeline.bindless_buffer<float>(pdf.size());
            _resource_store.insert(_resource_store.end(), {alias_table_index, pdf_index});
            command_buffer << alias_table_buffer_view.copy_from(alias_table.data())
                           << pdf_buffer_view.copy_from(pdf.data())
                           << compute::commit();
            return ShapeGeometry{geom_resource, buffer_id_base};
        }();
            
        auto instance_id = static_cast<uint>(_accel.size());
        auto properties = shape->vertex_properties();

        if (shape->is_mesh()) {
            properties |= Shape::property_flag_triangle;
        }

        // transform
        auto [t_node, is_static] = _transform_tree.leaf(shape->transform());
        InstancedTransform inst_xform{t_node, instance_id};
        if (!is_static) { _dynamic_transforms.emplace_back(inst_xform); }
        auto object_to_world = inst_xform.matrix(time);

        // TODO: _world_max/min cannot support updating
        // if (shape->is_mesh()) {
        //     auto vertices = shape->mesh().vertices;
        //     for (const auto &v : vertices) {
        //         auto tv = make_float3(object_to_world * make_float4(v.position(), 1.f));
        //         _world_max = max(_world_max, tv);
        //         _world_min = min(_world_min, tv);
        //     }
        // } else {     // just a coarse boundary
        //     auto aabbs = shape->spheres().aabbs;
        //     for (const auto &ab: aabbs) {
        //         auto tmin = make_float3(object_to_world * make_float4(
        //             ab.packed_min[0], ab.packed_min[1], ab.packed_min[2], 1.f));
        //         auto tmax = make_float3(object_to_world * make_float4(
        //             ab.packed_max[0], ab.packed_max[1], ab.packed_max[2], 1.f));
        //         _world_max = max(max(_world_max, tmax), tmin);
        //         _world_min = min(min(_world_min, tmin), tmax);
        //     }
        // }

        // create instance
        auto surface_tag = 0u;
        if (surface != nullptr && !surface->is_null()) {
            surface_tag = _pipeline.register_surface(command_buffer, surface);
            properties |= Shape::property_flag_has_surface;
            if (_pipeline.surfaces().impl(surface_tag)->maybe_non_opaque()) {
                properties |= Shape::property_flag_maybe_non_opaque;
                _any_non_opaque = true;
            }
        }

        auto light_tag = 0u;
        if (light != nullptr && !light->is_null()) {
            light_tag = _pipeline.register_light(command_buffer, light);
            properties |= Shape::property_flag_has_light;
            _instanced_lights.emplace_back(Light::Handle{
                .instance_id = instance_id,
                .light_tag = light_tag
            });
        }

        auto medium_tag = 0u;
        if (medium != nullptr && !medium->is_null()) {
            medium_tag = _pipeline.register_medium(command_buffer, medium);
            properties |= Shape::property_flag_has_medium;
        }

        // emplace instance here since we need to know the opaque property
        if (shape->is_mesh()) {
            _accel.emplace_back(
                *(Mesh *)(shape_geom.resource), object_to_world, visible,
                (properties & Shape::property_flag_maybe_non_opaque) == 0u, instance_id
            );
        } else {
            _accel.emplace_back(
                *(ProceduralPrimitive *)(shape_geom.resource), object_to_world,
                visible, instance_id
            );
        }

        _instances.emplace_back(Shape::Handle::encode(
            shape_geom.buffer_id_base, properties,
            surface_tag, light_tag, medium_tag, primitive_areas.size(),
            shape->has_vertex_normal() ? shape->shadow_terminator_factor() : 0.f,
            shape->intersection_offset_factor(),
            radians(shape->clamp_normal_factor())
        ));

        LUISA_INFO(
            "Instance {}: accel: {}, instance_id: {}, num_dyna: {}, matrix: {}, "
            "surface: {}, light: {}, medium: {}, properties: {}, prim_count: {}",
            shape->impl_type(), _accel.size(), instance_id,
            _dynamic_transforms.size(), object_to_world,
            surface_tag, light_tag, medium_tag, properties, primitive_areas.size()
        );
            
    } else {
        _transform_tree.push(shape->transform());
        for (auto child : shape->children()) {
            _process_shape(command_buffer, time, child, surface, light, medium, visible);
        }
        _transform_tree.pop(shape->transform());
    }
}

Bool Geometry::_alpha_skip(const Interaction &it, Expr<float> u) const noexcept {
    auto skip = def(true);
    $if (it.shape().maybe_non_opaque() & it.shape().has_surface()) {
        $switch (it.shape().surface_tag()) {
            for (auto i = 0u; i < _pipeline.surfaces().size(); i++) {
                if (auto surface = _pipeline.surfaces().impl(i);
                    surface->maybe_non_opaque()) {
                    $case (i) {
                        // TODO: pass the correct swl and time
                        if (auto opacity = surface->evaluate_opacity(it, 0.f)) {
                            skip = u > *opacity;
                        } else {
                            skip = false;
                        }
                    };
                }
            }
            $default { compute::unreachable(); };
        };
    }
    $else {
        skip = false;
    };
    return skip;
}

Bool Geometry::_alpha_skip(const Var<Ray> &ray, const Var<SurfaceHit> &hit) const noexcept {
    auto it = interaction(ray, hit);
    auto u = as<float>(xxhash32(make_uint4(hit.inst, hit.prim, as<uint2>(hit.bary)))) * 0x1p-32f;
    return _alpha_skip(*it, u);
}

Bool Geometry::_alpha_skip(const Var<Ray> &ray, const Var<ProceduralHit> &hit) const noexcept {
    auto it = interaction(ray, hit);
    auto u = as<float>(xxhash32(make_uint2(hit.inst, hit.prim))) * 0x1p-32f;
    return _alpha_skip(*it, u);
}

void Geometry::_procedural_filter(ProceduralCandidate &c) const noexcept {
    Var<ProceduralHit> h = c.hit();
    Var<Ray> ray = c.ray();
    Var<AABB> ab = aabb(instance(h.inst), h.prim);
    Float4x4 shape_to_world = instance_to_world(h.inst);
    Float3x3 m = make_float3x3(shape_to_world);
    Float3 t = make_float3(shape_to_world[3]);
    Float3 aabb_min = m * ab->min() + t;
    Float3 aabb_max = m * ab->max() + t;

    Float3 origin = (aabb_min + aabb_max) * .5f;
    Float radius = length(aabb_max - aabb_min) * .5f * inv_sqrt3;
    Float3 ray_origin = ray->origin();
    Float3 L = origin - ray_origin;
    Float3 dir = ray->direction();
    Float cos_theta = dot(dir, normalize(L));
    $if (cos_theta > 0.f) {
        Float d_oc = length(L);
        Float tc = d_oc * cos_theta;
        Float d = sqrt(d_oc * d_oc - tc * tc);
        $if (d <= radius) {
            Float t1c = sqrt(radius * radius - d * d);
            Float dist = tc - t1c;
            $if (dist < ray->t_max()) {
                c.commit(dist);
            };
        };
    };
}

Var<CommittedHit> Geometry::trace_closest(const Var<Ray> &ray_in) const noexcept {
    if (!_any_non_opaque) {
        // happy path
        return _accel->traverse(ray_in, {})
            .on_procedural_candidate([&](ProceduralCandidate &c) noexcept {
                this->_procedural_filter(c);
            })
            .trace();
    } else {
        return _accel->traverse(ray_in, {})
            .on_surface_candidate([&](SurfaceCandidate &c) noexcept {
                $if (!this->_alpha_skip(c.ray(), c.hit())) {
                    c.commit();
                };
            })
            .on_procedural_candidate([&](ProceduralCandidate &c) noexcept {
                $if (!this->_alpha_skip(c.ray(), c.hit())) {
                    this->_procedural_filter(c);
                };
            })
            .trace();
    }
    // TODO: DirectX has bug with ray query, so we manually march the ray here
//     if (_pipeline.device().backend_name() == "dx") {
//         auto ray = ray_in;
//         auto hit = _accel->intersect(ray, {});
//         constexpr auto max_iterations = 100u;
//         constexpr auto epsilone = 1e-5f;
//         $for (i [[maybe_unused]], max_iterations) {
//             $if (hit->miss()) { $break; };
//             $if (!this->_alpha_skip(ray, hit)) { $break; };
// #ifndef NDEBUG
//             $if (i == max_iterations - 1u) {
//                 compute::device_log(luisa::format(
//                     "ERROR: max iterations ({}) exceeded in trace closest",
//                     max_iterations));
//             };
// #endif
//             ray = compute::make_ray(ray->origin(), ray->direction(),
//                                     hit.committed_ray_t + epsilone,
//                                     ray->t_max());
//             hit = _accel->intersect(ray, {});
//         };
//         return Var<Hit>{hit.inst, hit.prim, hit.bary};
//     }
    // use ray query
    // Callable impl = [this](Var<Ray> ray) noexcept {
    //     auto rq_hit =
    // };
    // return impl(ray_in);
}

Var<bool> Geometry::trace_any(const Var<Ray> &ray) const noexcept {
    if (!_any_non_opaque) {
        // happy path
        return !_accel->traverse_any(ray, {})
            .on_procedural_candidate([&](ProceduralCandidate &c) noexcept {
                this->_procedural_filter(c);
            })
            .trace()->miss();
    } else {
        return !_accel->traverse_any(ray, {})
            .on_surface_candidate([&](SurfaceCandidate &c) noexcept {
                $if (!this->_alpha_skip(c.ray(), c.hit())) {
                    c.commit();
                };
            })
            .on_procedural_candidate([&](ProceduralCandidate &c) noexcept {
                $if (!this->_alpha_skip(c.ray(), c.hit())) {
                    this->_procedural_filter(c);
                };
            })
            .trace()->miss();
    }
}

Interaction Geometry::triangle_interaction(
    const Var<Ray> &ray, Expr<uint> inst_id, Expr<uint> prim_id, Expr<float3> bary
) const noexcept {
    auto shape = instance(inst_id);
    auto m = instance_to_world(inst_id);
    auto tri = triangle(shape, prim_id);
    auto attrib = shading_point(shape, tri, bary, m);
    return Interaction(
        std::move(shape), inst_id, prim_id, attrib,
        dot(ray->direction(), attrib.g.n) > 0.0f);
}

Interaction Geometry::aabb_interaction(
    const Var<Ray> &ray, Expr<uint> inst_id, Expr<uint> prim_id
) const noexcept {
    auto shape = instance(inst_id);
    auto m = instance_to_world(inst_id);
    auto ab = aabb(shape, prim_id);
    auto attrib = shading_point(shape, ab, ray, m);
    return Interaction(
        std::move(shape), inst_id, prim_id, attrib,
        dot(ray->direction(), attrib.g.n) > 0.0f);
}

luisa::shared_ptr<Interaction> Geometry::interaction(
    const Var<Ray> &ray, const Var<SurfaceHit> &hit
) const noexcept {
    Interaction it;
    $if (!hit->miss()) {
        it = triangle_interaction(
            ray, hit.inst, hit.prim,
            make_float3(1.f - hit.bary.x - hit.bary.y, hit.bary)
        );
    };
    return luisa::make_shared<Interaction>(std::move(it));
}

luisa::shared_ptr<Interaction> Geometry::interaction(
    const Var<Ray> &ray, const Var<ProceduralHit> &hit
) const noexcept {
    return luisa::make_shared<Interaction>(
        aabb_interaction(ray, hit.inst, hit.prim));
}

luisa::shared_ptr<Interaction> Geometry::interaction(
    const Var<Ray> &ray, const Var<CommittedHit> &hit
) const noexcept {
    Interaction it;
    $if (hit->is_triangle()) {
        it = triangle_interaction(
            ray, hit.inst, hit.prim,
            make_float3(1.f - hit.bary.x - hit.bary.y, hit.bary)
        );
    }
    $elif (hit->is_procedural()) {
        it = aabb_interaction(ray, hit.inst, hit.prim);
    };
    return luisa::make_shared<Interaction>(std::move(it));
}

Shape::Handle Geometry::instance(Expr<uint> inst_id) const noexcept {
    return Shape::Handle::decode(_instance_buffer->read(inst_id));
}

Float4x4 Geometry::instance_to_world(Expr<uint> inst_id) const noexcept {
    return _accel->instance_transform(inst_id);
}

Var<Triangle> Geometry::triangle(const Shape::Handle &instance, Expr<uint> triangle_id) const noexcept {
    return _pipeline.buffer<Triangle>(instance.triangle_buffer_id()).read(triangle_id);
}

Var<Vertex> Geometry::vertex(const Shape::Handle &instance, Expr<uint> vertex_id) const noexcept {
    return _pipeline.buffer<Vertex>(instance.vertex_buffer_id()).read(vertex_id);
}

Var<AABB> Geometry::aabb(const Shape::Handle &instance, Expr<uint> aabb_id) const noexcept {
    return _pipeline.buffer<AABB>(instance.aabb_buffer_id()).read(aabb_id);
}

template<typename T>
[[nodiscard]] inline auto tri_interpolate(
    Expr<float3> uvw, const T &v0, const T &v1, const T &v2
) noexcept {
    return uvw.x * v0 + uvw.y * v1 + uvw.z * v2;
}

GeometryAttribute Geometry::geometry_point(
    const Shape::Handle &instance, const Var<Triangle> &triangle,
    const Var<float3> &bary, const Var<float4x4> &shape_to_world
) const noexcept {
    auto v0 = vertex(instance, triangle.i0);
    auto v1 = vertex(instance, triangle.i1);
    auto v2 = vertex(instance, triangle.i2);
    // object space
    auto p0 = v0->position();
    auto p1 = v1->position();
    auto p2 = v2->position();
    auto m = make_float3x3(shape_to_world);
    auto t = make_float3(shape_to_world[3]);
    // world space
    auto p = m * tri_interpolate(bary, p0, p1, p2) + t;
    auto dp0 = p1 - p0;
    auto dp1 = p2 - p0;
    auto c = cross(m * dp0, m * dp1);
    auto area = length(c) * .5f;
    auto ng = normalize(c);
    return {.p = p, .n = ng, .area = area};
}

GeometryAttribute Geometry::geometry_point(
    const Shape::Handle &instance, const Var<AABB> &ab,
    const Var<float3> &w, const Var<float4x4> &shape_to_world
) const noexcept {
    auto m = make_float3x3(shape_to_world);
    auto t = make_float3(shape_to_world[3]);
    auto aabb_min = ab->min();
    auto aabb_max = ab->max();
    auto o_local = (aabb_min + aabb_max) * .5f;
    
    auto p = m * (o_local + w) + t;
    auto c = m * w;
    auto radius = length(c);
    auto ng = normalize(c);
    auto area = 4 * pi * radius * radius;
    return {.p = p, .n = ng, .area = area};
}

ShadingAttribute Geometry::shading_point(
    const Shape::Handle &instance, const Var<Triangle> &triangle,
    const Var<float3> &bary, const Var<float4x4> &shape_to_world
) const noexcept {
    auto v0 = vertex(instance, triangle.i0);
    auto v1 = vertex(instance, triangle.i1);
    auto v2 = vertex(instance, triangle.i2);

    // object space
    auto p0_local = v0->position();
    auto p1_local = v1->position();
    auto p2_local = v2->position();

    // compute dpdu and dpdv
    auto uv0 = v0->uv();
    auto uv1 = v1->uv();
    auto uv2 = v2->uv();
    auto duv0 = uv1 - uv0;
    auto duv1 = uv2 - uv0;
    auto det = duv0.x * duv1.y - duv0.y * duv1.x;
    auto inv_det = 1.f / det;
    auto dp0_local = p1_local - p0_local;
    auto dp1_local = p2_local - p0_local;
    auto dpdu_local = (dp0_local * duv1.y - dp1_local * duv0.y) * inv_det;
    auto dpdv_local = (dp1_local * duv0.x - dp0_local * duv1.x) * inv_det;

    // world space
    // clamp normal
    auto clamp_angle = instance.clamp_normal_factor();
    auto m = make_float3x3(shape_to_world);
    auto t = make_float3(shape_to_world[3]);
    auto ng_local = normalize(cross(dp0_local, dp1_local));
    auto n0_local = clamp_normal_angle(v0->normal(), ng_local, clamp_angle);
    auto n1_local = clamp_normal_angle(v1->normal(), ng_local, clamp_angle);
    auto n2_local = clamp_normal_angle(v2->normal(), ng_local, clamp_angle);
    auto ns_local = tri_interpolate(bary, n0_local, n1_local, n2_local);

    auto p = m * tri_interpolate(bary, p0_local, p1_local, p2_local) + t;
    auto c = cross(m * dp0_local, m * dp1_local);
    auto area = length(c) * .5f;
    auto ng = normalize(c);
    auto fallback_frame = Frame::make(ng);
    auto dpdu = ite(det == 0.f, fallback_frame.s(), m * dpdu_local);
    auto dpdv = ite(det == 0.f, fallback_frame.t(), m * dpdv_local);
    auto ns = ite(instance.has_vertex_normal(), normalize(m * ns_local), ng);
    auto uv = ite(instance.has_vertex_uv(), tri_interpolate(bary, uv0, uv1, uv2), bary.yz());
    return {.g = {.p = p,
                  .n = ng,
                  .area = area},
            .ps = p,
            .ns = face_forward(ns, ng),
            .dpdu = dpdu,
            .dpdv = dpdv,
            .uv = uv};
}

ShadingAttribute Geometry::shading_point(
    const Shape::Handle &instance, const Var<AABB> &ab,
    const Var<Ray> &ray, const Var<float4x4> &shape_to_world
) const noexcept {
    auto m = make_float3x3(shape_to_world);
    auto t = make_float3(shape_to_world[3]);
    auto aabb_min = m * ab->min() + t;
    auto aabb_max = m * ab->max() + t;
    auto origin = (aabb_min + aabb_max) * .5f;
    auto radius = length(aabb_max - aabb_min) * .5f * inv_sqrt3;
    
    auto ray_origin = ray->origin();
    auto L = origin - ray_origin;
    auto dir = ray->direction();
    auto cos_theta = dot(dir, normalize(L));
    auto d_oc = length(L);
    auto tc = d_oc * cos_theta;
    auto t1c = sqrt(tc * tc - d_oc * d_oc + radius * radius);
    auto dist = tc - t1c;

    auto p = ray_origin + dir * dist;
    auto ng = normalize(p - origin);
    auto area = 4 * pi * radius * radius;

    auto frame = Frame::make(ng);
    auto dpdu = frame.s();
    auto dpdv = frame.t();
    return {.g = {.p = p,
                  .n = ng,
                  .area = area},
            .ps = p,
            .ns = ng,
            .dpdu = dpdu,
            .dpdv = dpdv,
            .uv = make_float2(0.f)};
}

}// namespace luisa::render
