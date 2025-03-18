//
// Created by Mike Smith on 2022/9/14.
//

#include <util/thread_pool.h>
#include <base/geometry.h>
#include <base/pipeline.h>

namespace luisa::render {

Geometry::ShapeData::ShapeData() noexcept:
    primitive_count{0u}, buffer_id_base{Pipeline::bindless_array_capacity} {}

void Geometry::ShapeData::build(Pipeline &pipeline, uint prim_count) noexcept {
    primitive_count = prim_count;
    alias_table = pipeline.device().create_buffer<AliasEntry>(prim_count);
    pdf = pipeline.device().create_buffer<float>(prim_count);
}

void Geometry::ShapeData::register_bindless(Pipeline &pipeline) noexcept {
    buffer_id_base = pipeline.register_bindless(alias_table.view());
    auto pdf_id = pipeline.register_bindless(pdf.view());
    LUISA_ASSERT(pdf_id - buffer_id_base == Shape::Handle::pdf_bindless_offset,
                 "Invalid pdf bindless buffer id.");
}

void Geometry::ShapeData::update_bindless(Pipeline &pipeline) noexcept {
    pipeline.update_bindless(alias_table.view(), buffer_id_base + Shape::Handle::alias_bindless_offset);
    pipeline.update_bindless(pdf.view(), buffer_id_base + Shape::Handle::pdf_bindless_offset);
}

bool Geometry::ShapeData::registered() const noexcept {
    return buffer_id_base < Pipeline::bindless_array_capacity;
}

void Geometry::MeshData::build(Pipeline &pipeline, uint vertex_count, uint triangle_count, AccelOption build_option) noexcept {
    ShapeData::build(pipeline, triangle_count);
    vertices = pipeline.device().create_buffer<Vertex>(vertex_count);
    triangles = pipeline.device().create_buffer<Triangle>(triangle_count);
    mesh = pipeline.device().create<Mesh>(vertices.view(), triangles.view(), build_option);
}

void Geometry::MeshData::register_bindless(Pipeline &pipeline) noexcept {
    ShapeData::register_bindless(pipeline);
    auto vertices_id = pipeline.register_bindless(vertices.view());
    auto triangles_id = pipeline.register_bindless(triangles.view());
    LUISA_ASSERT(vertices_id - buffer_id_base == Shape::Handle::vertices_bindless_offset,
                 "Invalid vertices bindless buffer id.");
    LUISA_ASSERT(triangles_id - buffer_id_base == Shape::Handle::triangles_bindless_offset,
                 "Invalid triangles bindless buffer id.");
}

void Geometry::MeshData::update_bindless(Pipeline &pipeline) noexcept {
    ShapeData::update_bindless(pipeline);
    pipeline.update_bindless(vertices.view(), buffer_id_base + Shape::Handle::vertices_bindless_offset);
    pipeline.update_bindless(triangles.view(), buffer_id_base + Shape::Handle::triangles_bindless_offset);
}

void Geometry::SpheresData::build(Pipeline &pipeline, uint sphere_count, AccelOption build_option) noexcept {
    ShapeData::build(pipeline, sphere_count);
    aabbs = pipeline.device().create_buffer<AABB>(sphere_count);
    procedural = pipeline.device().create<ProceduralPrimitive>(aabbs.view(), build_option);
}

void Geometry::SpheresData::register_bindless(Pipeline &pipeline) noexcept {
    ShapeData::register_bindless(pipeline);
    auto aabbs_id = pipeline.register_bindless(aabbs.view());
    LUISA_ASSERT(aabbs_id - buffer_id_base == Shape::Handle::aabbs_bindless_offset,
                 "Invalid aabbs bindless buffer id.");
}

void Geometry::SpheresData::update_bindless(Pipeline &pipeline) noexcept {
    ShapeData::update_bindless(pipeline);
    pipeline.update_bindless(aabbs.view(), buffer_id_base + Shape::Handle::aabbs_bindless_offset);
}

Geometry::Geometry(Pipeline &pipeline) noexcept:
    _pipeline{pipeline} {}

void Geometry::update(
    CommandBuffer &command_buffer, const luisa::unordered_set<Shape *> &shapes, float time
) noexcept {
    global_thread_pool().synchronize();
    command_buffer << compute::synchronize();
    _accel = _pipeline.device().create_accel({});    // TODO: AccelOption
    _instances_geometry.clear();
    _instances_property.clear();
    _light_instances.clear();
    _any_non_opaque = false;

    for (auto shape : shapes) { _process_shape(command_buffer, time, shape); }

    if (_instances_geometry.size() > 0) {
        if (!_instance_geometry_buffer || _instance_geometry_buffer.size() != _instances_geometry.size()) {
            _instance_geometry_buffer = _pipeline.device().create_buffer<uint3>(_instances_geometry.size());
            _instance_property_buffer = _pipeline.device().create_buffer<uint4>(_instances_property.size());
        }
        command_buffer << _instance_geometry_buffer.copy_from(_instances_geometry.data())
                       << _instance_property_buffer.copy_from(_instances_property.data());
    }
    if (_light_instances.size() > 0) {
        if (!_light_instance_buffer || _light_instance_buffer.size() != _light_instances.size()) {
            _light_instance_buffer = _pipeline.device().create_buffer<uint>(_light_instances.size());
        }
        command_buffer << _light_instance_buffer.copy_from(_light_instances.data());
    }
    if (_accel.dirty()) {
        command_buffer << _accel.build();
    }
}

void Geometry::shutter_update(
    CommandBuffer &command_buffer, float time
) noexcept {
    if (!_dynamic_transforms.empty()) {
        if (_dynamic_transforms.size() < 128u) {
            for (auto t : _dynamic_transforms) {
                _accel.set_transform_on_update(t.instance_id(), t.matrix(time));
            }
        } else {
            global_thread_pool().parallel(
                _dynamic_transforms.size(),
                [this, time](auto i) noexcept {
                    auto t = _dynamic_transforms[i];
                    _accel.set_transform_on_update(t.instance_id(), t.matrix(time));
                });
            global_thread_pool().synchronize();
        }
        command_buffer << _accel.build();
    }
}

void Geometry::_process_shape(
    CommandBuffer &command_buffer, float time,
    const Shape *shape,
    const Surface *overridden_surface,
    const Light *overridden_light,
    const Medium *overridden_medium,
    const Subsurface *overridden_subsurface,
    bool overridden_visible,
    uint64_t parent_hash) noexcept {

    auto surface = overridden_surface == nullptr ? shape->surface() : overridden_surface;
    auto light = overridden_light == nullptr ? shape->light() : overridden_light;
    auto medium = overridden_medium == nullptr ? shape->medium() : overridden_medium;
    auto subsurface = overridden_subsurface == nullptr ? shape->subsurface() : overridden_subsurface;
    auto visible = overridden_visible && shape->visible();
    auto hash = luisa::pointer_hash<Shape>()(shape, parent_hash);

    if (shape->is_mesh() || shape->is_spheres()) {
        if (shape->empty()) return;
        auto [iter, first_def] = shape_data_ids.emplace(hash, _instances_geometry.size());
        auto data_id = iter->second;
        auto accel_id = static_cast<uint>(_accel.size());
        auto properties = shape->vertex_properties();
        auto surface_tag = 0u;
        auto light_tag = 0u;
        auto medium_tag = 0u;
        auto subsurface_tag = 0u;
        luisa::vector<float> primitive_areas;

        // transform
        auto [transform_node, transform_static] = _transform_tree.leaf(shape->transform());
        if (!transform_static) {
            _dynamic_transforms.emplace_back(transform_node, accel_id);
        }
        auto object_to_world = transform_node == nullptr ? make_float4x4(1.0f) : transform_node->matrix(time);

        if (surface != nullptr && !surface->is_null()) {
            surface_tag = _pipeline.register_surface(command_buffer, surface);
            properties |= Shape::property_flag_has_surface;
            if (_pipeline.surfaces().impl(surface_tag)->maybe_non_opaque()) {
                properties |= Shape::property_flag_maybe_non_opaque;
                _any_non_opaque = true;
            }
        }
        if (light != nullptr && !light->is_null()) {
            light_tag = _pipeline.register_light(command_buffer, light);
            properties |= Shape::property_flag_has_light;
            _light_instances.emplace_back(accel_id);
        }
        if (medium != nullptr && !medium->is_null()) {
            medium_tag = _pipeline.register_medium(command_buffer, medium);
            properties |= Shape::property_flag_has_medium;
        }
        if (subsurface != nullptr && !subsurface->is_null()) {
            subsurface_tag = _pipeline.register_subsurface(command_buffer, subsurface);
            properties |= Shape::property_flag_has_subsurface;
        }

        if (shape->is_mesh()) {
            if (first_def) {
                _shapes_data.emplace_back(luisa::make_unique<MeshData>());
            }
            auto mesh_data = dynamic_cast<MeshData *>(_shapes_data[data_id].get());
            auto [vertices, triangles] = shape->mesh();
            if (mesh_data->primitive_count != triangles.size()) {
                mesh_data->build(_pipeline, vertices.size(), triangles.size(), shape->build_option());
                if (mesh_data->registered()) {
                    mesh_data->update_bindless(_pipeline);
                } else {
                    mesh_data->register_bindless(_pipeline);
                }
            }
            properties |= Shape::property_flag_triangle;
            // auto hash = luisa::hash64(
            //     triangles.data(), triangles.size_bytes(), luisa::hash64(
            //     vertices.data(), vertices.size_bytes(), luisa::hash64_default_seed
            // ));
            // if (auto mesh_iter = _mesh_cache.find(hash);
            //     mesh_iter != _mesh_cache.end()) {
            //     return mesh_iter->second;
            // }

            // create mesh
            command_buffer << mesh_data->vertices.copy_from(vertices.data())
                           << mesh_data->triangles.copy_from(triangles.data())
                           << mesh_data->mesh.build()
                           << compute::commit();

            // compute alias table
            primitive_areas.resize(triangles.size());
            for (auto i = 0u; i < triangles.size(); ++i) {
                auto t = triangles[i];
                auto p0 = vertices[t.i0].position();
                auto p1 = vertices[t.i1].position();
                auto p2 = vertices[t.i2].position();
                primitive_areas[i] = std::abs(length(cross(p1 - p0, p2 - p0)));
            }

            _accel.emplace_back(
                mesh_data->mesh, object_to_world, visible,
                (properties & Shape::property_flag_maybe_non_opaque) == 0u, accel_id
            );
        } else {
            if (first_def) {
                _shapes_data.emplace_back(luisa::make_unique<SpheresData>());
            }
            auto spheres_data = dynamic_cast<SpheresData *>(_shapes_data[data_id].get());
            auto [aabbs] = shape->spheres();
            if (spheres_data->primitive_count != aabbs.size()) {
                spheres_data->build(_pipeline, aabbs.size(), shape->build_option());
                if (spheres_data->registered()) {
                    spheres_data->update_bindless(_pipeline);
                } else {
                    spheres_data->register_bindless(_pipeline);
                }
            }
            command_buffer << spheres_data->aabbs.copy_from(aabbs.data())
                           << spheres_data->procedural.build()
                           << compute::commit();

            // compute alias table
            primitive_areas.resize(aabbs.size());
            for (auto i = 0u; i < aabbs.size(); ++i) {
                auto diameter = aabbs[i].packed_max[0] - aabbs[i].packed_min[0];
                primitive_areas[i] = diameter * diameter;
            }
            
            _accel.emplace_back(spheres_data->procedural, object_to_world, visible, accel_id);
        }

        auto instance_data = _shapes_data[data_id].get();
        auto [alias_table, pdf] = create_alias_table(primitive_areas);
        command_buffer << instance_data->alias_table.copy_from(alias_table.data())
                       << instance_data->pdf.copy_from(pdf.data())
                       << compute::commit();
            
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

        auto [comp_geom, comp_prop] = Shape::Handle::encode(
            instance_data->buffer_id_base, properties, primitive_areas.size(),
            surface_tag, light_tag, medium_tag, subsurface_tag,
            shape->has_vertex_normal() ? shape->shadow_terminator_factor() : 0.f,
            shape->intersection_offset_factor(),
            radians(shape->clamp_normal_factor())
        );
        _instances_geometry.emplace_back(comp_geom);
        _instances_property.emplace_back(comp_prop);

        LUISA_INFO(
            "Add shape {} to geometry: accel id: {}; size of instances: {} & accel: {}, "
            "dynamic transform: {}, matrix: {}, "
            "surface: {}, light: {}, medium: {}, properties: {}, prim_count: {}",
            shape->impl_type(), accel_id, _instances_geometry.size(), _accel.size(),
            _dynamic_transforms.size(), object_to_world,
            surface_tag, light_tag, medium_tag, properties, primitive_areas.size()
        );
            
    } else {
        _transform_tree.push(shape->transform());
        for (auto child : shape->children()) {
            _process_shape(command_buffer, time, child, surface, light, medium, subsurface, visible);
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
    auto u = cast<float>(xxhash32(make_uint4(hit.inst, hit.prim, as<uint2>(hit.bary)))) * 0x1p-32f;
    return _alpha_skip(*it, u);
}

Bool Geometry::_alpha_skip(const Var<Ray> &ray, const Var<ProceduralHit> &hit) const noexcept {
    auto it = interaction(ray, hit);
    auto u = cast<float>(xxhash32(make_uint2(hit.inst, hit.prim))) * 0x1p-32f;
    return _alpha_skip(*it, u);
}

// void Geometry::_procedural_filter(ProceduralCandidate &c) const noexcept {
//     Var<ProceduralHit> h = c.hit();
//     Var<Ray> ray = c.ray();
//     Var<AABB> ab = aabb(instance(h.inst), h.prim);
//     Float4x4 shape_to_world = instance_to_world(h.inst);
//     Float3x3 m = make_float3x3(shape_to_world);
//     Float3 t = make_float3(shape_to_world[3]);
//     Float3 aabb_min = m * ab->min() + t;
//     Float3 aabb_max = m * ab->max() + t;

//     Float3 origin = (aabb_min + aabb_max) * .5f;
//     Float radius = length(aabb_max - aabb_min) * .5f * inv_sqrt3;
//     Float3 ray_origin = ray->origin();
//     Float3 L = origin - ray_origin;
//     Float3 dir = ray->direction();
//     Float cos_theta = dot(dir, normalize(L));
//     $if (cos_theta > 0.f) {
//         Float d_oc = length(L);
//         Float tc = d_oc * cos_theta;
//         Float d = sqrt(d_oc * d_oc - tc * tc);
//         $if (d <= radius) {
//             Float t1c = sqrt(radius * radius - d * d);
//             Float dist = tc - t1c;
//             $if (dist < ray->t_max()) {
//                 c.commit(dist);
//             };
//         };
//     };
// }

Float Geometry::_procedural_intersect(const Var<Ray> &ray, const Var<ProceduralHit> &hit) const noexcept {
    Var<AABB> ab = aabb(instance(hit.inst), hit.prim);
    Float4x4 shape_to_world = instance_to_world(hit.inst);
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

    Float dist = 0.f;
    $if (cos_theta > 0.f) {
        Float d_oc = length(L);
        Float tc = d_oc * cos_theta;
        Float d = sqrt(d_oc * d_oc - tc * tc);
        $if (d <= radius) {
            Float t1c = sqrt(radius * radius - d * d);
            Float rd = tc - t1c;
            dist = ite(rd < ray->t_max(), rd, 0.f);
        };
    };
    return dist;
}

Var<CommittedHit> Geometry::trace_closest(const Var<Ray> &ray_in) const noexcept {
    if (!_any_non_opaque) {
        // happy path
        return _accel->traverse(ray_in, {})
            .on_procedural_candidate([&](ProceduralCandidate &c) noexcept {
                // this->_procedural_filter(c);
                Float dist = this->_procedural_intersect(c.ray(), c.hit());
                $if (dist > 0.f) { c.commit(dist); };
            })
            .trace();
    }
    // TODO: DirectX has bug with ray query, so we manually march the ray here
    // TODO: _accel->intersect only returns TriangleHit not CommittedHit
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
    return _accel->traverse(ray_in, {})
        .on_surface_candidate([&](SurfaceCandidate &c) noexcept {
            $if (!this->_alpha_skip(c.ray(), c.hit())) {
                c.commit();
            };
        })
        .on_procedural_candidate([&](ProceduralCandidate &c) noexcept {
            $if (!this->_alpha_skip(c.ray(), c.hit())) {
                // this->_procedural_filter(c);
                Float dist = this->_procedural_intersect(c.ray(), c.hit());
                $if (dist > 0.f) { c.commit(dist); };
            };
        })
        .trace();
}

Var<bool> Geometry::trace_any(const Var<Ray> &ray_in) const noexcept {
    if (!_any_non_opaque) {
        // happy path
        return !_accel->traverse_any(ray_in, {})
            .on_procedural_candidate([&](ProceduralCandidate &c) noexcept {
                // this->_procedural_filter(c);
                Float dist = this->_procedural_intersect(c.ray(), c.hit());
                $if (dist > 0.f) { c.commit(dist); };
            })
            .trace()->miss();
    } else {
        return !_accel->traverse_any(ray_in, {})
            .on_surface_candidate([&](SurfaceCandidate &c) noexcept {
                $if (!this->_alpha_skip(c.ray(), c.hit())) {
                    c.commit();
                };
            })
            .on_procedural_candidate([&](ProceduralCandidate &c) noexcept {
                $if (!this->_alpha_skip(c.ray(), c.hit())) {
                    // this->_procedural_filter(c);
                    Float dist = this->_procedural_intersect(c.ray(), c.hit());
                    $if (dist > 0.f) { c.commit(dist); };
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

    // $if (inst_id == 1u) {
    //     compute::device_log("Matrix: m=({}, {}, {}), inst={}, prim={}",
    //         m[3][0], m[3][1], m[3][2],
    //         inst_id, prim_id);
    // };

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
    return Shape::Handle::decode(
        _instance_geometry_buffer->read(inst_id),
        _instance_property_buffer->read(inst_id)
    );
}

UInt Geometry::light_instance(Expr<uint> inst_id) const noexcept {
    return _light_instance_buffer->read(inst_id);
}

Float4x4 Geometry::instance_to_world(Expr<uint> inst_id) const noexcept {
    return _accel->instance_transform(inst_id);
}

Var<Triangle> Geometry::triangle(const Shape::Handle &instance_geom, Expr<uint> triangle_id) const noexcept {
    return _pipeline.buffer<Triangle>(instance_geom.triangle_buffer_id()).read(triangle_id);
}

Var<Vertex> Geometry::vertex(const Shape::Handle &instance_geom, Expr<uint> vertex_id) const noexcept {
    return _pipeline.buffer<Vertex>(instance_geom.vertex_buffer_id()).read(vertex_id);
}

Var<AABB> Geometry::aabb(const Shape::Handle &instance_geom, Expr<uint> aabb_id) const noexcept {
    return _pipeline.buffer<AABB>(instance_geom.aabb_buffer_id()).read(aabb_id);
}

template<typename T>
[[nodiscard]] inline auto tri_interpolate(
    Expr<float3> uvw, const T &v0, const T &v1, const T &v2
) noexcept {
    return uvw.x * v0 + uvw.y * v1 + uvw.z * v2;
}

GeometryAttribute Geometry::geometry_point(
    const Shape::Handle &instance_geom, const Var<Triangle> &triangle,
    const Var<float3> &bary, const Var<float4x4> &shape_to_world
) const noexcept {
    auto v0 = vertex(instance_geom, triangle.i0);
    auto v1 = vertex(instance_geom, triangle.i1);
    auto v2 = vertex(instance_geom, triangle.i2);
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
    const Shape::Handle &instance_geom, const Var<AABB> &ab,
    const Var<float3> &w, const Var<float4x4> &shape_to_world
) const noexcept {
    auto m = make_float3x3(shape_to_world);
    auto t = make_float3(shape_to_world[3]);
    auto aabb_min = ab->min();
    auto aabb_max = ab->max();
    auto o_local = (aabb_min + aabb_max) * .5f;
    
    auto radius = length(aabb_max - aabb_min) * .5f * inv_sqrt3;
    auto p = m * (o_local + w * radius) + t;
    auto ng = normalize(m * w);
    auto area = 4 * pi * radius * radius;
    return {.p = p, .n = ng, .area = area};
}

ShadingAttribute Geometry::shading_point(
    const Shape::Handle &instance_geom, const Var<Triangle> &triangle,
    const Var<float3> &bary, const Var<float4x4> &shape_to_world
) const noexcept {
    auto v0 = vertex(instance_geom, triangle.i0);
    auto v1 = vertex(instance_geom, triangle.i1);
    auto v2 = vertex(instance_geom, triangle.i2);

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
    auto clamp_angle = instance_geom.clamp_normal_factor();
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
    auto ns = ite(instance_geom.has_vertex_normal(), normalize(m * ns_local), ng);
    auto uv = ite(instance_geom.has_vertex_uv(), tri_interpolate(bary, uv0, uv1, uv2), bary.yz());
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
    const Shape::Handle &instance_geom, const Var<AABB> &ab,
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
