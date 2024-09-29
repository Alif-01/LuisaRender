//
// Created by Mike Smith on 2022/9/14.
//

#pragma once

#include <luisa/dsl/syntax.h>
#include <luisa/runtime/rtx/accel.h>
#include <base/transform.h>
#include <base/light.h>
#include <base/shape.h>
#include <base/interaction.h>

namespace luisa::render {

using compute::Accel;
using compute::AccelOption;
using compute::Buffer;
using compute::Expr;
using compute::Var;
using compute::Float4x4;
using compute::Mesh;
using compute::ProceduralPrimitive;
using compute::Ray;
using compute::CommittedHit;
using compute::SurfaceHit;
using compute::ProceduralHit;
using compute::SurfaceCandidate;
using compute::ProceduralCandidate;

class Pipeline;

class Geometry {

public:
    struct ShapeGeometry {
        void *resource;
        uint buffer_id_base;
    };

    static constexpr float inv_sqrt3 = 0.57735026918962576450914878050196f;

private:
    Pipeline &_pipeline;
    Accel _accel;
    TransformTree _transform_tree;
    luisa::vector<uint> _resource_store;
    luisa::vector<Light::Handle> _instanced_lights;
    luisa::vector<uint4> _instances;
    luisa::vector<InstancedTransform> _dynamic_transforms;
    Buffer<uint4> _instance_buffer;
    // float3 _world_min;
    // float3 _world_max;
    bool _any_non_opaque{false};

private:
    void _process_shape(
        CommandBuffer &command_buffer, const Shape *shape, float init_time,
        const Surface *overridden_surface = nullptr,
        const Light *overridden_light = nullptr,
        const Medium *overridden_medium = nullptr,
        bool overridden_visible = true) noexcept;

    [[nodiscard]] Bool _alpha_skip(const Interaction &it, Expr<float> u) const noexcept;
    [[nodiscard]] Bool _alpha_skip(const Var<Ray> &ray, const Var<SurfaceHit> &hit) const noexcept;
    [[nodiscard]] Bool _alpha_skip(const Var<Ray> &ray, const Var<ProceduralHit> &hit) const noexcept;
    void _procedural_filter(ProceduralCandidate &c) const noexcept;

public:
    explicit Geometry(Pipeline &pipeline) noexcept : _pipeline{pipeline} {};
    ~Geometry() noexcept;
    void build(CommandBuffer &command_buffer, luisa::span<const Shape *const> shapes, float init_time) noexcept;
    bool update(CommandBuffer &command_buffer, float time) noexcept;
    [[nodiscard]] auto instances() const noexcept { return luisa::span{_instances}; }
    [[nodiscard]] auto light_instances() const noexcept { return luisa::span{_instanced_lights}; }
    // [[nodiscard]] auto world_min() const noexcept { return _world_min; }
    // [[nodiscard]] auto world_max() const noexcept { return _world_max; }
    [[nodiscard]] Var<CommittedHit> trace_closest(const Var<Ray> &ray) const noexcept;
    [[nodiscard]] Var<bool> trace_any(const Var<Ray> &ray) const noexcept;
    [[nodiscard]] luisa::shared_ptr<Interaction> interaction(const Var<Ray> &ray, const Var<CommittedHit> &hit) const noexcept;
    [[nodiscard]] luisa::shared_ptr<Interaction> interaction(const Var<Ray> &ray, const Var<SurfaceHit> &hit) const noexcept;
    [[nodiscard]] luisa::shared_ptr<Interaction> interaction(const Var<Ray> &ray, const Var<ProceduralHit> &hit) const noexcept;
    [[nodiscard]] Interaction triangle_interaction(
        const Var<Ray> &ray, Expr<uint> inst_id, Expr<uint> prim_id, Expr<float3> bary) const noexcept;
    [[nodiscard]] Interaction aabb_interaction(
        const Var<Ray> &ray, Expr<uint> inst_id, Expr<uint> prim_id) const noexcept;
    [[nodiscard]] Shape::Handle instance(Expr<uint> inst_id) const noexcept;
    [[nodiscard]] Float4x4 instance_to_world(Expr<uint> inst_id) const noexcept;
    [[nodiscard]] Var<Triangle> triangle(const Shape::Handle &instance, Expr<uint> triangle_id) const noexcept;
    [[nodiscard]] Var<Vertex> vertex(const Shape::Handle &instance, Expr<uint> vertex_id) const noexcept;
    [[nodiscard]] Var<AABB> aabb(const Shape::Handle &instance, Expr<uint> aabb_id) const noexcept;
    [[nodiscard]] GeometryAttribute geometry_point(
        const Shape::Handle &instance, const Var<Triangle> &triangle,
        const Var<float3> &bary, const Var<float4x4> &shape_to_world) const noexcept;
    [[nodiscard]] GeometryAttribute geometry_point(
        const Shape::Handle &instance, const Var<AABB> &ab,
        const Var<float3> &w, const Var<float4x4> &shape_to_world) const noexcept;
    [[nodiscard]] ShadingAttribute shading_point(
        const Shape::Handle &instance, const Var<Triangle> &triangle,
        const Var<float3> &bary, const Var<float4x4> &shape_to_world) const noexcept;
    [[nodiscard]] ShadingAttribute shading_point(
        const Shape::Handle &instance, const Var<AABB> &ab,
        const Var<Ray> &ray, const Var<float4x4> &shape_to_world) const noexcept;
    [[nodiscard]] auto intersect(const Var<Ray> &ray) const noexcept { return interaction(ray, trace_closest(ray)); }
    [[nodiscard]] auto intersect_any(const Var<Ray> &ray) const noexcept { return trace_any(ray); }
};

}// namespace luisa::render
