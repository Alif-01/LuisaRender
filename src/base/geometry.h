//
// Created by Mike Smith on 2022/9/14.
//

#pragma once

#include <luisa/dsl/syntax.h>
#include <luisa/runtime/rtx/accel.h>
#include <util/sampling.h>
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
    class ShapeData {
    public:
        ShapeData() noexcept;
        virtual ~ShapeData() noexcept = default;
        void build(Pipeline &pipeline, uint prim_count) noexcept;
        virtual void register_bindless(Pipeline &pipeline) noexcept;
        virtual void update_bindless(Pipeline &pipeline) noexcept;
        [[nodiscard]] bool registered() const noexcept;

    public:
        Buffer<AliasEntry> alias_table;
        Buffer<float> pdf;
        uint primitive_count;
        uint buffer_id_base;
    };

    class MeshData : public ShapeData {
    public:
        ~MeshData() noexcept = default;
        void build(Pipeline &pipeline, uint vertex_count, uint triangle_count, AccelOption build_option) noexcept;
        void register_bindless(Pipeline &pipeline) noexcept override;
        void update_bindless(Pipeline &pipeline) noexcept override;
    public:
        Buffer<Vertex> vertices;
        Buffer<Triangle> triangles;
        Mesh mesh;
    };
    
    class SpheresData : public ShapeData {
    public:
        ~SpheresData() noexcept = default;
        void build(Pipeline &pipeline, uint sphere_count, AccelOption build_option) noexcept;
        void register_bindless(Pipeline &pipeline) noexcept override;
        void update_bindless(Pipeline &pipeline) noexcept override;
    public:
        Buffer<AABB> aabbs;
        ProceduralPrimitive procedural;
    };

    class InstancedTransform {
    private:
        const TransformTree::Node *_node;
        size_t _instance_id;

    public:
        InstancedTransform(const TransformTree::Node *node, size_t inst) noexcept:
            _node{node}, _instance_id{inst} {}
        [[nodiscard]] auto instance_id() const noexcept { return _instance_id; }
        [[nodiscard]] auto matrix(float time) const noexcept {
            return _node == nullptr ? make_float4x4(1.0f) : _node->matrix(time);
        }
    };

    static constexpr float inv_sqrt3 = 0.57735026918962576450914878050196f;

private:
    Pipeline &_pipeline;

    luisa::unordered_map<uint64_t, uint> shape_data_ids;
    luisa::vector<luisa::unique_ptr<ShapeData>> _shapes_data;

    TransformTree _transform_tree;
    luisa::vector<InstancedTransform> _dynamic_transforms;

    Accel _accel;
    luisa::vector<uint3> _instances_geometry;
    Buffer<uint3> _instance_geometry_buffer;
    luisa::vector<uint4> _instances_property;
    Buffer<uint4> _instance_property_buffer;

    luisa::vector<uint> _light_instances;
    Buffer<uint> _light_instance_buffer;
    bool _any_non_opaque{false};
    // float3 _world_min;
    // float3 _world_max;

private:
    void _process_shape(
        CommandBuffer &command_buffer, float time,
        const Shape *shape,
        const Surface *overridden_surface = nullptr,
        const Light *overridden_light = nullptr,
        const Medium *overridden_medium = nullptr,
        const Subsurface *overridden_subsurface = nullptr,
        bool overridden_visible = true,
        uint64_t parent_hash = luisa::hash64_default_seed) noexcept;

    [[nodiscard]] Bool _alpha_skip(const Interaction &it, Expr<float> u) const noexcept;
    [[nodiscard]] Bool _alpha_skip(const Var<Ray> &ray, const Var<SurfaceHit> &hit) const noexcept;
    [[nodiscard]] Bool _alpha_skip(const Var<Ray> &ray, const Var<ProceduralHit> &hit) const noexcept;
    // void _procedural_filter(ProceduralCandidate &c) const noexcept;
    [[nodiscard]] Float _procedural_intersect(const Var<Ray> &ray, const Var<ProceduralHit> &hit) const noexcept;

public:
    explicit Geometry(Pipeline &pipeline) noexcept;
    ~Geometry() noexcept = default;
    void update(CommandBuffer &command_buffer, const luisa::unordered_set<Shape *> &shapes, float time) noexcept;
    void shutter_update(CommandBuffer &command_buffer, float time) noexcept;
    // [[nodiscard]] auto instances() const noexcept { return luisa::span{_instances}; }
    // [[nodiscard]] auto light_instances() const noexcept { return luisa::span{_light_instances}; }
    [[nodiscard]] Var<CommittedHit> trace_closest(const Var<Ray> &ray_in) const noexcept;
    [[nodiscard]] Var<bool> trace_any(const Var<Ray> &ray_in) const noexcept;
    [[nodiscard]] luisa::shared_ptr<Interaction> interaction(const Var<Ray> &ray, const Var<CommittedHit> &hit) const noexcept;
    [[nodiscard]] luisa::shared_ptr<Interaction> interaction(const Var<Ray> &ray, const Var<SurfaceHit> &hit) const noexcept;
    [[nodiscard]] luisa::shared_ptr<Interaction> interaction(const Var<Ray> &ray, const Var<ProceduralHit> &hit) const noexcept;
    [[nodiscard]] Interaction triangle_interaction(
        const Var<Ray> &ray, Expr<uint> inst_id, Expr<uint> prim_id, Expr<float3> bary) const noexcept;
    [[nodiscard]] Interaction aabb_interaction(
        const Var<Ray> &ray, Expr<uint> inst_id, Expr<uint> prim_id) const noexcept;
    [[nodiscard]] Shape::Handle instance(Expr<uint> inst_id) const noexcept;
    [[nodiscard]] UInt light_instance(Expr<uint> inst_id) const noexcept;
    [[nodiscard]] Float4x4 instance_to_world(Expr<uint> inst_id) const noexcept;
    [[nodiscard]] Var<Triangle> triangle(const Shape::Handle &instance_geom, Expr<uint> triangle_id) const noexcept;
    [[nodiscard]] Var<Vertex> vertex(const Shape::Handle &instance_geom, Expr<uint> vertex_id) const noexcept;
    [[nodiscard]] Var<AABB> aabb(const Shape::Handle &instance_geom, Expr<uint> aabb_id) const noexcept;
    [[nodiscard]] GeometryAttribute geometry_point(
        const Shape::Handle &instance_geom, const Var<Triangle> &triangle,
        const Var<float3> &bary, const Var<float4x4> &shape_to_world) const noexcept;
    [[nodiscard]] GeometryAttribute geometry_point(
        const Shape::Handle &instance_geom, const Var<AABB> &ab,
        const Var<float3> &w, const Var<float4x4> &shape_to_world) const noexcept;
    [[nodiscard]] ShadingAttribute shading_point(
        const Shape::Handle &instance_geom, const Var<Triangle> &triangle,
        const Var<float3> &bary, const Var<float4x4> &shape_to_world) const noexcept;
    [[nodiscard]] ShadingAttribute shading_point(
        const Shape::Handle &instance_geom, const Var<AABB> &ab,
        const Var<Ray> &ray, const Var<float4x4> &shape_to_world) const noexcept;
    [[nodiscard]] auto intersect(const Var<Ray> &ray) const noexcept { return interaction(ray, trace_closest(ray)); }
    [[nodiscard]] auto intersect_any(const Var<Ray> &ray) const noexcept { return trace_any(ray); }
};

}// namespace luisa::render
