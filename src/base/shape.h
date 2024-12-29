//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <runtime/rtx/mesh.h>
#include <runtime/rtx/procedural_primitive.h>
#include <runtime/rtx/aabb.h>
#include <util/vertex.h>
#include <base/scene_node.h>
#include <base/scene.h>

namespace luisa::render {

class Surface;
class Light;
class Medium;
class Subsurface;
class Transform;

using compute::AccelOption;
using compute::Triangle;
using compute::AABB;

struct MeshView {
    luisa::span<const Vertex> vertices;
    luisa::span<const Triangle> triangles;
};

struct SpheresView {
    luisa::span<const AABB> aabbs;
};


class Shape : public SceneNode {

public:
    class Handle;

public:
    static constexpr auto property_flag_has_vertex_normal = 1u << 0u;
    static constexpr auto property_flag_has_vertex_uv = 1u << 1u;
    static constexpr auto property_flag_has_surface = 1u << 2u;
    static constexpr auto property_flag_has_light = 1u << 3u;
    static constexpr auto property_flag_has_medium = 1u << 4u;
    static constexpr auto property_flag_has_subsurface = 1u << 5u;
    static constexpr auto property_flag_maybe_non_opaque = 1u << 6u;
    static constexpr auto property_flag_triangle = 1u << 7u;

private:
    const Surface *_surface;
    const Light *_light;
    const Medium *_medium;
    const Subsurface *_subsurface;
    const Transform *_transform;

public:
    Shape(Scene *scene, const SceneNodeDesc *desc) noexcept;
    virtual void update(Scene *scene, const SceneNodeDesc *desc) noexcept override;
    [[nodiscard]] virtual luisa::string info() const noexcept override;
    [[nodiscard]] const Surface *surface() const noexcept;
    [[nodiscard]] const Light *light() const noexcept;
    [[nodiscard]] const Medium *medium() const noexcept;
    [[nodiscard]] const Subsurface *subsurface() const noexcept;
    [[nodiscard]] const Transform *transform() const noexcept;
    [[nodiscard]] virtual bool visible() const noexcept;
    [[nodiscard]] virtual float shadow_terminator_factor() const noexcept;
    [[nodiscard]] virtual float intersection_offset_factor() const noexcept;
    [[nodiscard]] virtual float clamp_normal_factor() const noexcept;
    [[nodiscard]] virtual bool is_mesh() const noexcept;
    [[nodiscard]] virtual bool is_spheres() const noexcept;
    [[nodiscard]] virtual bool empty() const noexcept;
    [[nodiscard]] virtual uint vertex_properties() const noexcept;
    [[nodiscard]] bool has_vertex_normal() const noexcept;
    [[nodiscard]] bool has_vertex_uv() const noexcept;
    [[nodiscard]] virtual MeshView mesh() const noexcept;                               // empty if the shape is not a mesh
    [[nodiscard]] virtual SpheresView spheres() const noexcept;                         // empty if the shape is not spheres
    [[nodiscard]] virtual luisa::span<const Shape *const> children() const noexcept;    // empty if the shape is a mesh
    [[nodiscard]] virtual AccelOption build_option() const noexcept;                    // accel struct build quality, only considered for meshes
};

template<typename BaseShape>
class ShadingShapeWrapper : public BaseShape {

private:
    float _shadow_terminator;
    float _intersection_offset;
    float _clamp_normal;

public:
    ShadingShapeWrapper(Scene *scene, const SceneNodeDesc *desc) noexcept :
        BaseShape{scene, desc},
        _shadow_terminator{std::clamp(
            desc->property_float_or_default("shadow_terminator", scene->shadow_terminator_factor()),
        0.f, 1.f)},
        _intersection_offset{std::clamp(
            desc->property_float_or_default("intersection_offset", scene->intersection_offset_factor()),
        0.f, 1.f)},
        _clamp_normal{std::clamp(
            desc->property_float_or_default("clamp_normal", scene->clamp_normal_factor()),
        0.f, 180.f)} {}
    
    [[nodiscard]] virtual luisa::string info() const noexcept override {
        return luisa::format("{} clamp_normal=[{}]", BaseShape::info(), _clamp_normal);
    }

    [[nodiscard]] float shadow_terminator_factor() const noexcept override {
        return _shadow_terminator;
    }
    [[nodiscard]] float intersection_offset_factor() const noexcept override {
        return _intersection_offset;
    }
    [[nodiscard]] float clamp_normal_factor() const noexcept override {
        return _clamp_normal;
    }
};

template<typename BaseShape>
class VisibilityShapeWrapper : public BaseShape {

private:
    bool _visible;

public:
    VisibilityShapeWrapper(Scene *scene, const SceneNodeDesc *desc) noexcept :
        BaseShape{scene, desc}, _visible{desc->property_bool_or_default("visible", true)} {}
        
    [[nodiscard]] bool visible() const noexcept override { return _visible; }
};

using compute::Expr;
using compute::Float;
using compute::UInt;

class Shape::Handle {

public:
    static constexpr auto property_flag_bits = 10u;
    static constexpr auto property_flag_mask = (1u << property_flag_bits) - 1u;
    static constexpr auto buffer_base_max = (1u << (32u - property_flag_bits)) - 1u;

    static constexpr auto shadow_term_bits = 10u;
    static constexpr auto shadow_term_mask = (1u << shadow_term_bits) - 1u;
    static constexpr auto shadow_term_offset = 0u;
    static constexpr auto inter_offset_bits = 10u;
    static constexpr auto inter_offset_mask = (1u << inter_offset_bits) - 1u;
    static constexpr auto inter_offset_offset = shadow_term_offset + shadow_term_bits;
    static constexpr auto clamp_normal_bits = 32u - shadow_term_bits - inter_offset_bits;
    static constexpr auto clamp_normal_mask = (1u << clamp_normal_bits) - 1u;
    static constexpr auto clamp_normal_offset = inter_offset_offset + inter_offset_bits;

    static constexpr auto alias_bindless_offset = 0u;
    static constexpr auto pdf_bindless_offset = 1u;
    static constexpr auto vertices_bindless_offset = 2u;
    static constexpr auto triangles_bindless_offset = 3u;
    static constexpr auto aabbs_bindless_offset = 2u;

private:
    UInt _buffer_base;
    UInt _properties;
    UInt _primitive_count;
    UInt _surface_tag;
    UInt _light_tag;
    UInt _medium_tag;
    UInt _subsurface_tag;
    Float _shadow_terminator;
    Float _intersection_offset;
    Float _clamp_normal;

private:
    Handle(
        Expr<uint> buffer_base, Expr<uint> flags, Expr<uint> primitive_count,
        Expr<uint> surface_tag, Expr<uint> light_tag,
        Expr<uint> medium_tag, Expr<uint> subsurface_tag,
        Expr<float> shadow_terminator, Expr<float> intersection_offset, Expr<float> clamp_normal
    ) noexcept :
        _buffer_base{buffer_base},
        _properties{flags},
        _primitive_count{primitive_count},
        _surface_tag{surface_tag},
        _light_tag{light_tag},
        _medium_tag{medium_tag},
        _subsurface_tag{subsurface_tag},
        _shadow_terminator{shadow_terminator},
        _intersection_offset{intersection_offset},
        _clamp_normal{clamp_normal} {}

public:
    Handle() noexcept = default;
    [[nodiscard]] static std::pair<uint3, uint4> encode(
        uint buffer_base, uint flags, uint primitive_count,
        uint surface_tag, uint light_tag, uint medium_tag, uint subsurface_tag,
        float shadow_terminator, float intersection_offset, float clamp_normal) noexcept;
    [[nodiscard]] static Shape::Handle decode(Expr<uint3> comp_geom, Expr<uint4> comp_prop) noexcept;

public:
    [[nodiscard]] auto geometry_buffer_base() const noexcept { return _buffer_base; }
    [[nodiscard]] auto vertex_buffer_id() const noexcept { return geometry_buffer_base() + Shape::Handle::vertices_bindless_offset; }
    [[nodiscard]] auto triangle_buffer_id() const noexcept { return geometry_buffer_base() + Shape::Handle::triangles_bindless_offset; }
    [[nodiscard]] auto aabb_buffer_id() const noexcept { return geometry_buffer_base() + Shape::Handle::aabbs_bindless_offset; }
    [[nodiscard]] auto alias_table_buffer_id() const noexcept { return geometry_buffer_base() + Shape::Handle::alias_bindless_offset; }
    [[nodiscard]] auto pdf_buffer_id() const noexcept { return geometry_buffer_base() + Shape::Handle::pdf_bindless_offset; }
    [[nodiscard]] auto property_flags() const noexcept { return _properties; }
    [[nodiscard]] auto test_property_flag(uint flag) const noexcept { return (property_flags() & flag) != 0u; }
    [[nodiscard]] auto has_vertex_normal() const noexcept { return test_property_flag(Shape::property_flag_has_vertex_normal); }
    [[nodiscard]] auto has_vertex_uv() const noexcept { return test_property_flag(Shape::property_flag_has_vertex_uv); }
    [[nodiscard]] auto has_light() const noexcept { return test_property_flag(Shape::property_flag_has_light); }
    [[nodiscard]] auto has_surface() const noexcept { return test_property_flag(Shape::property_flag_has_surface); }
    [[nodiscard]] auto has_medium() const noexcept { return test_property_flag(Shape::property_flag_has_medium); }
    [[nodiscard]] auto has_subsurface() const noexcept { return test_property_flag(Shape::property_flag_has_subsurface); }
    [[nodiscard]] auto maybe_non_opaque() const noexcept { return test_property_flag(Shape::property_flag_maybe_non_opaque); }
    [[nodiscard]] auto is_triangle() const noexcept { return test_property_flag(Shape::property_flag_triangle); }
    [[nodiscard]] auto primitive_count() const noexcept { return _primitive_count; }
    [[nodiscard]] auto surface_tag() const noexcept { return _surface_tag; }
    [[nodiscard]] auto light_tag() const noexcept { return _light_tag; }
    [[nodiscard]] auto medium_tag() const noexcept { return _medium_tag; }
    [[nodiscard]] auto subsurface_tag() const noexcept { return _subsurface_tag; }
    [[nodiscard]] auto shadow_terminator_factor() const noexcept { return _shadow_terminator; }
    [[nodiscard]] auto intersection_offset_factor() const noexcept { return _intersection_offset; }
    [[nodiscard]] auto clamp_normal_factor() const noexcept { return _clamp_normal; }
};

// class Shape::PropertyHandle {
// public:
//     static constexpr auto light_tag_bits = 13u;
//     static constexpr auto light_tag_max = (1u << light_tag_bits) - 1u;
//     static constexpr auto light_tag_offset = 0u;
//     static constexpr auto surface_tag_bits = 13u;
//     static constexpr auto surface_tag_max = (1u << surface_tag_bits) - 1u;
//     static constexpr auto surface_tag_offset = light_tag_offset + light_tag_bits;
//     static constexpr auto medium_tag_bits = 32u - light_tag_bits - surface_tag_bits;
//     static constexpr auto medium_tag_max = (1u << medium_tag_bits) - 1u;
//     static constexpr auto medium_tag_offset = surface_tag_offset + surface_tag_bits;
// private:
// private:
//     PropertyHandle(
//     ) noexcept: {}

// public:
//     PropertyHandle() noexcept = default;
//     [[nodiscard]] static uint4 encode(
//     [[nodiscard]] static Shape::PropertyHandle decode(Expr<uint4> compressed) noexcept;
    
    // LUISA_ASSERT(surface_tag <= surface_tag_max, "Invalid surface tag: {}.", surface_tag);
    // LUISA_ASSERT(light_tag <= light_tag_max, "Invalid light tag: {}.", light_tag);
    // LUISA_ASSERT(medium_tag <= medium_tag_max, "Invalid medium tag: {}.", medium_tag);
    // LUISA_ASSERT(subsurface_tag <= subsurface_tag_max, "Invalid subsurface tag: {}.", medium_tag);
    // auto tags =
    // (surface_tag << surface_tag_offset) |
    // (light_tag << light_tag_offset) |
    // (medium_tag << medium_tag_offset);
    // auto surface_tag = (tags >> surface_tag_offset) & surface_tag_max;
    // auto light_tag = (tags >> light_tag_offset) & light_tag_max;
    // auto medium_tag = (tags >> medium_tag_offset) & medium_tag_max;
// };

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Shape::Handle)