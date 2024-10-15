//
// Created by Mike on 2021/12/14.
//

#include <sdl/scene_node_desc.h>
#include <base/texture.h>
#include <base/surface.h>
#include <base/light.h>
#include <base/medium.h>
#include <base/subsurface.h>
#include <base/transform.h>
#include <base/scene.h>
#include <base/shape.h>

namespace luisa::render {

uint encode_fixed_point(float x, uint mask) noexcept {
    // constexpr auto fixed_point_mask = (1u << fixed_point_bits) - 1u;
    // constexpr auto fixed_point_scale = 1.f / static_cast<float>(1u << fixed_point_bits);
    return static_cast<uint>(round(std::clamp(x, 0.f, 1.f) * static_cast<float>(mask)));
};

Expr<float> decode_fixed_point(Expr<uint> x, uint mask) noexcept {
    // constexpr auto fixed_point_bits = 16u;
    // constexpr auto fixed_point_mask = (1u << fixed_point_bits) - 1u;
    // constexpr auto fixed_point_scale = 1.0f / static_cast<float>(1u << fixed_point_bits);
    return cast<float>(x) / static_cast<float>(mask);
};

Shape::Shape(Scene *scene, const SceneNodeDesc *desc) noexcept :
    SceneNode{scene, desc, SceneNodeTag::SHAPE},
    _surface{scene->load_surface(desc->property_node_or_default("surface"))},
    _light{scene->load_light(desc->property_node_or_default("light"))},
    _medium{scene->load_medium(desc->property_node_or_default("medium"))},
    _subsurface{scene->load_subsurface(desc->property_node_or_default("subsurface"))},
    _transform{scene->load_transform(desc->property_node_or_default("transform"))} {}

void Shape::update(Scene *scene, const SceneNodeDesc *desc) noexcept {
    set_updated(
        update_value(_transform, (const Transform *)scene->load_transform(desc->property_node_or_default("transform")))
    );
}

luisa::string Shape::info() const noexcept {
    return luisa::format(
        "{} surface=[{}] light=[{}] medium=[{}] subsurface=[{}] transform=[{}]", SceneNode::info(),
        _surface ? _surface->info() : "",
        _light ? _light->info() : "",
        _medium ? _medium->info(): "",
        _subsurface ? _subsurface->info(): "",
        _transform ? _transform->info() : "");
}


bool Shape::visible() const noexcept { return true; }
const Surface *Shape::surface() const noexcept { return _surface; }
const Light *Shape::light() const noexcept { return _light; }
const Medium *Shape::medium() const noexcept { return _medium; }
const Subsurface *Shape::subsurface() const noexcept { return _subsurface; }
const Transform *Shape::transform() const noexcept { return _transform; }
float Shape::shadow_terminator_factor() const noexcept { return 0.f; }
float Shape::intersection_offset_factor() const noexcept { return 0.f; }
float Shape::clamp_normal_factor() const noexcept { return 180.f; }

bool Shape::is_mesh() const noexcept { return false; }
bool Shape::is_spheres() const noexcept { return false; }
bool Shape::empty() const noexcept { return true; }
uint Shape::vertex_properties() const noexcept { return 0u; }
bool Shape::has_vertex_normal() const noexcept {
    return is_mesh() && (vertex_properties() & property_flag_has_vertex_normal) != 0u;
}
bool Shape::has_vertex_uv() const noexcept {
    return is_mesh() && (vertex_properties() & property_flag_has_vertex_uv) != 0u;
}
MeshView Shape::mesh() const noexcept { return {}; }
SpheresView Shape::spheres() const noexcept { return {}; }
luisa::span<const Shape *const> Shape::children() const noexcept { return {}; }
AccelOption Shape::build_option() const noexcept { return {}; }


std::pair<uint3, uint4> Shape::Handle::encode(
    uint buffer_base, uint flags, uint primitive_count,
    uint surface_tag, uint light_tag, uint medium_tag, uint subsurface_tag,
    float shadow_terminator, float intersection_offset, float clamp_normal
) noexcept {
    LUISA_ASSERT(buffer_base <= buffer_base_max, "Invalid geometry buffer base: {}.", buffer_base);
    LUISA_ASSERT(flags <= property_flag_mask, "Invalid property flags: {:016x}.", flags);
    auto buffer_base_and_properties = (buffer_base << property_flag_bits) | flags;
    auto shadow_inter_clamp =
        (encode_fixed_point(shadow_terminator, shadow_term_mask) << shadow_term_offset) |
        (encode_fixed_point(intersection_offset, inter_offset_mask) << inter_offset_offset) |
        (encode_fixed_point(clamp_normal * inv_pi, clamp_normal_mask) << clamp_normal_offset);
    return std::make_pair(
        make_uint3(buffer_base_and_properties, primitive_count, shadow_inter_clamp),
        make_uint4(surface_tag, light_tag, medium_tag, subsurface_tag)
    );
}

Shape::Handle Shape::Handle::decode(Expr<uint3> comp_geom, Expr<uint4> comp_prop) noexcept {
    auto buffer_base_and_properties = comp_geom.x;
    auto triangle_buffer_size = comp_geom.y;
    auto shadow_intersect_clamp = comp_geom.z;

    auto buffer_base = buffer_base_and_properties >> property_flag_bits;
    auto flags = buffer_base_and_properties & property_flag_mask;

    auto shadow_terminator = decode_fixed_point(
        (shadow_intersect_clamp >> shadow_term_offset) & shadow_term_mask, shadow_term_mask);
    auto intersection_offset = decode_fixed_point(
        (shadow_intersect_clamp >> inter_offset_offset) & inter_offset_mask, inter_offset_mask);
    auto clamp_normal = decode_fixed_point(
        (shadow_intersect_clamp >> clamp_normal_offset) & clamp_normal_mask, clamp_normal_mask) * pi;

    // TODO: Why *255?
    return {
        buffer_base, flags, triangle_buffer_size,
        comp_prop.x, comp_prop.y, comp_prop.z, comp_prop.w,
        shadow_terminator, clamp(intersection_offset * 255.f + 1.f, 1.f, 256.f), clamp_normal};
        // shadow_terminator, intersection_offset, clamp_normal};
}

// uint4 Shape::PropertyHandle::encode(
// ) noexcept {
//     return make_uint4(surface_tag, light_tag, medium_tag, subsurface_tag);
// }

// Shape::PropertyHandle Shape::PropertyHandle::decode(Expr<uint4> compressed) noexcept {
//     return {compressed.x, compressed.y, compressed.z, compressed.w};
// }

}// namespace luisa::render
