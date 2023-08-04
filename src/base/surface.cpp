//
// Created by Mike on 2021/12/14.
//

#include <base/surface.h>
#include <base/scene.h>
#include <base/interaction.h>
#include <base/pipeline.h>

namespace luisa::render {

const luisa::string RawSurfaceInfo::mat_string[5] = {
    "null", "metal", "substrate", "matte", "glass"
};

Surface::Surface(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SURFACE} {}

Surface::Surface(Scene *scene) noexcept
    : SceneNode{scene, SceneNodeTag::SURFACE} {}

luisa::unique_ptr<Surface::Instance> Surface::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    LUISA_ASSERT(!is_null(), "Building null Surface.");
    LUISA_ASSERT(!(is_transmissive() && is_thin()), "Surface cannot be both transmissive and thin.");
    return _build(pipeline, command_buffer);
}

void Surface::Instance::closure(PolymorphicCall<Closure> &call,
                                const Interaction &it, const SampledWavelengths &swl,
                                Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept {
    
    LUISA_INFO("DEBUG_10.47");

    auto cls = call.collect(closure_identifier(), [&] {
        return create_closure(swl, time);
    });

    LUISA_INFO("DEBUG_10.48");

    populate_closure(cls, it, wo, eta_i);
}

luisa::string Surface::Instance::closure_identifier() const noexcept {
    return luisa::string{node()->impl_type()};
}

static auto validate_surface_sides(Expr<float3> ng, Expr<float3> ns,
                                   Expr<float3> wo, Expr<float3> wi) noexcept {
    static Callable is_valid = [](Float3 ng, Float3 ns, Float3 wo, Float3 wi) noexcept {
        auto gs = dot(wo, ng) * dot(wi, ng) > 0.f;
        auto ss = dot(wo, ns) * dot(wi, ns) > 0.f;
        return gs == ss;
    };
    return is_valid(ng, ns, wo, wi);
}

Surface::Evaluation Surface::Closure::evaluate(
    Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
    auto eval = _evaluate(wo, wi, mode);
    auto valid = validate_surface_sides(it().ng(), it().shading().n(), wo, wi);
    eval.f = ite(valid, eval.f, 0.f);
    eval.pdf = ite(valid, eval.pdf, 0.f);
    return eval;
}

Surface::Sample Surface::Closure::sample(Expr<float3> wo,
                                         Expr<float> u_lobe, Expr<float2> u,
                                         TransportMode mode) const noexcept {
    auto s = _sample(wo, u_lobe, u, mode);
    auto valid = validate_surface_sides(it().ng(), it().shading().n(), wo, s.wi);
    s.eval.f = ite(valid, s.eval.f, 0.f);
    s.eval.pdf = ite(valid, s.eval.pdf, 0.f);
    return s;
}

}// namespace luisa::render
