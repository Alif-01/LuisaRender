//
// Created by Mike Smith on 2022/1/10.
//

#include <util/sampling.h>
#include <base/light_sampler.h>
#include <base/pipeline.h>

namespace luisa::render {

class UniformLightSampler final : public LightSampler {

private:
    float _environment_weight{.5f};

public:
    UniformLightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept:
        LightSampler{scene, desc},
        _environment_weight{desc->property_float_or_default("environment_weight", 0.5f)} {}
    [[nodiscard]] float environment_weight() const noexcept override { return _environment_weight; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class UniformLightSamplerInstance final : public LightSampler::Instance {

public:
    UniformLightSamplerInstance(Pipeline &pipeline, CommandBuffer &command_buffer,
                                const UniformLightSampler *sampler) noexcept:
        LightSampler::Instance{pipeline, sampler} {}

    [[nodiscard]] Light::Evaluation evaluate_hit(
        const Interaction &it, Expr<float3> p_from,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto eval = Light::Evaluation::zero(swl.dimension());
        if (pipeline().lights().empty()) [[unlikely]] {// no lights
            LUISA_WARNING_WITH_LOCATION("No lights in scene.");
            return eval;
        }
        pipeline().lights().dispatch(it.shape().light_tag(), [&](auto light) noexcept {
            auto closure = light->closure(swl, time);
            eval = closure->evaluate(it, p_from);
        });
        auto n = static_cast<float>(pipeline().lights().size());
        if (pipeline().environment() == nullptr) {
            eval.pdf *= 1.f / n;
        } else {
            eval.pdf *= (1.f - node()->environment_weight()) / n;
        }
        return eval;
    }

    [[nodiscard]] Light::Evaluation evaluate_miss(
        Expr<float3> wi, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        if (pipeline().environment() == nullptr) [[unlikely]] {     // no environment
            LUISA_WARNING_WITH_LOCATION("No environment in scene");
            return {.L = SampledSpectrum{swl.dimension()}, .pdf = 0.f};
        }
        auto eval = pipeline().environment()->evaluate(wi, swl, time);
        eval.pdf *= node()->environment_weight();
        return eval;
    }

    [[nodiscard]] LightSampler::Selection select(
        const Interaction &it_from, Expr<float> u,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        LUISA_ASSERT(pipeline().has_lighting(), "No lights in scene.");
        auto n = static_cast<float>(pipeline().lights().size());
        if (pipeline().environment() == nullptr) {
            return {.tag = cast<uint>(clamp(u * n, 0.f, n - 1.f)), .prob = 1.f / n};
        }
        if (pipeline().lights().empty()) {
            return {.tag = LightSampler::selection_environment, .prob = 1.f};
        }
        auto env_prob = node()->environment_weight();
        auto uu = (u - env_prob) / (1.f - env_prob);
        auto tag = cast<uint>(clamp(uu * n, 0.f, n - 1.f));
        auto is_env = u < env_prob;
        return {.tag = ite(is_env, LightSampler::selection_environment, tag),
                .prob = ite(is_env, env_prob, (1.f - env_prob) / n)};
    }
    [[nodiscard]] LightSampler::Selection select(
        Expr<float> u,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        LUISA_ASSERT(pipeline().has_lighting(), "No lights in scene.");

        auto n = static_cast<float>(pipeline().lights().size());
        if (pipeline().environment() == nullptr) {
            return {.tag = cast<uint>(clamp(u * n, 0.f, n - 1.f)), .prob = 1.f / n};
        }
        if (pipeline().lights().empty()) {
            return {.tag = LightSampler::selection_environment, .prob = 1.f};
        }
        auto env_prob = node()->environment_weight();
        auto uu = (u - env_prob) / (1.f - env_prob);
        auto tag = cast<uint>(clamp(uu * n, 0.f, n - 1.f));
        auto is_env = u < env_prob;
        return {.tag = ite(is_env, LightSampler::selection_environment, tag),
                .prob = ite(is_env, env_prob, (1.f - env_prob) / n)};
    }


private:
    [[nodiscard]] auto _sample_area(Expr<float3> p_from,
                                    Expr<uint> tag,
                                    Expr<float2> u_in) const noexcept {
        auto light_inst_id = pipeline().geometry()->light_instance(tag);
        auto light_inst = pipeline().geometry()->instance(light_inst_id);
        auto light_to_world = pipeline().geometry()->instance_to_world(light_inst_id);
        auto alias_table_buffer_id = light_inst.alias_table_buffer_id();
        auto [prim_id, ux] = sample_alias_table(
            pipeline().buffer<AliasEntry>(alias_table_buffer_id),
            light_inst.primitive_count(), u_in.x);

        Interaction it;
        $if (light_inst.is_triangle()) {
            auto triangle = pipeline().geometry()->triangle(light_inst, prim_id);
            auto uvw = sample_uniform_triangle(make_float2(ux, u_in.y));
            auto attrib = pipeline().geometry()->shading_point(light_inst, triangle, uvw, light_to_world);
            // auto attrib = pipeline().geometry()->geometry_point(light_inst, triangle, uvw, light_to_world);
        
            it = Interaction(
                std::move(light_inst), light_inst_id, prim_id,
                std::move(attrib), dot(attrib.g.n, p_from - attrib.g.p) < 0.f);
        }
        $else {
            auto aabb = pipeline().geometry()->aabb(light_inst, prim_id);
            auto w = sample_uniform_sphere(make_float2(ux, u_in.y));
            auto attrib = pipeline().geometry()->geometry_point(light_inst, aabb, w, light_to_world);
            it = Interaction(
                std::move(light_inst), light_inst_id, prim_id,
                attrib.area, attrib.p, attrib.n, dot(attrib.n, p_from - attrib.p) < 0.f);
        };
        return luisa::make_shared<Interaction>(std::move(it));
    }

    [[nodiscard]] Light::Sample _sample_light(const Interaction &it_from,
                                              Expr<uint> tag, Expr<float2> u,
                                              const SampledWavelengths &swl,
                                              Expr<float> time) const noexcept override {
        LUISA_ASSERT(!pipeline().lights().empty(), "No lights in the scene.");
        auto it = _sample_area(it_from.p(), tag, u);
        auto eval = Light::Evaluation::zero(swl.dimension());
        pipeline().lights().dispatch(it->shape().light_tag(), [&](auto light) noexcept {
            auto closure = light->closure(swl, time);
            eval = closure->evaluate(*it, it_from.p_shading());
        });
        return {.eval = std::move(eval), .p = it->p()};
    }

    [[nodiscard]] Environment::Sample _sample_environment(Expr<float2> u,
                                                          const SampledWavelengths &swl,
                                                          Expr<float> time) const noexcept override {
        LUISA_ASSERT(pipeline().environment() != nullptr, "No environment in the scene.");
        return pipeline().environment()->sample(swl, time, u);
    }
    //sample single light for L_emit.
    [[nodiscard]] LightSampler::Sample _sample_light_le(
            Expr<uint> tag, Expr<float2> u_light, Expr<float2> u_direction,
            const SampledWavelengths &swl,
            Expr<float> time) const noexcept override {
        LUISA_ASSERT(!pipeline().lights().empty(), "No lights in the scene.");
        auto light_inst_id = pipeline().geometry()->light_instance(tag);
        auto light_inst = pipeline().geometry()->instance(light_inst_id);
        auto sp = Light::Sample::zero(swl.dimension());
        Var<Ray> shadow_ray{};
        pipeline().lights().dispatch(light_inst.light_tag(), [&](auto light) noexcept {
            auto closure = light->closure(swl, time);
            auto [sp_tp, ray_tp] = closure->sample_le(light_inst_id, u_light, u_direction);
            sp = sp_tp;
            shadow_ray = ray_tp;
        });
        return {.eval = sp.eval, .shadow_ray = shadow_ray};
    }
};

unique_ptr<LightSampler::Instance> UniformLightSampler::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<UniformLightSamplerInstance>(pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::UniformLightSampler)
