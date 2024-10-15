//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <util/spec.h>
#include <util/scattering.h>
#include <base/scene_node.h>
#include <base/sampler.h>
#include <base/spectrum.h>
#include <base/interaction.h>
#include <util/polymorphic_closure.h>

#include <utility>

namespace luisa::render {

using compute::BindlessArray;
using compute::Expr;
using compute::Float3;
using compute::Local;
using compute::Var;
using compute::ArrayFloat;
using compute::ArrayVar;

class Shape;
class Frame;
class Interaction;

class Subsurface : public SceneNode {       // Separable

public:
    struct Evaluation {
        SampledSpectrum f;
        Float pdf;
        [[nodiscard]] static auto zero(uint spec_dim) noexcept {
            return Evaluation{
                .f = SampledSpectrum{spec_dim},
                .pdf = 0.f};
        }
    };

    struct Sample {
        Evaluation eval;
        Interaction it;
        Float dist;

        [[nodiscard]] static auto zero(uint spec_dim) noexcept {
            return Sample{
                .eval = Evaluation::zero(spec_dim),
                .it = Interaction(),
                .dist = 0.f};
        }
    };

    class Closure : public PolymorphicClosure {

    public:
        static constexpr auto sample_capacity = 32u;

    private:
        const Pipeline &_pipeline;
        const SampledWavelengths &_swl;
        Float _time;

    private:
        friend class Subsurface;

    public:
        Closure(const Pipeline &pipeline,
                const SampledWavelengths &swl,
                Expr<float> time) noexcept:
            _pipeline{pipeline}, _swl{swl}, _time{time} {}
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] auto &swl() const noexcept { return _swl; }
        [[nodiscard]] auto time() const noexcept { return _time; }
        [[nodiscard]] virtual const Interaction &it() const noexcept = 0;

        [[nodiscard]] Evaluation evaluate(
            const Interaction &it_i, TransportMode mode = TransportMode::RADIANCE
        ) const noexcept;
        [[nodiscard]] Sample sample(
            Expr<float> u_lobe, Expr<float2> u,
            TransportMode mode = TransportMode::RADIANCE
        ) const noexcept;
        [[nodiscard]] virtual SampledSpectrum sr(Expr<float> r) const noexcept = 0;
        [[nodiscard]] virtual Float pdf_sr(Expr<float> r) const noexcept = 0;
        [[nodiscard]] virtual Float sample_r(Expr<float> u) const noexcept = 0;
    };

    class Instance : public SceneNode::Instance {

    private:
        const Subsurface *_subsurface;

    private:
        friend class Subsurface;

    public:
        Instance(Pipeline &pipeline, const Subsurface *subsurface) noexcept:
            SceneNode::Instance{pipeline}, _subsurface{subsurface} {}
        Instance(const Instance &) noexcept = delete;
        virtual ~Instance() noexcept = default;
        template<typename T = Subsurface>
            requires std::is_base_of_v<Subsurface, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_subsurface); }

        [[nodiscard]] virtual luisa::string closure_identifier() const noexcept;
        void closure(PolymorphicCall<Closure> &call, const Interaction &it,
                     const SampledWavelengths &swl, Expr<float> time) const noexcept;
        virtual void populate_closure(
            Closure *closure, const Interaction &it) const noexcept = 0;
        [[nodiscard]] virtual luisa::unique_ptr<Closure> create_closure(
            const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
    };

protected:
    [[nodiscard]] virtual luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;

public:
    Subsurface(Scene *scene, const SceneNodeDesc *desc) noexcept;
    virtual void update(Scene *scene, const SceneNodeDesc *desc) noexcept override;
    [[nodiscard]] virtual bool is_null() const noexcept { return false; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept;
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Subsurface::Instance)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Subsurface::Closure)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Subsurface::Sample)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Subsurface::Evaluation)
