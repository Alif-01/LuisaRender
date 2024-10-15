//
// Created by Mike Smith on 2022/1/12.
//

#include <base/subsurface.h>
#include <base/scene.h>
#include <base/pipeline.h>

namespace luisa::render {

struct UniformSubsurface : public Subsurface {

private:
    const Texture *_thickness;

public:
    UniformSubsurface(Scene *scene, const SceneNodeDesc *desc) noexcept:
        Subsurface{scene, desc},
        _thickness{scene->load_texture(desc->property_node_or_default("thickness"))} {}

    [[nodiscard]] luisa::string info() const noexcept override {
        return luisa::format(
            "{} thickness=[{}]", Subsurface::info(),
            _thickness ? _thickness->info() : ""
        );
    }

    [[nodiscard]] bool is_null() const noexcept override { return !_thickness || _thickness->is_black(); }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] auto thickness() const noexcept { return _thickness; }

private:
    [[nodiscard]] luisa::unique_ptr<Subsurface::Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class UniformInstance : public Subsurface::Instance {

private:
    const Texture::Instance *_thickness;

public:
    UniformInstance(Pipeline &pipeline, CommandBuffer &command_buffer,
                    const UniformSubsurface *subsurface) noexcept:
        Subsurface::Instance{pipeline, subsurface},
        _thickness{pipeline.build_texture(command_buffer, subsurface->thickness())} {}

public:
    [[nodiscard]] luisa::unique_ptr<Subsurface::Closure> create_closure(
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
    void populate_closure(Subsurface::Closure *closure, const Interaction &it) const noexcept override;
};

class UniformClosure : public Subsurface::Closure {

public:
    struct Context {
        Interaction it;
        Float thickness;
    };

public:
    using Subsurface::Closure::Closure;

public:
    [[nodiscard]] const Interaction &it() const noexcept override { return context<Context>().it; }
    [[nodiscard]] SampledSpectrum sr(Expr<float> r) const noexcept override;
    [[nodiscard]] Float pdf_sr(Expr<float> r) const noexcept override;
    [[nodiscard]] Float sample_r(Expr<float> u) const noexcept override;
};

luisa::unique_ptr<Subsurface::Instance> UniformSubsurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer
) const noexcept {
    return luisa::make_unique<UniformInstance>(pipeline, command_buffer, this);
}

luisa::unique_ptr<Subsurface::Closure> UniformInstance::create_closure(
    const SampledWavelengths &swl, Expr<float> time
) const noexcept {
    return luisa::make_unique<UniformClosure>(pipeline(), swl, time);
}
    
void UniformInstance::populate_closure(
    Subsurface::Closure *closure, const Interaction &it
) const noexcept {
    auto time = closure->time();
    auto thickness = _thickness ? _thickness->evaluate(it, time).x : 0.f;
    UniformClosure::Context context{
        .it = it,
        .thickness = thickness
    };
    closure->bind(std::move(context));
}

SampledSpectrum UniformClosure::sr(Expr<float> r) const noexcept {
    // sigma_t = 1
    return SampledSpectrum(swl().dimension(), 1.f / (2.f * pi * r));
}

Float UniformClosure::sample_r(Expr<float> u) const noexcept {
    auto &&ctx = context<Context>();
    return ctx.thickness * u;
}

Float UniformClosure::pdf_sr(Expr<float> r) const noexcept {
    auto &&ctx = context<Context>();
    return 1.f / ctx.thickness;
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::UniformSubsurface)