//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <util/command_buffer.h>
#include <base/scene_node.h>
#include <base/sampler.h>
#include <base/spectrum.h>
#include <base/light_sampler.h>
#include <base/camera.h>

namespace luisa::render {

class Pipeline;
class Display;

class Integrator : public SceneNode {

public:
    class Instance : public SceneNode::Instance {

    private:
        const Integrator *_integrator;
        luisa::unique_ptr<Sampler::Instance> _sampler;
        luisa::unique_ptr<LightSampler::Instance> _light_sampler;

    public:
        explicit Instance(Pipeline &pipeline, CommandBuffer &command_buffer, const Integrator *integrator) noexcept;
        virtual ~Instance() noexcept = default;

        template<typename T = Integrator>
            requires std::is_base_of_v<Integrator, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_integrator); }
        [[nodiscard]] auto sampler() noexcept { return _sampler.get(); }
        [[nodiscard]] auto sampler() const noexcept { return _sampler.get(); }
        [[nodiscard]] auto light_sampler() noexcept { return _light_sampler.get(); }
        [[nodiscard]] auto light_sampler() const noexcept { return _light_sampler.get(); }
        [[nodiscard]] bool use_progress() const noexcept { return _integrator->use_progress(); }
        virtual void render(Stream &stream) noexcept = 0;
        virtual void render_to_buffer(Stream &stream, Camera *camera, luisa::vector<float4> &buffer) noexcept = 0;
    };

private:
    const Sampler *_sampler;
    const LightSampler *_light_sampler;
    bool _use_progress;

public:
    Integrator(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::string info() const noexcept override;
    [[nodiscard]] auto sampler() const noexcept { return _sampler; }
    [[nodiscard]] auto light_sampler() const noexcept { return _light_sampler; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
    [[nodiscard]] bool use_progress() const noexcept { return _use_progress; }
};

class ProgressiveIntegrator : public Integrator {

public:
    class Instance : public Integrator::Instance {

    protected:
        [[nodiscard]] virtual Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index,
                                        Expr<uint2> pixel_id, Expr<float> time) const noexcept;
        virtual void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept;

    public:
        Instance(Pipeline &pipeline,
                 CommandBuffer &command_buffer,
                 const ProgressiveIntegrator *node) noexcept;
        ~Instance() noexcept override;
        void render(Stream &stream) noexcept override;
        void render_to_buffer(Stream &stream, Camera *camera, luisa::vector<float4> &buffer) noexcept override;
    };

public:
    ProgressiveIntegrator(Scene *scene, const SceneNodeDesc *desc) noexcept;
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Integrator::Instance)
