//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <span>
#include <mutex>
#include <functional>

#include <luisa/core/stl.h>
#include <luisa/core/dynamic_module.h>
#include <luisa/core/basic_types.h>
#include <luisa/runtime/context.h>
#include <base/scene_node.h>

namespace luisa::render {

using compute::Context;

class SceneDesc;
class SceneNodeDesc;

class Camera;
class Film;
class Filter;
class Integrator;
class Surface;
class Light;
class Sampler;
class Shape;
class Transform;
class LightSampler;
class Environment;
class Texture;
class TextureMapping;
class Spectrum;
class Medium;
class PhaseFunction;

class Scene {

public:
    using NodeCreaterDesc = SceneNode *(Scene *, const SceneNodeDesc *);
    using NodeDeleter = void(SceneNode *);
    using NodeHandle = luisa::unique_ptr<SceneNode, NodeDeleter *>;

    struct Config;

private:
    const Context &_context;
    luisa::unique_ptr<Config> _config;
    std::recursive_mutex _mutex;

public:
    // for internal use only, call Scene::create() instead
    explicit Scene(const Context &ctx) noexcept;
    ~Scene() noexcept;
    Scene(Scene &&scene) noexcept = delete;
    Scene(const Scene &scene) noexcept = delete;
    Scene &operator=(Scene &&scene) noexcept = delete;
    Scene &operator=(const Scene &scene) noexcept = delete;
    
    template <typename NodeCreater>
    [[nodiscard]] auto get_handle_creater(
        SceneNodeTag tag, luisa::string_view impl_type, luisa::string_view creater_name) noexcept;
    template <typename... Args, typename Callable>
    [[nodiscard]] std::pair<SceneNode*, bool> load_from_nodes(
        luisa::string_view name, Callable &&handle_creater, Args&&... args) noexcept;
    template <typename Callable>
    [[nodiscard]] SceneNode *load_node(
        SceneNodeTag tag, const SceneNodeDesc *desc, Callable &&updated_callback) noexcept;
    [[nodiscard]] Camera *load_camera(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Film *load_film(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Filter *load_filter(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Integrator *load_integrator(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Surface *load_surface(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Light *load_light(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Sampler *load_sampler(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Shape *load_shape(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Transform *load_transform(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] LightSampler *load_light_sampler(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Environment *load_environment(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Texture *load_texture(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] TextureMapping *load_texture_mapping(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Spectrum *load_spectrum(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Medium *load_medium(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] PhaseFunction *load_phase_function(const SceneNodeDesc *desc) noexcept;
    
    [[nodiscard]] Environment *update_environment(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] std::pair<Camera *, uint> update_camera(const SceneNodeDesc *desc, bool first_def) noexcept;
    [[nodiscard]] Shape *update_shape(const SceneNodeDesc *desc, bool first_def) noexcept;

public:
    [[nodiscard]] static luisa::unique_ptr<Scene> create(const Context &ctx, const SceneDesc *desc) noexcept;
    [[nodiscard]] const Integrator *integrator() const noexcept;
    [[nodiscard]] const Environment *environment() const noexcept;
    [[nodiscard]] const Medium *environment_medium() const noexcept;
    [[nodiscard]] const Spectrum *spectrum() const noexcept;
    [[nodiscard]] luisa::span<const Shape *const> shapes() const noexcept;
    [[nodiscard]] luisa::span<const Camera *const> cameras() const noexcept;
    [[nodiscard]] float shadow_terminator_factor() const noexcept;
    [[nodiscard]] float intersection_offset_factor() const noexcept;
    [[nodiscard]] float clamp_normal_factor() const noexcept;

    [[nodiscard]] luisa::string info() const noexcept;
    [[nodiscard]] bool shapes_updated() const noexcept;
    [[nodiscard]] bool cameras_updated() const noexcept;
    [[nodiscard]] bool transforms_updated() const noexcept;
    [[nodiscard]] bool environment_updated() const noexcept;
    void clear_update() noexcept;
};

}// namespace luisa::render
