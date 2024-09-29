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
    using NodeHandle = luisa::unique_ptr<SceneNode, NodeDeleter *>;
    using NodeCreater = SceneNode *(Scene *, const SceneNodeDesc *);
    using NodeDeleter = void(SceneNode *);

    struct Config {
        float shadow_terminator{0.f};
        float intersection_offset{0.f};
        float clamp_normal{180.f};
        luisa::vector<NodeHandle> internal_nodes;
        luisa::unordered_map<luisa::string, NodeHandle> nodes;
        Integrator *integrator{nullptr};
        Environment *environment{nullptr};
        Medium *environment_medium{nullptr};
        Spectrum *spectrum{nullptr};
        luisa::unordered_set<Camera *> cameras;
        luisa::unordered_set<Shape *> shapes;

        // bool environment_updated{false};
        // bool cameras_updated{false};
        // bool shapes_updated{false};
        // bool transforms_updated{false};
    };

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
    
    [[nodiscard]] NodeHandle get_node_handle(SceneNodeTag tag, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] SceneNode *load_node(SceneNodeTag tag, const SceneNodeDesc *desc) noexcept;

#define LUISA_SCENE_NODE_LOAD(name, type, tag)                                      \
    [[nodiscard]] type *load_##name(const SceneNodeDesc *desc) noexcept {   \
        return dynamic_cast<type *>(SceneNodeTag::tag, desc);               \
    }

    LUISA_SCENE_NODE_LOAD(camera, Camera, CAMERA)
    LUISA_SCENE_NODE_LOAD(film, Film, FILM)
    LUISA_SCENE_NODE_LOAD(filter, Filter, FILTER)
    LUISA_SCENE_NODE_LOAD(integrator, Integrator, INTEGRATOR)
    LUISA_SCENE_NODE_LOAD(surface, Surface, SURFACE)
    LUISA_SCENE_NODE_LOAD(light, Light, LIGHT)
    LUISA_SCENE_NODE_LOAD(sampler, Sampler, SAMPLER)
    LUISA_SCENE_NODE_LOAD(shape, Shape, SHAPE)
    LUISA_SCENE_NODE_LOAD(transform, Transform, TRANSFORM)
    LUISA_SCENE_NODE_LOAD(light_sampler, LightSampler, LIGHT_SAMPLER)
    LUISA_SCENE_NODE_LOAD(environment, Environment, ENVIRONMENT)
    LUISA_SCENE_NODE_LOAD(texture, Texture, TEXTURE)
    LUISA_SCENE_NODE_LOAD(texture_mapping, TextureMapping, TEXTURE_MAPPING)
    LUISA_SCENE_NODE_LOAD(spectrum, Spectrum, SPECTRUM)
    LUISA_SCENE_NODE_LOAD(medium, Medium, MEDIUM)
    LUISA_SCENE_NODE_LOAD(phase_function, PhaseFunction, PHASE_FUNCTION)

    // template <typename NodeCreater>
    // [[nodiscard]] auto get_handle_creater(
    //     SceneNodeTag tag, luisa::string_view impl_type, luisa::string_view creater_name) noexcept;
    // template <typename... Args, typename Callable>
    // [[nodiscard]] std::pair<SceneNode*, bool> load_from_nodes(
    //     luisa::string_view name, Callable &&handle_creater, Args&&... args) noexcept;
    // template <typename Callable>
    // [[nodiscard]] Camera *load_camera(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] Film *load_film(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] Filter *load_filter(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] Integrator *load_integrator(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] Surface *load_surface(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] Light *load_light(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] Sampler *load_sampler(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] Shape *load_shape(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] Transform *load_transform(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] LightSampler *load_light_sampler(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] Environment *load_environment(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] Texture *load_texture(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] TextureMapping *load_texture_mapping(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] Spectrum *load_spectrum(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] Medium *load_medium(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] PhaseFunction *load_phase_function(const SceneNodeDesc *desc) noexcept;
    
    [[nodiscard]] Environment *update_environment(const SceneNodeDesc *desc) noexcept;
    // [[nodiscard]] Camera *update_camera(const SceneNodeDesc *desc, bool first_def) noexcept;
    // [[nodiscard]] Shape *update_shape(const SceneNodeDesc *desc, bool first_def) noexcept;
    [[nodiscard]] Camera *update_camera(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Shape *update_shape(const SceneNodeDesc *desc) noexcept;

public:
    // [[nodiscard]] static luisa::unique_ptr<Scene> create(const Context &ctx, const SceneDesc *desc) noexcept;
    // [[nodiscard]] const Integrator *integrator() const noexcept;
    // [[nodiscard]] const Environment *environment() const noexcept;
    // [[nodiscard]] const Medium *environment_medium() const noexcept;
    // [[nodiscard]] const Spectrum *spectrum() const noexcept;
    // [[nodiscard]] luisa::span<const Shape *const> shapes() const noexcept;
    // [[nodiscard]] luisa::span<const Camera *const> cameras() const noexcept;
    // [[nodiscard]] float shadow_terminator_factor() const noexcept;
    // [[nodiscard]] float intersection_offset_factor() const noexcept;
    // [[nodiscard]] float clamp_normal_factor() const noexcept;
    
    [[nodiscard]] Integrator *integrator() const noexcept { return _config->integrator; }
    [[nodiscard]] Environment *environment() const noexcept { return _config->environment; }
    [[nodiscard]] Medium *environment_medium() const noexcept { return _config->environment_medium; }
    [[nodiscard]] Spectrum *spectrum() const noexcept { return _config->spectrum; }
    [[nodiscard]] luisa::unordered_set<Shape *> &shapes() const noexcept { return _config->shapes; }
    [[nodiscard]] luisa::unordered_set<Camera *> &cameras() const noexcept { return _config->cameras; }
    [[nodiscard]] float shadow_terminator_factor() const noexcept { return _config->shadow_terminator; }
    [[nodiscard]] float intersection_offset_factor() const noexcept { return _config->intersection_offset; }
    [[nodiscard]] float clamp_normal_factor() const noexcept { return _config->clamp_normal; }


    [[nodiscard]] luisa::string info() const noexcept;
    // [[nodiscard]] bool shapes_updated() const noexcept;
    // [[nodiscard]] bool cameras_updated() const noexcept;
    // [[nodiscard]] bool transforms_updated() const noexcept;
    // [[nodiscard]] bool environment_updated() const noexcept;
    // void clear_update() noexcept;
};

}// namespace luisa::render
