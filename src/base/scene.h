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
class Subsurface;

class Scene {

public:
    using NodeCreater = SceneNode *(Scene *, const SceneNodeDesc *);
    using NodeDeleter = void(SceneNode *);
    using NodeHandle = luisa::unique_ptr<SceneNode, NodeDeleter *>;

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

#define LUISA_SCENE_NODE_LOAD_DECLARE(name, type)                                 \
    [[nodiscard]] type *load_##name(const SceneNodeDesc *desc) noexcept;

LUISA_SCENE_NODE_LOAD_DECLARE(camera, Camera)
LUISA_SCENE_NODE_LOAD_DECLARE(film, Film)
LUISA_SCENE_NODE_LOAD_DECLARE(filter, Filter)
LUISA_SCENE_NODE_LOAD_DECLARE(integrator, Integrator)
LUISA_SCENE_NODE_LOAD_DECLARE(surface, Surface)
LUISA_SCENE_NODE_LOAD_DECLARE(light, Light)
LUISA_SCENE_NODE_LOAD_DECLARE(sampler, Sampler)
LUISA_SCENE_NODE_LOAD_DECLARE(shape, Shape)
LUISA_SCENE_NODE_LOAD_DECLARE(transform, Transform)
LUISA_SCENE_NODE_LOAD_DECLARE(light_sampler, LightSampler)
LUISA_SCENE_NODE_LOAD_DECLARE(environment, Environment)
LUISA_SCENE_NODE_LOAD_DECLARE(texture, Texture)
LUISA_SCENE_NODE_LOAD_DECLARE(texture_mapping, TextureMapping)
LUISA_SCENE_NODE_LOAD_DECLARE(spectrum, Spectrum)
LUISA_SCENE_NODE_LOAD_DECLARE(medium, Medium)
LUISA_SCENE_NODE_LOAD_DECLARE(phase_function, PhaseFunction)
LUISA_SCENE_NODE_LOAD_DECLARE(subsurface, Subsurface)

public:
    [[nodiscard]] Environment *update_environment(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Camera *update_camera(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] Shape *update_shape(const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] static luisa::unique_ptr<Scene> create(const Context &ctx, const SceneDesc *desc) noexcept;

public:
    [[nodiscard]] luisa::string info() const noexcept;
    [[nodiscard]] Integrator *integrator() const noexcept { return _config->integrator; }
    [[nodiscard]] Environment *environment() const noexcept { return _config->environment; }
    [[nodiscard]] Medium *environment_medium() const noexcept { return _config->environment_medium; }
    [[nodiscard]] Spectrum *spectrum() const noexcept { return _config->spectrum; }
    [[nodiscard]] auto &shapes() const noexcept { return _config->shapes; }
    [[nodiscard]] auto &cameras() const noexcept { return _config->cameras; }
    [[nodiscard]] float shadow_terminator_factor() const noexcept { return _config->shadow_terminator; }
    [[nodiscard]] float intersection_offset_factor() const noexcept { return _config->intersection_offset; }
    [[nodiscard]] float clamp_normal_factor() const noexcept { return _config->clamp_normal; }
};

}// namespace luisa::render
