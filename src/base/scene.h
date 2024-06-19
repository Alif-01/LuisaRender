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

struct RawTextureInfo;
struct RawLightInfo;
struct RawSamplerInfo;
struct RawSpectrumInfo;
struct RawIntegratorInfo;
struct RawEnvironmentInfo;
struct RawSurfaceInfo;
struct RawTransformInfo;
struct RawFilmInfo;
struct RawFilterInfo;
struct RawCameraInfo;
struct RawShapeInfo;
struct RawSceneInfo;

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
        luisa::string_view name, Callable &&handle_creater, Args&&... args
    ) noexcept;
    [[nodiscard]] SceneNode *load_node(SceneNodeTag tag, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] SceneNode *load_node_from_name(luisa::string_view name) noexcept;
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
    
    [[nodiscard]] Film *update_film(luisa::string_view name, const RawFilmInfo &film_info) noexcept;
    [[nodiscard]] Sampler *add_sampler(const RawSamplerInfo &sampler_info) noexcept;
    [[nodiscard]] Filter *add_filter(luisa::string_view name, const RawFilterInfo &filter_info) noexcept;
    [[nodiscard]] Spectrum *add_spectrum(const RawSpectrumInfo &spectrum_info) noexcept;
    [[nodiscard]] Integrator *add_integrator(const RawIntegratorInfo &integrator_info) noexcept;
    [[nodiscard]] Environment *add_environment(const RawEnvironmentInfo &environment_info) noexcept;
    [[nodiscard]] Light *add_light(const RawLightInfo &light_info) noexcept;
    [[nodiscard]] Texture *add_texture(luisa::string_view name, const RawTextureInfo &texture_info) noexcept;
    [[nodiscard]] Surface *add_surface(const RawSurfaceInfo &surface_info) noexcept;

    [[nodiscard]] Transform *update_transform(luisa::string_view name, const RawTransformInfo &transform_info) noexcept;
    [[nodiscard]] Camera *update_camera(const RawCameraInfo &camera_info) noexcept;
    [[nodiscard]] Shape *update_shape(const RawShapeInfo &shape_info) noexcept;

public:
    [[nodiscard]] static luisa::unique_ptr<Scene> create(const Context &ctx, const SceneDesc *desc) noexcept;
    [[nodiscard]] static luisa::unique_ptr<Scene> create(const Context &ctx, const RawSceneInfo &scene_info) noexcept;
    [[nodiscard]] const Integrator *integrator() const noexcept;
    [[nodiscard]] const Environment *environment() const noexcept;
    [[nodiscard]] const Medium *environment_medium() const noexcept;
    [[nodiscard]] const Spectrum *spectrum() const noexcept;
    [[nodiscard]] luisa::span<const Shape *const> shapes() const noexcept;
    [[nodiscard]] luisa::span<const Camera *const> cameras() const noexcept;
    [[nodiscard]] float shadow_terminator_factor() const noexcept;
    [[nodiscard]] float intersection_offset_factor() const noexcept;
    [[nodiscard]] float clamp_normal_factor() const noexcept;

    [[nodiscard]] bool shapes_updated() const noexcept;
    [[nodiscard]] bool cameras_updated() const noexcept;
    [[nodiscard]] bool film_updated() const noexcept;
    [[nodiscard]] bool transforms_updated() const noexcept;
    [[nodiscard]] bool environment_updated() const noexcept;
    void clear_update() noexcept;
};

}// namespace luisa::render
