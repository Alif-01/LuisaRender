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

struct RawMeshInfo;
struct CameraStorage;
struct RawCameraInfo;
struct RawSurfaceInfo;

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
        luisa::string_view name, SceneNodeTag tag,
        luisa::string_view impl_type, luisa::string_view creater_name) noexcept;
    template <typename... Args, typename Callable>
    [[nodiscard]] std::pair<SceneNode*, bool> load_from_nodes(
        luisa::string_view name, Callable &&handle_creater, Args&&... args
    ) noexcept;
    [[nodiscard]] SceneNode *load_node(SceneNodeTag tag, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] SceneNode *load_node_from_name(luisa::string name) noexcept;
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
    
    [[nodiscard]] Film *add_film(luisa::string_view name, const uint2 &resolution) noexcept;
    [[nodiscard]] Filter *add_filter(luisa::string_view name, const float &radius) noexcept;
    [[nodiscard]] Transform *update_transform(luisa::string_view name, const RawTransform &trans) noexcept;
    [[nodiscard]] Texture *add_constant_texture(luisa::string_view name, const luisa::vector<float> &v) noexcept;
    [[nodiscard]] Texture *add_image_texture(
        luisa::string_view name, luisa::string_view image, const float &image_scale) noexcept;
    [[nodiscard]] Shape *update_shape(const RawMeshInfo &mesh_info) noexcept;
    [[nodiscard]] Camera *add_camera(
        const RawCameraInfo &camera_info, luisa::unordered_map<luisa::string, CameraStorage> &camera_storage, Device &device) noexcept;
    [[nodiscard]] Surface *add_surface(const RawSurfaceInfo &surface_info) noexcept;
    void append_shape(Shape *shape) noexcept;

public:
    [[nodiscard]] static luisa::unique_ptr<Scene> create(
        const Context &ctx, const SceneDesc *desc, Device &device,
        luisa::unordered_map<luisa::string, CameraStorage> &camera_storage) noexcept;
    [[nodiscard]] const Integrator *integrator() const noexcept;
    [[nodiscard]] const Environment *environment() const noexcept;
    [[nodiscard]] const Medium *environment_medium() const noexcept;
    [[nodiscard]] const Spectrum *spectrum() const noexcept;
    [[nodiscard]] luisa::span<const Shape *const> shapes() const noexcept;
    [[nodiscard]] luisa::span<const Camera *const> cameras() const noexcept;
    [[nodiscard]] bool shapes_updated() const noexcept;
    [[nodiscard]] bool cameras_updated() const noexcept;
    void clear_shapes_update() noexcept;
    void clear_cameras_update() noexcept;
    [[nodiscard]] float shadow_terminator_factor() const noexcept;
    [[nodiscard]] float intersection_offset_factor() const noexcept;
};

}// namespace luisa::render
