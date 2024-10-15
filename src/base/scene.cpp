//
// Created by Mike on 2021/12/8.
//

#include <mutex>

#include <util/thread_pool.h>
#include <sdl/scene_desc.h>
#include <sdl/scene_node_desc.h>
#include <base/camera.h>
#include <base/film.h>
#include <base/filter.h>
#include <base/integrator.h>
#include <base/surface.h>
#include <base/light.h>
#include <base/sampler.h>
#include <base/shape.h>
#include <base/transform.h>
#include <base/environment.h>
#include <base/light_sampler.h>
#include <base/texture.h>
#include <base/texture_mapping.h>
#include <base/spectrum.h>
#include <base/scene.h>
#include <base/medium.h>
#include <base/phase_function.h>
#include <base/subsurface.h>

namespace luisa::render {

namespace detail {

[[nodiscard]] static auto &scene_plugin_registry() noexcept {
    static luisa::unordered_map<luisa::string, luisa::unique_ptr<DynamicModule>> registry;
    return registry;
}

[[nodiscard]] static auto &scene_plugin_registry_mutex() noexcept {
    static std::mutex mutex;
    return mutex;
}

[[nodiscard]] static auto &scene_plugin_load(
    const std::filesystem::path &runtime_dir, SceneNodeTag tag, luisa::string_view impl_type) noexcept {
    std::scoped_lock lock{detail::scene_plugin_registry_mutex()};
    auto name = luisa::format("luisa-render-{}-{}", scene_node_tag_description(tag), impl_type);
    for (auto &c : name) { c = static_cast<char>(std::tolower(c)); }
    auto &&registry = detail::scene_plugin_registry();
    if (auto iter = registry.find(name); iter != registry.end()) {
        return *iter->second;
    }
    auto module = luisa::make_unique<DynamicModule>(DynamicModule::load(runtime_dir, name));
    return *registry.emplace(name, std::move(module)).first->second;
}

}// namespace detail

inline Scene::Scene(const Context &ctx) noexcept:
    _context{ctx}, _config{luisa::make_unique<Scene::Config>()} {}

Scene::NodeHandle Scene::get_node_handle(SceneNodeTag tag, const SceneNodeDesc *desc) noexcept {
    auto &&plugin = detail::scene_plugin_load(_context.runtime_directory(), tag, desc->impl_type());
    auto create = plugin.function<NodeCreater>("create");
    auto destroy = plugin.function<NodeDeleter>("destroy");
    return Scene::NodeHandle(create(this, desc), destroy);
}

SceneNode *Scene::load_node(
    SceneNodeTag tag, const SceneNodeDesc *desc
) noexcept {
    if (desc == nullptr) { return nullptr; }
    if (!desc->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Undefined scene description node '{}' (type = {}::{}).",
            desc->identifier(), scene_node_tag_description(desc->tag()),
            desc->impl_type());
    }
    
    if (desc->is_internal()) {
        std::scoped_lock lock{_mutex};
        return _config->internal_nodes.emplace_back(get_node_handle(tag, desc)).get();
    }
    if (desc->tag() != tag) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid tag {} of scene description node '{}' (expected {}). [{}]",
            scene_node_tag_description(desc->tag()),
            desc->identifier(), scene_node_tag_description(tag),
            desc->source_location().string());
    }
    
    std::scoped_lock lock{_mutex};
    if (auto iter = _config->nodes.find(desc->identifier()); iter != _config->nodes.end()) {
        auto node = iter->second.get();
        if (node->tag() != tag || node->impl_type() != desc->impl_type()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Scene node `{}` (type = {}::{}) is already in the graph (type = {}::{}). [{}]",
                desc->identifier(),
                scene_node_tag_description(tag), desc->impl_type(),
                scene_node_tag_description(node->tag()), node->impl_type(),
                desc->source_location().string());
        }
        node->update(this, desc);
        return node;
    }

    return _config->nodes.emplace(desc->identifier(), get_node_handle(tag, desc)).first->second.get();
}

#define LUISA_SCENE_NODE_LOAD_DEFINITION(name, type, tag)               \
type *Scene::load_##name(const SceneNodeDesc *desc) noexcept {          \
    return dynamic_cast<type *>(load_node(SceneNodeTag::tag, desc));    \
}

LUISA_SCENE_NODE_LOAD_DEFINITION(camera, Camera, CAMERA)
LUISA_SCENE_NODE_LOAD_DEFINITION(film, Film, FILM)
LUISA_SCENE_NODE_LOAD_DEFINITION(filter, Filter, FILTER)
LUISA_SCENE_NODE_LOAD_DEFINITION(integrator, Integrator, INTEGRATOR)
LUISA_SCENE_NODE_LOAD_DEFINITION(surface, Surface, SURFACE)
LUISA_SCENE_NODE_LOAD_DEFINITION(light, Light, LIGHT)
LUISA_SCENE_NODE_LOAD_DEFINITION(sampler, Sampler, SAMPLER)
LUISA_SCENE_NODE_LOAD_DEFINITION(shape, Shape, SHAPE)
LUISA_SCENE_NODE_LOAD_DEFINITION(transform, Transform, TRANSFORM)
LUISA_SCENE_NODE_LOAD_DEFINITION(light_sampler, LightSampler, LIGHT_SAMPLER)
LUISA_SCENE_NODE_LOAD_DEFINITION(environment, Environment, ENVIRONMENT)
LUISA_SCENE_NODE_LOAD_DEFINITION(texture, Texture, TEXTURE)
LUISA_SCENE_NODE_LOAD_DEFINITION(texture_mapping, TextureMapping, TEXTURE_MAPPING)
LUISA_SCENE_NODE_LOAD_DEFINITION(spectrum, Spectrum, SPECTRUM)
LUISA_SCENE_NODE_LOAD_DEFINITION(medium, Medium, MEDIUM)
LUISA_SCENE_NODE_LOAD_DEFINITION(phase_function, PhaseFunction, PHASE_FUNCTION)
LUISA_SCENE_NODE_LOAD_DEFINITION(subsurface, Subsurface, SUBSURFACE)

Environment *Scene::update_environment(const SceneNodeDesc *desc) noexcept {
    return _config->environment = load_environment(desc);
}

Camera *Scene::update_camera(const SceneNodeDesc *desc) noexcept {
    std::scoped_lock lock{_mutex};
    return *(_config->cameras.emplace(load_camera(desc)).first);
}

Shape *Scene::update_shape(const SceneNodeDesc *desc) noexcept {
    std::scoped_lock lock{_mutex};
    return *(_config->shapes.emplace(load_shape(desc)).first);
}

luisa::string Scene::info() const noexcept {
    return luisa::format("Scene integrator=[{}] clamp_normal=[{}]",
        _config->integrator ? _config->integrator->info() : "", _config->clamp_normal);
}

luisa::unique_ptr<Scene> Scene::create(const Context &ctx, const SceneDesc *desc) noexcept {
    if (!desc->root()->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Root node is not defined in the scene description.");
    }
    auto scene = luisa::make_unique<Scene>(ctx);
    scene->_config->shadow_terminator = desc->root()->property_float_or_default("shadow_terminator", 0.f);
    scene->_config->intersection_offset = desc->root()->property_float_or_default("intersection_offset", 0.f);
    scene->_config->clamp_normal = std::clamp(desc->root()->property_float_or_default("clamp_normal", 180.f), 0.f, 180.f);

    scene->_config->spectrum = scene->load_spectrum(
        desc->root()->property_node_or_default(
            "spectrum", SceneNodeDesc::shared_default_spectrum("sRGB")
        ));
    scene->_config->integrator = scene->load_integrator(desc->root()->property_node("integrator"));
    scene->_config->environment = scene->load_environment(desc->root()->property_node_or_default("environment"));
    scene->_config->environment_medium = scene->load_medium(desc->root()->property_node_or_default("environment_medium"));

    auto cameras = desc->root()->property_node_list_or_default("cameras");
    auto shapes = desc->root()->property_node_list_or_default("shapes");
    scene->_config->cameras.reserve(cameras.size());
    scene->_config->shapes.reserve(shapes.size());
    for (auto c : cameras) {
        scene->_config->cameras.emplace(scene->load_camera(c));
    }
    for (auto s : shapes) {
        scene->_config->shapes.emplace(scene->load_shape(s));
    }

    global_thread_pool().synchronize();
    return scene;
}

Scene::~Scene() noexcept = default;

}// namespace luisa::render
