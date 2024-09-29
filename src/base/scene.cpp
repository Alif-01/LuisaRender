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

namespace luisa::render {

// const Integrator *Scene::integrator() const noexcept { return _config->integrator; }
// const Environment *Scene::environment() const noexcept { return _config->environment; }
// const Medium *Scene::environment_medium() const noexcept { return _config->environment_medium; }
// const Spectrum *Scene::spectrum() const noexcept { return _config->spectrum; }
// luisa::span<const Shape *const> Scene::shapes() const noexcept { return _config->shapes; }
// luisa::span<const Camera *const> Scene::cameras() const noexcept { return _config->cameras; }
// float Scene::shadow_terminator_factor() const noexcept { return _config->shadow_terminator; }
// float Scene::intersection_offset_factor() const noexcept { return _config->intersection_offset; }
// float Scene::clamp_normal_factor() const noexcept { return _config->clamp_normal; }

luisa::string Scene::info() const noexcept {
    return luisa::format("Scene integrator=[{}] clamp_normal=[{}]",
        _config->integrator ? _config->integrator->info() : "", _config->clamp_normal);
}
// bool Scene::environment_updated() const noexcept { return _config->environment_updated; }
// bool Scene::shapes_updated() const noexcept { return _config->shapes_updated; }
// bool Scene::cameras_updated() const noexcept { return _config->cameras_updated; }
// bool Scene::transforms_updated() const noexcept { return _config->transforms_updated; }
// void Scene::clear_update() noexcept {
//     _config->environment_updated = false;
//     _config->shapes_updated = false;
//     _config->cameras_updated = false;
//     _config->transforms_updated = false;
// }

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

// template <typename... Args, typename Callable>
// std::pair<SceneNode*, bool> Scene::load_from_nodes(
//     luisa::string_view name, Callable &&handle_creater, Args&&... args
// ) noexcept {
//     std::scoped_lock lock{_mutex};
//     if (auto iter = _config->nodes.find(name);
//         iter != _config->nodes.end()) {
//         return std::make_pair(iter->second.get(), false);
//     }

//     NodeHandle new_node = handle_creater(std::forward<Args>(args)...);
//     auto ptr = new_node.get();
//     _config->nodes.emplace(name, std::move(new_node));
//     return std::make_pair(ptr, true);
// }

// template<typename NodeCreater> 
// auto Scene::get_handle_creater(
//     SceneNodeTag tag, luisa::string_view impl_type, luisa::string_view creater_name
// ) noexcept {
//     return [=, this]<typename... Args>(Args&&... args) -> NodeHandle {
//         auto &&plugin = detail::scene_plugin_load(
//             _context.runtime_directory(), tag, impl_type);
//         auto create = plugin.function<NodeCreater>(creater_name);
//         auto destroy = plugin.function<NodeDeleter>("destroy");
//         return NodeHandle(create(std::forward<Args>(args)...), destroy);
//     };
// }

NodeHandle Scene::get_node_handle(SceneNodeTag tag, const SceneNodeDesc *desc) noexcept {
    auto &&plugin = detail::scene_plugin_load(_context.runtime_directory(), tag, desc->impl_type());
    auto create = plugin.function<NodeCreater>("create");
    auto destroy = plugin.function<NodeDeleter>("destroy");
    return NodeHandle(create(this, desc), destroy);
}

// template <typename Callable>
SceneNode *Scene::load_node(
    SceneNodeTag tag, const SceneNodeDesc *desc //, Callable &&updated_callback
) noexcept {
    if (desc == nullptr) { return nullptr; }
    if (!desc->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Undefined scene description node '{}' (type = {}::{}).",
            desc->identifier(), scene_node_tag_description(desc->tag()),
            desc->impl_type());
    }
    
    // auto handle_creater = get_handle_creater<NodeCreaterDesc>(tag, desc->impl_type(), "create");
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
    if (auto iter = _config->nodes.find(name); iter != _config->nodes.end()) {
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
    // NodeHandle new_node_handle = handle_creater(std::forward<Args>(args)...);
    // auto ptr = new_node.get();
    // _config->nodes.emplace(name, std::move(new_node));
    // return std::make_pair(ptr, true);


    // auto [node, first_def] = load_from_nodes(desc->identifier(), handle_creater, this, desc);
    // bool updated = false;
    // if (!first_def) {
        // updated |= node->update(this, desc);
    // } else {
        // updated = true;
    // }
    // if (updated) updated_callback(node);
    // return node;
}

// Camera *Scene::load_camera(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<Camera *>(load_node(SceneNodeTag::CAMERA, desc,
//         [this](SceneNode *){ _config->cameras_updated = true; }
//     ));
// }

// Film *Scene::load_film(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<Film *>(load_node(SceneNodeTag::FILM, desc, [](SceneNode *){}));
// }

// Filter *Scene::load_filter(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<Filter *>(load_node(SceneNodeTag::FILTER, desc, [](SceneNode *){}));
// }

// Integrator *Scene::load_integrator(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<Integrator *>(load_node(SceneNodeTag::INTEGRATOR, desc, [](SceneNode *){}));
// }

// Surface *Scene::load_surface(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<Surface *>(load_node(SceneNodeTag::SURFACE, desc, [](SceneNode *){}));
// }

// Light *Scene::load_light(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<Light *>(load_node(SceneNodeTag::LIGHT, desc, [](SceneNode *){}));
// }

// Sampler *Scene::load_sampler(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<Sampler *>(load_node(SceneNodeTag::SAMPLER, desc, [](SceneNode *){}));
// }

// Shape *Scene::load_shape(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<Shape *>(load_node(SceneNodeTag::SHAPE, desc,
//         [this](SceneNode *){ _config->shapes_updated = true; }));
// }

// Transform *Scene::load_transform(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<Transform *>(load_node(SceneNodeTag::TRANSFORM, desc, 
//         [this](SceneNode *node){
//             if (dynamic_cast<Transform *>(node)->is_registered())
//                 _config->transforms_updated = true;
//         }
//     ));
// }

// LightSampler *Scene::load_light_sampler(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<LightSampler *>(load_node(SceneNodeTag::LIGHT_SAMPLER, desc, [](SceneNode *node){}));
// }

// Environment *Scene::load_environment(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<Environment *>(load_node(SceneNodeTag::ENVIRONMENT, desc,
//         [this](SceneNode *node){ _config->environment_updated = true; }
//     ));
// }

// Texture *Scene::load_texture(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<Texture *>(load_node(SceneNodeTag::TEXTURE, desc, [](SceneNode *node){}));
// }

// TextureMapping *Scene::load_texture_mapping(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<TextureMapping *>(load_node(SceneNodeTag::TEXTURE_MAPPING, desc, [](SceneNode *node){}));
// }

// Spectrum *Scene::load_spectrum(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<Spectrum *>(load_node(SceneNodeTag::SPECTRUM, desc, [](SceneNode *node){}));
// }

// Medium *Scene::load_medium(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<Medium *>(load_node(SceneNodeTag::MEDIUM, desc, [](SceneNode *node){}));
// }

// PhaseFunction *Scene::load_phase_function(const SceneNodeDesc *desc) noexcept {
//     return dynamic_cast<PhaseFunction *>(load_node(SceneNodeTag::PHASE_FUNCTION, desc, [](SceneNode *node){}));
// }

Environment *Scene::update_environment(const SceneNodeDesc *desc) noexcept {
    return _config->environment = load_environment(desc);
}

// std::pair<Camera *, uint> Scene::update_camera(const SceneNodeDesc *desc, bool first_def) noexcept {
Camera *Scene::update_camera(const SceneNodeDesc *desc) noexcept {
    // uint index = 0;
    std::scoped_lock lock{_mutex};
    return *(_config->cameras.emplace(load_camera(desc)).first);
    // if (first_def) {
    //     auto &cameras = _config->cameras;
    //     if (std::find(cameras.begin(), cameras.end(), camera) != cameras.end()) [[unlikely]] {
    //         LUISA_ERROR_WITH_LOCATION("Camera is not defined the first time.");
    //     }
    //     index = cameras.size();
    //     cameras.emplace_back(camera);
    // }
}

// Shape *Scene::update_shape(const SceneNodeDesc *desc, bool first_def) noexcept {
Shape *Scene::update_shape(const SceneNodeDesc *desc) noexcept {
    std::scoped_lock lock{_mutex};
    return *(_config->shapes.emplace(load_shape(desc)).first);
    // auto shape = load_shape(desc);
    // if (first_def) {
    //     std::scoped_lock lock{_mutex};
    //     auto &shapes = _config->shapes;
    //     if (std::find(shapes.begin(), shapes.end(), shape) != shapes.end()) [[unlikely]] {
    //         LUISA_ERROR_WITH_LOCATION("Shape is not defined the first time.");
    //     }
    //     shapes.emplace_back(shape);
    // }
    // return shape;
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
        scene->_config->cameras.emplace_back(scene->load_camera(c));
    }
    for (auto s : shapes) {
        scene->_config->shapes.emplace_back(scene->load_shape(s));
    }

    global_thread_pool().synchronize();
    return scene;
}

Scene::~Scene() noexcept = default;

}// namespace luisa::render
