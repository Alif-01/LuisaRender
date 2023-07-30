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
#include "scene.h"

namespace luisa::render {

struct Scene::Config {
    float shadow_terminator{0.f};
    float intersection_offset{0.f};
    luisa::vector<NodeHandle> internal_nodes;
    luisa::unordered_map<luisa::string, NodeHandle> nodes;
    Integrator *integrator{nullptr};
    Environment *environment{nullptr};
    Medium *environment_medium{nullptr};
    Spectrum *spectrum{nullptr};
    luisa::vector<Camera *> cameras;
    luisa::vector<Shape *> shapes;
    bool cameras_updated;
    bool shapes_updated;
};

const Integrator *Scene::integrator() const noexcept { return _config->integrator; }
const Environment *Scene::environment() const noexcept { return _config->environment; }
const Medium *Scene::environment_medium() const noexcept { return _config->environment_medium; }
const Spectrum *Scene::spectrum() const noexcept { return _config->spectrum; }
luisa::span<const Shape *const> Scene::shapes() const noexcept { return _config->shapes; }
luisa::span<const Camera *const> Scene::cameras() const noexcept { return _config->cameras; }
bool Scene::shapes_updated() const noexcept { return _config->shapes_updated; }
bool Scene::cameras_updated() const noexcept { return _config->cameras_updated; }
void Scene::clear_shapes_update() noexcept { _config->shapes_updated = false; }
void Scene::clear_cameras_update() noexcept { _config->cameras_updated = false; }
float Scene::shadow_terminator_factor() const noexcept { return _config->shadow_terminator; }
float Scene::intersection_offset_factor() const noexcept { return _config->intersection_offset; }

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

template <typename... Args, typename Callable>
std::pair<SceneNode*, bool> Scene::load_from_nodes(
    luisa::string_view name, Callable &&handle_creater, Args&&... args
) noexcept {
    std::scoped_lock lock{_mutex};
    if (auto iter = _config->nodes.find(name);
        iter != _config->nodes.end()) {
        return std::make_pair(iter->second.get(), false);
    }

    // NodeHandle new_node{create(this, desc), destroy};
    NodeHandle new_node = handle_creater(std::forward<Args>(args)...);
    auto ptr = new_node.get();
    _config->nodes.emplace(name, std::move(new_node));
    return std::make_pair(ptr, true);
}

template<typename NodeCreater> 
auto Scene::get_handle_creater(
    luisa::string_view name, SceneNodeTag tag,
    luisa::string_view impl_type, luisa::string_view creater_name
) noexcept {
    return [=, this]<typename... Args>(Args&&... args) -> NodeHandle {
        auto &&plugin = detail::scene_plugin_load(
            _context.runtime_directory(), tag, impl_type);
        auto create = plugin.function<NodeCreater>(creater_name);
        auto destroy = plugin.function<NodeDeleter>("destroy");
        LUISA_VERBOSE_WITH_LOCATION("Constructing scene node '{}'.", name);
        return NodeHandle(create(std::forward<Args>(args)...), destroy);
    };
}

inline Scene::Scene(const Context &ctx) noexcept
    : _context{ctx},
      _config{luisa::make_unique<Scene::Config>()} {}

SceneNode *Scene::load_node(SceneNodeTag tag, const SceneNodeDesc *desc) noexcept {
    if (desc == nullptr) { return nullptr; }
    if (!desc->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Undefined scene description "
            "node '{}' (type = {}::{}).",
            desc->identifier(),
            scene_node_tag_description(desc->tag()),
            desc->impl_type());
    }

    // auto &&plugin = detail::scene_plugin_load(
    //     _context.runtime_directory(), tag, desc->impl_type());
    // auto create = plugin.function<NodeCreater>("create");
    // auto destroy = plugin.function<NodeDeleter>("destroy");
    // auto handle_creater = [&] {
    //     LUISA_VERBOSE_WITH_LOCATION(
    //         "Constructing scene graph node '{}' (desc = {}).",
    //         , fmt::ptr(desc));
    //     return std::move(NodeHandle{create(), destroy});
    // };
    
    auto handle_creater = get_handle_creater<NodeCreaterDesc>(
        desc->identifier(), tag, desc->impl_type(), "create"
    );

    if (desc->is_internal()) {
        NodeHandle node = handle_creater(this, desc);
        std::scoped_lock lock{_mutex};
        return _config->internal_nodes.emplace_back(std::move(node)).get();
    }
    if (desc->tag() != tag) [[unlikely]] {
        LUISA_ERROR(
            "Invalid tag {} of scene description "
            "node '{}' (expected {}). [{}]",
            scene_node_tag_description(desc->tag()),
            desc->identifier(),
            scene_node_tag_description(tag),
            desc->source_location().string());
    }
    
    // auto message = [&] {// };
    // NodeHandle new_node{create(this, desc), destroy};
    auto [node, first_def] = load_from_nodes(
        desc->identifier(), handle_creater, this, desc
    );

    if (!first_def && (node->tag() != tag ||
                       node->impl_type() != desc->impl_type())) [[unlikely]] {
        LUISA_ERROR(
            "Scene node `{}` (type = {}::{}) is already "
            "in the graph (type = {}::{}). [{}]",
            desc->identifier(), scene_node_tag_description(tag),
            desc->impl_type(), scene_node_tag_description(node->tag()),
            node->impl_type(), desc->source_location().string());
    }
    return node;
}

SceneNode *Scene::load_node_from_name(luisa::string name) noexcept {
    if (name.empty()) {
        // LUISA_WARNING_WITH_LOCATION("Scene node {} not found");
        return nullptr;
    }
    
    std::scoped_lock lock{_mutex};
    if (auto iter = _config->nodes.find(name);
        iter != _config->nodes.end()) {
        return iter->second.get();
    }

    LUISA_ERROR_WITH_LOCATION("Scene node `{}` not founded.", name);
}

Camera *Scene::load_camera(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Camera *>(load_node(SceneNodeTag::CAMERA, desc));
}

Film *Scene::load_film(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Film *>(load_node(SceneNodeTag::FILM, desc));
}

Filter *Scene::load_filter(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Filter *>(load_node(SceneNodeTag::FILTER, desc));
}

Integrator *Scene::load_integrator(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Integrator *>(load_node(SceneNodeTag::INTEGRATOR, desc));
}

Surface *Scene::load_surface(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Surface *>(load_node(SceneNodeTag::SURFACE, desc));
}

Light *Scene::load_light(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Light *>(load_node(SceneNodeTag::LIGHT, desc));
}

Sampler *Scene::load_sampler(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Sampler *>(load_node(SceneNodeTag::SAMPLER, desc));
}

Shape *Scene::load_shape(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Shape *>(load_node(SceneNodeTag::SHAPE, desc));
}

Transform *Scene::load_transform(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Transform *>(load_node(SceneNodeTag::TRANSFORM, desc));
}

LightSampler *Scene::load_light_sampler(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<LightSampler *>(load_node(SceneNodeTag::LIGHT_SAMPLER, desc));
}

Environment *Scene::load_environment(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Environment *>(load_node(SceneNodeTag::ENVIRONMENT, desc));
}

Texture *Scene::load_texture(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Texture *>(load_node(SceneNodeTag::TEXTURE, desc));
}

TextureMapping *Scene::load_texture_mapping(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<TextureMapping *>(load_node(SceneNodeTag::TEXTURE_MAPPING, desc));
}

Spectrum *Scene::load_spectrum(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Spectrum *>(load_node(SceneNodeTag::SPECTRUM, desc));
}

Medium *Scene::load_medium(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<Medium *>(load_node(SceneNodeTag::MEDIUM, desc));
}

PhaseFunction *Scene::load_phase_function(const SceneNodeDesc *desc) noexcept {
    return dynamic_cast<PhaseFunction *>(load_node(SceneNodeTag::PHASE_FUNCTION, desc));
}

Film *Scene::add_film(luisa::string_view name, const uint2 &resolution) noexcept {
    // typedef SceneNode *(*NodeCreater)(Scene *, const uint2 &);
    using NodeCreater = SceneNode *(Scene *, const uint2 &);
    auto handle_creater = get_handle_creater<NodeCreater>(name, SceneNodeTag::FILM, "color", "create_raw");
    NodeHandle node = handle_creater(this, resolution);

    std::scoped_lock lock{_mutex};
    return dynamic_cast<Film *>(
        _config->internal_nodes.emplace_back(std::move(node)).get()
    );
}

Filter *Scene::add_filter(luisa::string_view name, const float &radius) noexcept {
    // typedef SceneNode *(*NodeCreater)(Scene *, const float &);
    using NodeCreater = SceneNode *(Scene *, const float &);
    auto handle_creater = get_handle_creater<NodeCreater>(name, SceneNodeTag::FILTER, "gaussian", "create_raw");
    NodeHandle node = handle_creater(this, radius);

    std::scoped_lock lock{_mutex};
    return dynamic_cast<Filter *>(
        _config->internal_nodes.emplace_back(std::move(node)).get()
    );
}

Transform *Scene::add_transform(luisa::string_view name, const luisa::vector<float> &m) noexcept {
    // typedef SceneNode *(*NodeCreater)(Scene *, const luisa::vector<float> &);
    using NodeCreater = SceneNode *(Scene *, const luisa::vector<float> &);
    auto handle_creater = get_handle_creater<NodeCreater>(name, SceneNodeTag::TRANSFORM, "matrix", "create_raw");
    NodeHandle node = handle_creater(this, m);
    
    std::scoped_lock lock{_mutex};
    return dynamic_cast<Transform *>(
        _config->internal_nodes.emplace_back(std::move(node)).get()
    );
}

Texture *Scene::add_constant_texture(luisa::string_view name, const luisa::vector<float> &v) noexcept {
    // typedef SceneNode *(*NodeCreater)(Scene *, const luisa::vector<float> &);
    using NodeCreater = SceneNode *(Scene *, const luisa::vector<float> &);
    auto handle_creater = get_handle_creater<NodeCreater>(name, SceneNodeTag::TEXTURE, "constant", "create_raw");
    NodeHandle node = handle_creater(this, v);
    
    std::scoped_lock lock{_mutex};
    return dynamic_cast<Texture *>(
        _config->internal_nodes.emplace_back(std::move(node)).get()
    );
}

Texture *Scene::add_image_texture(
    luisa::string_view name, luisa::string_view image, const float &image_scale
) noexcept {
    // typedef SceneNode *(*NodeCreater)(Scene *, luisa::string_view, const float &);
    using NodeCreater = SceneNode *(Scene *, luisa::string_view, const float &);
    auto handle_creater = get_handle_creater<NodeCreater>(name, SceneNodeTag::TEXTURE, "image", "create_raw");
    NodeHandle node = handle_creater(this, image, image_scale);
    
    std::scoped_lock lock{_mutex};
    return dynamic_cast<Texture *>(
        _config->internal_nodes.emplace_back(std::move(node)).get()
    );
}

Camera *Scene::add_camera(
    const RawCameraInfo &camera_info,
    luisa::unordered_map<luisa::string, uint> &camera_index) noexcept {
    // typedef SceneNode *(*NodeCreater)(Scene *, const RawCameraInfo &);
    using NodeCreater = SceneNode *(Scene *, const RawCameraInfo &);
    auto handle_creater = get_handle_creater<NodeCreater>(camera_info.name, SceneNodeTag::CAMERA, "pinhole", "create_raw");
    auto [node, first_def] = load_from_nodes(camera_info.name, handle_creater, this, camera_info);
    Camera *camera = dynamic_cast<Camera *>(node);

    if (first_def) {
        camera_index[camera_info.name] = _config->cameras.size();
        _config->cameras.emplace_back(camera);
        _config->cameras_updated = true;
    } else {
        LUISA_ERROR_WITH_LOCATION("Camera {} has been defined.", camera_info.name);
    }
    return camera;
}

Surface *Scene::add_surface(const RawSurfaceInfo &surface_info) noexcept {
    // typedef SceneNode *(*NodeCreater)(Scene *, const RawSurfaceInfo &);
    using NodeCreater = SceneNode *(Scene *, const RawSurfaceInfo &);
    auto handle_creater = get_handle_creater<NodeCreater>(
        surface_info.name, SceneNodeTag::SURFACE,
        RawSurfaceInfo::mat_string[surface_info.material], "create_raw"
    );
    auto [node, first_def] = load_from_nodes(surface_info.name, handle_creater, this, surface_info);
    Surface *surface = dynamic_cast<Surface *>(node);

    if (!first_def) {
        LUISA_ERROR_WITH_LOCATION("Surface {} has been defined.", surface_info.name);
    }
    return surface;
}

Shape *Scene::update_shape(const RawMeshInfo &mesh_info) noexcept {
    // typedef SceneNode *(*NodeCreater)(Scene *, const RawMeshInfo &);
    using NodeCreater = SceneNode *(Scene *, const RawMeshInfo &);
    auto handle_creater = get_handle_creater<NodeCreater>(mesh_info.name, SceneNodeTag::SHAPE, "dynamicmesh", "create_raw");
    auto [node, first_def] = load_from_nodes(mesh_info.name, handle_creater, this, mesh_info);
    Shape *shape = dynamic_cast<Shape *>(node);

    if (first_def) _config->shapes.emplace_back(shape);
    else shape->update_shape(this, mesh_info);
    _config->shapes_updated = true;
    return shape;
}

void Scene::append_shape(Shape *shape) noexcept {
    _config->shapes.emplace_back(shape);
}

luisa::unique_ptr<Scene> Scene::create(
    const Context &ctx, const SceneDesc *desc,
    luisa::unordered_map<luisa::string, uint> &camera_index) noexcept {
    if (!desc->root()->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Root node is not defined in the scene description.");
    }
    auto scene = luisa::make_unique<Scene>(ctx);
    scene->_config->shadow_terminator = desc->root()->property_float_or_default("shadow_terminator", 0.f);
    scene->_config->intersection_offset = desc->root()->property_float_or_default("intersection_offset", 0.f);
    scene->_config->spectrum = scene->load_spectrum(desc->root()->property_node_or_default(
        "spectrum", SceneNodeDesc::shared_default_spectrum("sRGB")));
    scene->_config->integrator = scene->load_integrator(
        desc->root()->property_node("integrator"));
    scene->_config->environment = scene->load_environment(
        desc->root()->property_node_or_default("environment"));
    scene->_config->environment_medium = scene->load_medium(
        desc->root()->property_node_or_default("environment_medium"));
    auto cameras = desc->root()->property_node_list("cameras");
    auto shapes = desc->root()->property_node_list("shapes");
    auto environments = desc->root()->property_node_or_default("environments", SceneNodeDesc::shared_default_medium("Null"));
    scene->_config->cameras.reserve(cameras.size());
    scene->_config->shapes.reserve(shapes.size());
    scene->_config->cameras_updated = cameras.size() > 0;
    scene->_config->shapes_updated = shapes.size() > 0;
    for (auto c : cameras) {
        camera_index[c->identifier()] = scene->_config->cameras.size();
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
