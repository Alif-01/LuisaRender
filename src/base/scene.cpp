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

    bool environment_updated{false};
    bool cameras_updated{false};
    bool shapes_updated{false};
    bool transforms_updated{false};
};

const Integrator *Scene::integrator() const noexcept { return _config->integrator; }
const Environment *Scene::environment() const noexcept { return _config->environment; }
const Medium *Scene::environment_medium() const noexcept { return _config->environment_medium; }
const Spectrum *Scene::spectrum() const noexcept { return _config->spectrum; }
luisa::span<const Shape *const> Scene::shapes() const noexcept { return _config->shapes; }
luisa::span<const Camera *const> Scene::cameras() const noexcept { return _config->cameras; }
float Scene::shadow_terminator_factor() const noexcept { return _config->shadow_terminator; }
float Scene::intersection_offset_factor() const noexcept { return _config->intersection_offset; }

bool Scene::environment_updated() const noexcept { return _config->environment_updated; }
bool Scene::shapes_updated() const noexcept { return _config->shapes_updated; }
bool Scene::cameras_updated() const noexcept { return _config->cameras_updated; }
bool Scene::transforms_updated() const noexcept { return _config->transforms_updated; }
// void Scene::clear_shapes_update() noexcept { _config->shapes_updated = false; }
// void Scene::clear_cameras_update() noexcept { _config->cameras_updated = false; }
// void Scene::clear_transforms_update() noexcept { _config->transforms_updated = false; }
void Scene::clear_update() noexcept {
    _config->environment_updated = false;
    _config->shapes_updated = false;
    _config->cameras_updated = false;
    _config->transforms_updated = false;
}

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

    NodeHandle new_node = handle_creater(std::forward<Args>(args)...);
    auto ptr = new_node.get();
    _config->nodes.emplace(name, std::move(new_node));
    return std::make_pair(ptr, true);
}

template<typename NodeCreater> 
auto Scene::get_handle_creater(
    SceneNodeTag tag, luisa::string_view impl_type, luisa::string_view creater_name
) noexcept {
    return [=, this]<typename... Args>(Args&&... args) -> NodeHandle {
        auto &&plugin = detail::scene_plugin_load(
            _context.runtime_directory(), tag, impl_type);
        auto create = plugin.function<NodeCreater>(creater_name);
        auto destroy = plugin.function<NodeDeleter>("destroy");
        return std::move(NodeHandle(create(std::forward<Args>(args)...), destroy));
    };
}

inline Scene::Scene(const Context &ctx) noexcept
    : _context{ctx},
      _config{luisa::make_unique<Scene::Config>()} {}

SceneNode *Scene::load_node(SceneNodeTag tag, const SceneNodeDesc *desc) noexcept {
    if (desc == nullptr) { return nullptr; }
    if (!desc->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Undefined scene description node '{}' (type = {}::{}).",
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
    
    auto handle_creater = get_handle_creater<NodeCreaterDesc>(tag, desc->impl_type(), "create");

    if (desc->is_internal()) {
        NodeHandle node = handle_creater(this, desc);
        std::scoped_lock lock{_mutex};
        return _config->internal_nodes.emplace_back(std::move(node)).get();
    }
    if (desc->tag() != tag) [[unlikely]] {
        LUISA_ERROR(
            "Invalid tag {} of scene description node '{}' (expected {}). [{}]",
            scene_node_tag_description(desc->tag()),
            desc->identifier(),
            scene_node_tag_description(tag),
            desc->source_location().string());
    }
    
    auto [node, first_def] = load_from_nodes(desc->identifier(), handle_creater, this, desc);

    if (!first_def && (node->tag() != tag || node->impl_type() != desc->impl_type())) [[unlikely]] {
        LUISA_ERROR(
            "Scene node `{}` (type = {}::{}) is already in the graph (type = {}::{}). [{}]",
            desc->identifier(), scene_node_tag_description(tag),
            desc->impl_type(), scene_node_tag_description(node->tag()),
            node->impl_type(), desc->source_location().string());
    }
    return node;
}

SceneNode *Scene::load_node_from_name(luisa::string_view name) noexcept {
    if (name.empty()) {
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
    using NodeCreater = SceneNode *(Scene *, const uint2 &);
    auto handle_creater = get_handle_creater<NodeCreater>(SceneNodeTag::FILM, "color", "create_raw");
    NodeHandle color_film = handle_creater(this, resolution);

    std::scoped_lock lock{_mutex};
    return dynamic_cast<Film *>(
        _config->internal_nodes.emplace_back(std::move(color_film)).get()
    );
}

Filter *Scene::add_filter(luisa::string_view name, const float &radius) noexcept {
    using NodeCreater = SceneNode *(Scene *, const float &);
    auto handle_creater = get_handle_creater<NodeCreater>(SceneNodeTag::FILTER, "gaussian", "create_raw");
    NodeHandle handle = handle_creater(this, radius);

    std::scoped_lock lock{_mutex};
    return dynamic_cast<Filter *>(
        _config->internal_nodes.emplace_back(std::move(handle)).get()
    );
}

Environment *Scene::add_environment(const RawEnvironmentInfo &environment_info) noexcept {
    using NodeCreater = SceneNode *(Scene *, const RawEnvironmentInfo &);
    auto handle_creater = get_handle_creater<NodeCreater>(SceneNodeTag::ENVIRONMENT, "spherical", "create_raw");
    NodeHandle handle = handle_creater(this, environment_info);

    std::scoped_lock lock{_mutex};
    auto node = _config->internal_nodes.emplace_back(std::move(handle)).get();
    Environment *environment = dynamic_cast<Environment *>(node);
    _config->environment = environment;
    _config->environment_updated = true;

    return environment;
}

Light *Scene::add_light(const RawLightInfo &light_info) noexcept {
    using NodeCreater = SceneNode *(Scene *, const RawLightInfo &);
    // luisa::string impl_type = texture_info.is_image() ? "image" : "constant";

    auto handle_creater = get_handle_creater<NodeCreater>(SceneNodeTag::LIGHT, "diffuse", "create_raw");
    auto [node, first_def] = load_from_nodes(light_info.name, handle_creater, this, light_info);
    Light *light = dynamic_cast<Light *>(node);

    if (!first_def) {
        LUISA_ERROR_WITH_LOCATION("Light {} has been defined.", light_info.name);
    }
    return light;
}

Texture *Scene::add_texture(luisa::string_view name, const RawTextureInfo &texture_info) noexcept {
    if (texture_info.empty) return nullptr;

    using NodeCreater = SceneNode *(Scene *, const RawTextureInfo &);
    luisa::string impl_type = texture_info.is_image() ? "image" : "constant";

    auto handle_creater = get_handle_creater<NodeCreater>(SceneNodeTag::TEXTURE, impl_type, "create_raw");
    NodeHandle handle = handle_creater(this, texture_info);
    
    std::scoped_lock lock{_mutex};
    return dynamic_cast<Texture *>(_config->internal_nodes.emplace_back(std::move(handle)).get());
}

Texture *Scene::add_constant_texture(luisa::string_view name, const luisa::vector<float> &v) noexcept {
    using NodeCreater = SceneNode *(Scene *, const luisa::vector<float> &);
    auto handle_creater = get_handle_creater<NodeCreater>(SceneNodeTag::TEXTURE, "constant", "create_dir");
    NodeHandle handle = handle_creater(this, v);
    
    std::scoped_lock lock{_mutex};
    return dynamic_cast<Texture *>(_config->internal_nodes.emplace_back(std::move(handle)).get());
}

// Texture *Scene::add_image_texture(
//     luisa::string_view name, luisa::string_view image, const float &image_scale
// ) noexcept {
//     using NodeCreater = SceneNode *(Scene *, luisa::string_view, const float &);
//     auto handle_creater = get_handle_creater<NodeCreater>(SceneNodeTag::TEXTURE, "image", "create_raw");
//     NodeHandle handle = handle_creater(this, image, image_scale);
    
//     std::scoped_lock lock{_mutex};
//     return dynamic_cast<Texture *>(
//         _config->internal_nodes.emplace_back(std::move(handle)).get()
//     );
// }
Transform *Scene::update_transform(luisa::string_view name, const RawTransformInfo &transform_info) noexcept {
    if (transform_info.empty) return nullptr;

    using NodeCreater = SceneNode *(Scene *, const RawTransformInfo &);
    luisa::string impl_type = transform_info.is_matrix() ? "matrix" :
                              transform_info.is_srt() ? "srt": "";
    if (impl_type.empty()) return nullptr;
    
    auto handle_creater = get_handle_creater<NodeCreater>(SceneNodeTag::TRANSFORM, impl_type, "create_raw");
    auto [node, first_def] = load_from_nodes(luisa::format("{}_{}", name, impl_type), handle_creater, this, transform_info);
    Transform *transform = dynamic_cast<Transform *>(node);

    if (!first_def) {
        transform->update_transform(this, transform_info);
        _config->transforms_updated = true;
    }
    return transform;
}

Camera *Scene::add_camera(const RawCameraInfo &camera_info) noexcept {
    using NodeCreater = SceneNode *(Scene *, const RawCameraInfo &);
    auto handle_creater = get_handle_creater<NodeCreater>(SceneNodeTag::CAMERA, "pinhole", "create_raw");
    auto [node, first_def] = load_from_nodes(camera_info.name, handle_creater, this, camera_info);
    Camera *camera = dynamic_cast<Camera *>(node);

    if (first_def) {
        _config->cameras.emplace_back(camera);
        _config->cameras_updated = true;
    } else {
        LUISA_ERROR_WITH_LOCATION("Camera {} has been defined.", camera_info.name);
    }
    return camera;
}

Camera *Scene::update_camera(luisa::string_view name, const RawTransformInfo &transform_info) noexcept {
    auto node = load_node_from_name(name);
    Camera *camera = dynamic_cast<Camera *>(node);
    _config->cameras_updated |= camera->update_camera(this, name, transform_info);
    return camera;
}

Surface *Scene::add_surface(const RawSurfaceInfo &surface_info) noexcept {
    using NodeCreater = SceneNode *(Scene *, const RawSurfaceInfo &);
    auto handle_creater = get_handle_creater<NodeCreater>(
        SceneNodeTag::SURFACE, RawSurfaceInfo::mat_string[surface_info.material], "create_raw"
    );
    auto [node, first_def] = load_from_nodes(surface_info.name, handle_creater, this, surface_info);
    Surface *surface = dynamic_cast<Surface *>(node);

    if (!first_def) {
        LUISA_ERROR_WITH_LOCATION("Surface {} has been defined.", surface_info.name);
    }
    return surface;
}

// Shape *Scene::update_particles(const RawShapeInfo &shape_info) noexcept {
//     using NodeCreater = SceneNode *(Scene *, const RawShapeInfo &);
//     auto handle_creater = get_handle_creater<NodeCreater>(SceneNodeTag::SHAPE, , "create_raw");
//     auto [node, first_def] = load_from_nodes(shape_info.name, handle_creater, this, shape_info);
//     Shape *shape = dynamic_cast<Shape *>(node);

//     if (first_def) {
//         _config->shapes.emplace_back(shape);
//     } else {
//         node->update_shape(this, shape_info);
//     }

    // shapes.emplace_back(shape);
    // for (auto i = 0u; i < sphere_infos.size(); ++i) {
    //     auto sphere_info = sphere_infos[i];
    //     auto name = luisa::format("{}_{}", sphere_info.shape_info.name, i);
    //     auto handle_creater = get_handle_creater<NodeCreater>(SceneNodeTag::SHAPE, "sphere", "create_raw");
    //     auto [node, first_def] = load_from_nodes(name, handle_creater, this, sphere_info);
    //     Shape *shape = dynamic_cast<Shape *>(node);
    //     if (first_def) {
    //         _config->shapes.emplace_back(shape);
    //     } else {
    //         shape->update_shape(this, sphere_info.shape_info);
    //     }
    //     shapes.emplace_back(shape);
    // }
//     _config->shapes_updated = true;
//     return shape;
// }

Shape *Scene::update_shape(
    const RawShapeInfo &shape_info, luisa::string impl_type, bool require_first
) noexcept {
    using NodeCreater = SceneNode *(Scene *, const RawShapeInfo &);
    auto handle_creater = get_handle_creater<NodeCreater>(SceneNodeTag::SHAPE, impl_type, "create_raw");
    auto [node, first_def] = load_from_nodes(shape_info.name, handle_creater, this, shape_info);
    Shape *shape = dynamic_cast<Shape *>(node);

    if (first_def) {
        _config->shapes.emplace_back(shape);
    } else {
        if (require_first)
            LUISA_ERROR_WITH_LOCATION("Shape has been defined!");
        shape->update_shape(this, shape_info);
    }
    _config->shapes_updated = true;
    return shape;
}

luisa::unique_ptr<Scene> Scene::create(const Context &ctx, const SceneDesc *desc) noexcept {
    if (!desc->root()->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Root node is not defined in the scene description.");
    }
    auto scene = luisa::make_unique<Scene>(ctx);
    scene->_config->shadow_terminator = desc->root()->property_float_or_default("shadow_terminator", 0.f);
    scene->_config->intersection_offset = desc->root()->property_float_or_default("intersection_offset", 0.f);
    scene->_config->spectrum = scene->load_spectrum(desc->root()->property_node_or_default(
        "spectrum", SceneNodeDesc::shared_default_spectrum("sRGB")));
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

    scene->_config->cameras_updated = scene->_config->cameras.size() > 0;
    scene->_config->shapes_updated = scene->_config->shapes.size() > 0;
    scene->_config->environment_updated = scene->_config->environment != nullptr;

    global_thread_pool().synchronize();
    return scene;
}

Scene::~Scene() noexcept = default;

}// namespace luisa::render
