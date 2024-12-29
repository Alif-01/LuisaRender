//
// Created by Mike on 2021/12/15.
//

#pragma once

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/buffer_arena.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/rtx/accel.h>

#include <util/spec.h>
#include <base/shape.h>
#include <base/light.h>
#include <base/camera.h>
#include <base/film.h>
#include <base/filter.h>
#include <base/surface.h>
#include <base/transform.h>
#include <base/integrator.h>
#include <base/interaction.h>
#include <base/light_sampler.h>
#include <base/environment.h>
#include <base/texture.h>
#include <base/geometry.h>
#include <base/medium.h>
#include <base/phase_function.h>
#include <base/subsurface.h>

namespace luisa::render {

using compute::Accel;
using compute::AccelOption;
using compute::BindlessArray;
using compute::BindlessBuffer;
using compute::BindlessTexture2D;
using compute::BindlessTexture3D;
using compute::Buffer;
using compute::BufferArena;
using compute::BufferView;
using compute::Callable;
using compute::Device;
using compute::Image;
using compute::PixelStorage;
using compute::Polymorphic;
using compute::Resource;
using compute::Volume;
using TextureSampler = compute::Sampler;

class Scene;

class Pipeline {

public:
    static constexpr auto bindless_array_capacity = 500'000u;   // limitation of Metal
    static constexpr auto transform_matrix_buffer_size = 65536u;
    static constexpr auto constant_buffer_size = 256u * 1024u;
    using ResourceHandle = luisa::unique_ptr<Resource>;

private:
    Device &_device;
    Scene &_scene;
    luisa::unique_ptr<BufferArena> _general_buffer_arena;
    luisa::vector<ResourceHandle> _resources;
    size_t _bindless_buffer_count{0u};
    size_t _bindless_tex2d_count{0u};
    size_t _bindless_tex3d_count{0u};
    BindlessArray _bindless_array;
    
    size_t _constant_count{0u};
    Buffer<float4> _constant_buffer;

    luisa::unordered_map<const Surface *, uint> _surface_tags;
    luisa::unordered_map<const Light *, uint> _light_tags;
    luisa::unordered_map<const Medium *, uint> _medium_tags;
    luisa::unordered_map<const Subsurface *, uint> _subsurface_tags;
    Polymorphic<Surface::Instance> _surfaces;
    Polymorphic<Light::Instance> _lights;
    Polymorphic<Medium::Instance> _media;
    Polymorphic<Subsurface::Instance> _subsurfaces;

    luisa::unordered_map<Transform *, uint> _transform_to_id;
    luisa::vector<float4x4> _transform_matrices;
    Buffer<float4x4> _transform_matrix_buffer;
    bool _transforms_dirty{false};
    bool _any_dynamic_transforms{false};

    luisa::unordered_map<const Texture *, luisa::unique_ptr<Texture::Instance>> _textures;
    luisa::unordered_map<const Filter *, luisa::unique_ptr<Filter::Instance>> _filters;
    luisa::unordered_map<const PhaseFunction *, luisa::unique_ptr<PhaseFunction::Instance>> _phasefunctions;
    luisa::unordered_map<const Camera *, luisa::unique_ptr<Camera::Instance>> _cameras;
    luisa::unique_ptr<Spectrum::Instance> _spectrum;
    luisa::unique_ptr<Integrator::Instance> _integrator;
    luisa::unique_ptr<Environment::Instance> _environment;
    uint _environment_medium_tag{Medium::INVALID_TAG};
    luisa::unique_ptr<Geometry> _geometry;

    // registered transforms
    luisa::unordered_map<luisa::string, uint> _named_ids;

    // other things
    float _time{0.f};

public:
    // for internal use only; use Pipeline::create() instead
    explicit Pipeline(Device &device, Scene &scene) noexcept;
    Pipeline(Pipeline &&) noexcept = delete;
    Pipeline(const Pipeline &) noexcept = delete;
    Pipeline &operator=(Pipeline &&) noexcept = delete;
    Pipeline &operator=(const Pipeline &) noexcept = delete;
    ~Pipeline() noexcept;

public:
    template<typename T>
    [[nodiscard]] auto register_bindless(BufferView<T> buffer) noexcept {
        _bindless_array.emplace_on_update(_bindless_buffer_count, buffer);
        return static_cast<uint>(_bindless_buffer_count++);
    }
    template<typename T>
    [[nodiscard]] auto register_bindless(const Buffer<T> &buffer) noexcept {
        return register_bindless(buffer.view());
    }
    template<typename T>
    [[nodiscard]] auto register_bindless(const Image<T> &image, TextureSampler sampler) noexcept {
        _bindless_array.emplace_on_update(_bindless_tex2d_count, image, sampler);
        return static_cast<uint>(_bindless_tex2d_count++);
    }
    template<typename T>
    [[nodiscard]] auto register_bindless(const Volume<T> &volume, TextureSampler sampler) noexcept {
        _bindless_array.emplace_on_update(_bindless_tex3d_count, volume, sampler);
        return static_cast<uint>(_bindless_tex3d_count++);
    }

    template<typename T>
    void update_bindless(BufferView<T> buffer, size_t buffer_id) noexcept {
        _bindless_array.emplace_on_update(buffer_id, buffer);
    }
    template<typename T>
    void update_bindless(const Buffer<T> &buffer, size_t buffer_id) noexcept {
        update_bindless(buffer.view(), buffer_id);
    }
    template<typename T>
    void update_bindless(const Image<T> &image, TextureSampler sampler, size_t buffer_id) noexcept {
        _bindless_array.emplace_on_update(buffer_id, image, sampler);
    }
    template<typename T>
    void update_bindless(const Volume<T> &volume, TextureSampler sampler, size_t buffer_id) noexcept {
        _bindless_array.emplace_on_update(buffer_id, volume, sampler);
    }

    void register_transform(Transform *transform) noexcept;
    [[nodiscard]] uint register_surface(CommandBuffer &command_buffer, const Surface *surface) noexcept;
    [[nodiscard]] uint register_light(CommandBuffer &command_buffer, const Light *light) noexcept;
    [[nodiscard]] uint register_medium(CommandBuffer &command_buffer, const Medium *medium) noexcept;
    [[nodiscard]] uint register_subsurface(CommandBuffer &command_buffer, const Subsurface *subsurface) noexcept;

    template<typename Create>
    uint register_named_id(luisa::string_view identifier, Create &&create_id) noexcept {
        if (auto it = _named_ids.find(identifier); it != _named_ids.end()) {
            return it->second;
        }
        auto new_id = std::invoke(std::forward<Create>(create_id));
        _named_ids.emplace(identifier, new_id);
        return new_id;
    }

    template<typename T, typename... Args>
        requires std::is_base_of_v<Resource, T>
    [[nodiscard]] auto create(Args &&...args) noexcept -> T * {
        auto resource = luisa::make_unique<T>(_device.create<T>(std::forward<Args>(args)...));
        auto p = resource.get();
        _resources.emplace_back(std::move(resource));
        return p;
    }

    template<uint dim, typename Def>
    void register_shader(luisa::string_view name, Def &&def) noexcept {
        static_assert(dim == 1u || dim == 2u || dim == 3u);
        register_named_id(name, [&] {
            auto shader = _device.compile<dim>(std::forward<Def>(def));
            auto resource = luisa::make_unique<decltype(shader)>(std::move(shader));
            auto index = static_cast<uint>(_resources.size());
            _resources.emplace_back(std::move(resource));
            return index;
        });
    }

    /* buffer view, resource id, bindless id */
    template<typename T>
    [[nodiscard]] std::pair<BufferView<T>, uint> bindless_buffer(size_t n) noexcept {
        auto buffer = create<Buffer<T>>(n);
        auto view = buffer->view();
        auto buffer_id = register_bindless(view);
        return std::make_pair(view, buffer_id);
    }

    [[nodiscard]] std::pair<BufferView<float4>, uint> allocate_constant_slot() noexcept;
    void update_bindless_if_dirty(CommandBuffer &command_buffer) noexcept;

public:
    // [[nodiscard]] static luisa::unique_ptr<Pipeline> create(Device &device, Stream &stream, Scene &scene) noexcept;
    [[nodiscard]] static luisa::unique_ptr<Pipeline> create(Device &device, Scene &scene) noexcept;
    void shutter_update(CommandBuffer &command_buffer, float time_offset) noexcept;
    void update(Stream &stream) noexcept;
    void render(Stream &stream) noexcept;
    void render_to_buffer(Stream &stream, Camera *camera, luisa::vector<float4> &buffer) noexcept;
    void set_time(float time) noexcept { _time = time; }

    [[nodiscard]] auto &device() const noexcept { return _device; }
    [[nodiscard]] auto &scene() const noexcept { return _scene; }
    [[nodiscard]] auto time() const noexcept { return _time; }
    [[nodiscard]] auto &bindless_array() noexcept { return _bindless_array; }
    [[nodiscard]] auto &bindless_array() const noexcept { return _bindless_array; }
    [[nodiscard]] auto &cameras() const noexcept { return _cameras; }
    [[nodiscard]] auto camera(const Camera *camera) const noexcept { return _cameras.find(camera)->second.get(); }
    [[nodiscard]] auto &surfaces() const noexcept { return _surfaces; }
    [[nodiscard]] auto &lights() const noexcept { return _lights; }
    [[nodiscard]] auto &media() const noexcept { return _media; }
    [[nodiscard]] auto &subsurfaces() const noexcept { return _subsurfaces; }
    [[nodiscard]] auto environment() const noexcept { return _environment.get(); }
    [[nodiscard]] auto environment_medium_tag() const noexcept { return _environment_medium_tag; }
    [[nodiscard]] auto integrator() const noexcept { return _integrator.get(); }
    [[nodiscard]] auto spectrum() const noexcept { return _spectrum.get(); }
    [[nodiscard]] auto geometry() const noexcept { return _geometry.get(); }
    [[nodiscard]] auto has_lighting() const noexcept { return !_lights.empty() || _environment != nullptr; }
    [[nodiscard]] const Texture::Instance *build_texture(CommandBuffer &command_buffer, const Texture *texture) noexcept;
    [[nodiscard]] const Filter::Instance *build_filter(CommandBuffer &command_buffer, const Filter *filter) noexcept;
    [[nodiscard]] const PhaseFunction::Instance *build_phasefunction(CommandBuffer &command_buffer, const PhaseFunction *phasefunction) noexcept;

    [[nodiscard]] uint named_id(luisa::string_view name) const noexcept;
    template<typename T, typename I>
    [[nodiscard]] auto buffer(I &&i) const noexcept { return _bindless_array->buffer<T>(std::forward<I>(i)); }
    template<typename I>
    [[nodiscard]] auto tex2d(I &&i) const noexcept { return _bindless_array->tex2d(std::forward<I>(i)); }
    template<typename I>
    [[nodiscard]] auto tex3d(I &&i) const noexcept { return _bindless_array->tex3d(std::forward<I>(i)); }
    template<typename T>
    [[nodiscard]] auto named_buffer(luisa::string_view name) const noexcept {
        return _bindless_array->buffer<T>(named_id(name));
    }
    [[nodiscard]] auto named_tex2d(luisa::string_view name) const noexcept {
        return _bindless_array->tex2d(named_id(name));
    }
    [[nodiscard]] auto named_tex3d(luisa::string_view name) const noexcept {
        return _bindless_array->tex3d(named_id(name));
    }
    [[nodiscard]] Float4x4 transform(Transform *transform) const noexcept;
    [[nodiscard]] Float4 constant(Expr<uint> index) const noexcept;

    template<uint dim, typename... Args, typename... CallArgs>
    [[nodiscard]] auto shader(luisa::string_view name, CallArgs &&...call_args) const noexcept {
        auto shader = dynamic_cast<const Shader<dim, Args...> *>(
            _resources[named_id(name)].get());
        return (*shader)(std::forward<CallArgs>(call_args)...);
    }
};

}// namespace luisa::render
