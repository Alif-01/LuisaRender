//
// Created by Mike on 2021/12/15.
//

#include <util/thread_pool.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

Pipeline::Pipeline(Device &device, Scene &scene) noexcept:
    _device{device},
    _scene{scene},
    _bindless_array{device.create_bindless_array(bindless_array_capacity)},
    _general_buffer_arena{luisa::make_unique<BufferArena>(device, 16_M)},
    _geometry{luisa::make_unique<Geometry>(*this)},
    _transform_matrices{transform_matrix_buffer_size},
    _transform_matrix_buffer{device.create_buffer<float4x4>(transform_matrix_buffer_size)},
    _time{0.f} {
}

Pipeline::~Pipeline() noexcept = default;

void Pipeline::update_bindless_if_dirty(CommandBuffer &command_buffer) noexcept {
    if (_bindless_array.dirty()) {
        command_buffer << _bindless_array.update();
    }
};

// TODO: We should split create into Build and Update, and no need to pass scene to pipeline build.
luisa::unique_ptr<Pipeline> Pipeline::create(Device &device, Scene &scene) noexcept {
    return luisa::make_unique<Pipeline>(device, scene);
}

void Pipeline::update(Stream &stream) noexcept {
    global_thread_pool().synchronize();
    CommandBuffer command_buffer{&stream};
    if (auto spectrum = _scene.spectrum(); spectrum->dirty()) {
        _spectrum = spectrum->build(*this, command_buffer);
        spectrum->clear_dirty();
    }

    // Camera
    for (auto camera : _scene.cameras()) {
        if (camera->dirty()) {
            _cameras[camera] = camera->build(*this, command_buffer);
            camera->clear_dirty();  // TODO: in Build
        }
    }
    update_bindless_if_dirty(command_buffer);

    // Geometry
    _geometry->update(command_buffer, _scene.shapes(), _time);
    update_bindless_if_dirty(command_buffer);

    // Environment
    bool environment_updated = false;
    if (auto env = _scene.environment(); env != nullptr && env->dirty()) {
        _environment = env->build(*this, command_buffer);
        env->clear_dirty();
        environment_updated = true;
        update_bindless_if_dirty(command_buffer);
    }
    if (auto env_medium = _scene.environment_medium(); env_medium != nullptr && env_medium->dirty()) {
        _environment_medium_tag = register_medium(command_buffer, env_medium);
    }

    // Integrator, may needs world min/max
    if (auto integrator = _scene.integrator(); integrator->dirty()) {
        _integrator = integrator->build(*this, command_buffer);
        integrator->clear_dirty();
        update_bindless_if_dirty(command_buffer);
    }

    // Transform
    bool transform_updated = false;
    for (auto &[transform, transform_id] : _transform_to_id) {
        if (transform->dirty()) {
            _transform_matrices[transform_id] = transform->matrix(_time);
            transform_updated = true;
            transform->clear_dirty();
        }
    }
    if (transform_updated || _transforms_dirty) {
        command_buffer << _transform_matrix_buffer
            .view(0u, _transform_to_id.size())
            .copy_from(_transform_matrices.data());
        _transforms_dirty = false;
    }
    
    update_bindless_if_dirty(command_buffer);
    command_buffer << compute::commit();

    LUISA_INFO(
        "Resource use: Buffer={}, Texture2D={}, Texture3D={}, Constant={}, Resources={}.",
        _bindless_buffer_count, _bindless_tex2d_count,
        _bindless_tex3d_count, _constant_count, _resources.size()
    );

}

void Pipeline::shutter_update(CommandBuffer &command_buffer, float time_offset) noexcept {
    // TODO: support deformable meshes
    _geometry->shutter_update(command_buffer, _time + time_offset);
    if (_any_dynamic_transforms) {
        for (auto &[transform, transform_id] : _transform_to_id) {
            if (!transform->is_static()) {
                _transform_matrices[transform_id] = transform->matrix(_time + time_offset);
            }
        }
        command_buffer << _transform_matrix_buffer
                          .view(0u, _transform_to_id.size())
                          .copy_from(_transform_matrices.data());
    }
}

uint Pipeline::register_surface(CommandBuffer &command_buffer, const Surface *surface) noexcept {
    if (auto iter = _surface_tags.find(surface);
        iter != _surface_tags.end()) { return iter->second; }
    auto tag = _surfaces.emplace(surface->build(*this, command_buffer));
    _surface_tags.emplace(surface, tag);
    return tag;
}

uint Pipeline::register_light(CommandBuffer &command_buffer, const Light *light) noexcept {
    if (auto iter = _light_tags.find(light);
        iter != _light_tags.end()) { return iter->second; }
    auto tag = _lights.emplace(light->build(*this, command_buffer));
    _light_tags.emplace(light, tag);
    return tag;
}

uint Pipeline::register_medium(CommandBuffer &command_buffer, const Medium *medium) noexcept {
    if (auto iter = _medium_tags.find(medium);
        iter != _medium_tags.end()) { return iter->second; }
    auto tag = _media.emplace(medium->build(*this, command_buffer));
    _medium_tags.emplace(medium, tag);
    return tag;
}

uint Pipeline::register_subsurface(CommandBuffer &command_buffer, const Subsurface *subsurface) noexcept {
    if (auto iter = _subsurface_tags.find(subsurface);
        iter != _subsurface_tags.end()) { return iter->second; }
    auto tag = _subsurfaces.emplace(subsurface->build(*this, command_buffer));
    _subsurface_tags.emplace(subsurface, tag);
    return tag;
}

void Pipeline::register_transform(Transform *transform) noexcept {
    if (transform == nullptr) { return; }
    if (!_transform_to_id.contains(transform)) {
        auto transform_id = static_cast<uint>(_transform_to_id.size());
        LUISA_ASSERT(transform_id < transform_matrix_buffer_size, "Transform matrix buffer overflows.");
        _transform_to_id.emplace(transform, transform_id);
        _transforms_dirty = true;
        if (!transform->is_static()) _any_dynamic_transforms = true;
    }
}

void Pipeline::render(Stream &stream) noexcept {
    _integrator->render(stream);
}

void Pipeline::render_to_buffer(Stream &stream, Camera *camera, luisa::vector<float4> &buffer) noexcept {
    _integrator->render_to_buffer(stream, camera, buffer);
}

const Texture::Instance *Pipeline::build_texture(CommandBuffer &command_buffer, const Texture *texture) noexcept {
    if (texture == nullptr) { return nullptr; }
    if (auto iter = _textures.find(texture); iter != _textures.end()) {
        return iter->second.get();
    }
    auto t = texture->build(*this, command_buffer);
    return _textures.emplace(texture, std::move(t)).first->second.get();
}

const Filter::Instance *Pipeline::build_filter(CommandBuffer &command_buffer, const Filter *filter) noexcept {
    if (filter == nullptr) { return nullptr; }
    if (auto iter = _filters.find(filter); iter != _filters.end()) {
        return iter->second.get();
    }
    auto f = filter->build(*this, command_buffer);
    return _filters.emplace(filter, std::move(f)).first->second.get();
}

const PhaseFunction::Instance *Pipeline::build_phasefunction(CommandBuffer &command_buffer, const PhaseFunction *phasefunction) noexcept {
    if (phasefunction == nullptr) { return nullptr; }
    if (auto iter = _phasefunctions.find(phasefunction); iter != _phasefunctions.end()) {
        return iter->second.get();
    }
    auto pf = phasefunction->build(*this, command_buffer);
    return _phasefunctions.emplace(phasefunction, std::move(pf)).first->second.get();
}

Float4x4 Pipeline::transform(Transform *transform) const noexcept {
    if (transform == nullptr) { return make_float4x4(1.f); }
    if (transform->is_identity()) { return make_float4x4(1.f); }
    auto iter = _transform_to_id.find(transform);
    LUISA_ASSERT(iter != _transform_to_id.cend(), "Transform is not registered.");
    return _transform_matrix_buffer->read(iter->second);
}

uint Pipeline::named_id(luisa::string_view name) const noexcept {
    auto iter = _named_ids.find(name);
    LUISA_ASSERT(iter != _named_ids.cend(), "Named ID '{}' not found.", name);
    return iter->second;
}

std::pair<BufferView<float4>, uint> Pipeline::allocate_constant_slot() noexcept {
    if (!_constant_buffer) {
        _constant_buffer = device().create_buffer<float4>(constant_buffer_size);
    }
    auto slot = _constant_count++;
    LUISA_ASSERT(slot < constant_buffer_size, "Constant buffer overflows.");
    return {_constant_buffer.view(static_cast<uint>(slot), 1u),
            static_cast<uint>(slot)};
}

Float4 Pipeline::constant(Expr<uint> index) const noexcept {
    return _constant_buffer->read(index);
}

}// namespace luisa::render
