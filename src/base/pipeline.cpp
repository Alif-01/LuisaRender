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
    _transform_matrices{transform_matrix_buffer_size},
    _transform_matrix_buffer{device.create_buffer<float4x4>(transform_matrix_buffer_size)},
    _time{0.f} {

    // for (auto c : scene.cameras()) {
    //     if (c->shutter_span().x < pipeline->_initial_time) {
    //         pipeline->_initial_time = c->shutter_span().x;
    //     }
    // }
}

Pipeline::~Pipeline() noexcept = default;

void Pipeline::update_bindless_if_dirty(CommandBuffer &command_buffer) noexcept {
    if (_bindless_array.dirty()) {
        command_buffer << _bindless_array.update();
    }
};

// TODO: We should split create into Build and Update, and no need to pass scene to pipeline build. 
luisa::unique_ptr<Pipeline> Pipeline::create(Device &device, Stream &stream, Scene &scene) noexcept {
    global_thread_pool().synchronize();
    return luisa::make_unique<Pipeline>(device, scene);

    // CommandBuffer command_buffer{&stream};
    // pipeline->_spectrum = scene.spectrum()->build(*pipeline, command_buffer);
    // for (auto camera : scene.cameras()) {
    //     if (camera->dirty()) {
    //         pipeline->_cameras.emplace(camera, camera->build(*pipeline, command_buffer));
    //         camera->clear_dirty();  // TODO: in Build
    //     }
    // }
    // update_bindless_if_dirty();

    // pipeline->_geometry = luisa::make_unique<Geometry>(*pipeline);
    // pipeline->_geometry->build(command_buffer, scene.shapes(), pipeline->_initial_time);
    // update_bindless_if_dirty();

    // if (auto env = scene.environment(); env != nullptr && env->dirty()) {
    //     pipeline->_environment = env->build(*pipeline, command_buffer);
    //     env->clear_dirty();
    // }
    // if (auto environment_medium = scene.environment_medium(); environment_medium != nullptr) {
    //     pipeline->_environment_medium_tag = pipeline->register_medium(command_buffer, environment_medium);
    // }
    // update_bindless_if_dirty();

    // integrator may needs world min/max
    // pipeline->_integrator = scene.integrator()->build(*pipeline, command_buffer);

    // bool transform_updated = false;
    // for (auto &transform_id : pipeline->_transform_to_id) {
    //     auto &transform = transform_id.first;
    //     if (transform.dirty()) {
    //         pipeline->_transform_matrices[transform_id.second] = transform->matrix(pipeline->_initial_time);
    //         transform_updated = true;
    //         transform.clear_dirty();
    //     }
    // }
    // if (transform_updated || pipeline->_transforms_dirty) {
    //     command_buffer << pipeline->_transform_matrix_buffer
    //         .view(0u, pipeline->_transforms_to_id.size())
    //         .copy_from(pipeline->_transform_matrices.data());
    //     pipeline->_transforms_dirty = false;
    // }

    // update_bindless_if_dirty();
    // command_buffer << compute::commit();
    
    // LUISA_INFO("Created pipeline with {} camera(s), {} shape instance(s), "
    //            "{} surface instance(s), and {} light instance(s).",
    //            pipeline->_cameras.size(),
    //            pipeline->_geometry->instances().size(),
    //            pipeline->_surfaces.size(),
    //            pipeline->_lights.size());
    // return pipeline;
}

// bool Pipeline::update(CommandBuffer &command_buffer, float time) noexcept {
void Pipeline::update(
    // Stream &stream, Scene &scene, float time
    CommandBuffer &command_buffer, float time_offset
) noexcept {
    float time = _time + time_offset;

    if (auto spectrum = _scene.spectrum(); spectrum->dirty()) {
        _spectrum = spectrum->build(*this, command_buffer);
        spectrum->clear_dirty();
    }

    for (auto camera : _scene.cameras()) {
        if (camera->dirty()) {
            _cameras[camera] = camera->build(*this, command_buffer);
            camera->clear_dirty();  // TODO: in Build
        }
    }
    update_bindless_if_dirty();

    // if (_scene.cameras_updated()) {
    //     _cameras.clear();
    //     _cameras.reserve(_scene.cameras().size());

    //     for (auto camera : _scene.cameras()) {
    //         _cameras.emplace_back(camera->build(*this, command_buffer));
    //     }
    //     update_bindless_if_dirty();
    // }

    _geometry = luisa::make_unique<Geometry>(*this);
    _geometry->build(command_buffer, _scene.shapes(), time);
    update_bindless_if_dirty();

    bool environment_updated = false;
    if (auto env = _scene.environment(); env != nullptr && env->dirty()) {
        _environment = env->build(*this, command_buffer);
        env->clear_dirty();
        environment_updated = true;
        update_bindless_if_dirty();
    }
    if (auto env_medium = _scene.environment_medium(); env_medium != nullptr && env_medium->dirty()) {
        _environment_medium_tag = register_medium(command_buffer, env_medium);
    }

    // TODO: integrator's light sampler reads lights and env, maybe it should be managed like transform.
    if (environment_updated || _lights_dirty) { 
        _integrator = _scene.integrator()->build(*this, command_buffer);
        _lights_dirty = false;
        update_bindless_if_dirty();
    }

    bool transform_updated = false;
    for (auto &transform_id : _transform_to_id) {
        auto &transform = transform_id.first;
        if (transform.dirty()) {
            _transform_matrices[transform_id.second] = transform->matrix(time);
            transform_updated = true;
            transform.clear_dirty();
        }
        
    }
    if (transform_updated || _transforms_dirty) {
        command_buffer << _transform_matrix_buffer
            .view(0u, _transforms_to_id.size())
            .copy_from(_transform_matrices.data());
        _transforms_dirty = false;
    }
    
    update_bindless_if_dirty();
    command_buffer << compute::commit();
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
    _lights_dirty = true;
    return tag;
}

uint Pipeline::register_medium(CommandBuffer &command_buffer, const Medium *medium) noexcept {
    if (auto iter = _medium_tags.find(medium);
        iter != _medium_tags.end()) { return iter->second; }
    auto tag = _media.emplace(medium->build(*this, command_buffer));
    _medium_tags.emplace(medium, tag);
    return tag;
}


void Pipeline::register_transform(Transform *transform) noexcept {
    if (transform == nullptr) { return; }
    if (!_transform_to_id.contains(transform)) {
        // transform->set_registered();
        auto transform_id = static_cast<uint>(_transforms.size());
        LUISA_ASSERT(transform_id < transform_matrix_buffer_size, "Transform matrix buffer overflows.");
        _transform_to_id.emplace(transform, transform_id);
        // _transforms.emplace_back(transform);
        _transforms_dirty = true;
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


// bool Pipeline::update(CommandBuffer &command_buffer, float time) noexcept {
//     // TODO: support deformable meshes
//     auto updated = _geometry->update(command_buffer, time);
//     return updated;
//     // if (_any_dynamic_transforms) {
//     //     updated = true;
//     //     for (auto i = 0u; i < _transforms.size(); ++i) {
//     //         _transform_matrices[i] = _transforms[i]->matrix(time);
//     //     }
//     //     command_buffer << _transform_matrix_buffer
//     //                       .view(0u, _transforms.size())
//     //                       .copy_from(_transform_matrices.data());
//     // }
// }

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
