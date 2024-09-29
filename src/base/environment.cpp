//
// Created by Mike Smith on 2022/1/14.
//

#include <base/environment.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

Environment::Environment(Scene *scene, const SceneNodeDesc *desc) noexcept:
    SceneNode{scene, desc, SceneNodeTag::ENVIRONMENT},
    _transform{scene->load_transform(desc->property_node_or_default("transform"))} {}

luisa::string Environment::info() const noexcept {
    return luisa::format("{} transform=[{}]", SceneNode::info(),
        _transform ? _transform->info() : "");
}

void Environment::update(Scene *scene, const SceneNodeDesc *desc) noexcept {
    set_updated(
        update_value(_transform, scene->load_transform(desc->property_node_or_default("transform")))
    );
}

Environment::Instance::Instance(Pipeline &pipeline, const Environment *env) noexcept:
    SceneNode::Instance{pipeline}, _env{env} {
    pipeline.register_transform(env->transform());
}

Float3x3 Environment::Instance::transform_to_world() const noexcept {
    return make_float3x3(pipeline().transform(node()->transform()));
}

}// namespace luisa::render
