//
// Created by Mike on 2021/12/13.
//

#include <luisa/core/logging.h>
#include <base/scene_node.h>
#include <base/pipeline.h>

namespace luisa::render {

SceneNode::SceneNode(const Scene *scene, const SceneNodeDesc *desc, SceneNodeTag tag) noexcept:
    _scene{scene}, _tag{tag}, _dirty{true} {
    if (!desc->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Undefined scene description "
            "node '{}' (type = {}::{}).",
            desc->identifier(),
            scene_node_tag_description(desc->tag()),
            desc->impl_type());
    }
    if (!desc->is_internal() && desc->tag() != tag) [[unlikely]] {
        LUISA_ERROR(
            "Invalid tag {} of scene description "
            "node '{}' (expected {}). [{}]",
            scene_node_tag_description(desc->tag()),
            desc->identifier(),
            scene_node_tag_description(tag),
            desc->source_location().string());
    }
}

void SceneNode::update(Scene *scene, const SceneNodeDesc *desc) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

luisa::string SceneNode::info() const noexcept {
    return luisa::format("Node <{}, {}>", scene_node_tag_description(_tag), impl_type());
}

}   // luisa::render