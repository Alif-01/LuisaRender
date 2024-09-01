//
// Created by Mike on 2021/12/13.
//

#include <luisa/core/logging.h>
#include <sdl/scene_desc.h>

namespace luisa::render {

SceneNodeDesc *SceneDesc::node(luisa::string_view identifier) const noexcept {
    if (auto iter = _global_nodes.find(identifier);
        iter != _global_nodes.cend()) {
        return iter->get();
    }
    LUISA_ERROR_WITH_LOCATION("Global node '{}' not found in scene description.", identifier);
}

const SceneNodeDesc *SceneDesc::reference(luisa::string_view identifier) noexcept {
    if (identifier == root_node_identifier) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid reference to root node.");
    }
    std::scoped_lock lock{_mutex};
    auto [iter, _] = _global_nodes.emplace(
        lazy_construct([identifier] {
            return luisa::make_unique<SceneNodeDesc>(
                luisa::string{identifier}, SceneNodeTag::DECLARATION);
        }));
    // auto [iter, _] = _global_nodes.emplace(
    //     luisa::make_unique<SceneNodeDesc>(luisa::string{identifier}, SceneNodeTag::DECLARATION)
    // );
    return iter->get();
}

SceneNodeDesc *SceneDesc::define(
    luisa::unique_ptr<SceneNodeDesc> node_ptr, luisa::string_view impl_type) noexcept {
    if (node_ptr->tag() == SceneNodeTag::INTERNAL ||
        node_ptr->tag() == SceneNodeTag::DECLARATION) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Defining internal or declaration node "
            "as a global node is not allowed.");
    }

    if (node_ptr->is_defined()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Incoming node '{}' ({}::{}) has been defined.",
            node_ptr->identifier(), scene_node_tag_description(node_ptr->tag()), impl_type
        );
    }

    std::scoped_lock lock{_mutex};
    auto [iter, _] = _global_nodes.emplace(std::move(node_ptr));
    auto node = iter->get();
    if (node->is_defined()) {
        if (node->tag() != node_ptr->tag() || node->impl_type() != impl_type) [[unlikely]]
            LUISA_ERROR_WITH_LOCATION(
                "A different node '{}' ({}::{}) has been defined in scene description. "
                "Different from node ({}::{})",
                node->identifier(), scene_node_tag_description(node->tag()), node->impl_type(),
                scene_node_tag_description(node_ptr->tag()), impl_type
            );
        LUISA_INFO_WITH_LOCATION(
            "Update scene node description: '{}' ({}::{})",
            node->identifier(), scene_node_tag_description(node->tag()), impl_type
        );
        node->update_properties(node_ptr.get());
    } else {
        node->define(node->tag(), impl_type);
    }

    if (node->tag() == SceneNodeTag::ROOT) {
        if (_root) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Redefinition of root node in scene description.");
        }
        _root = node;
    }

    return node;
}

SceneNodeDesc *SceneDesc::define(
    luisa::string_view identifier, SceneNodeTag tag, luisa::string_view impl_type,
    SceneNodeDesc::SourceLocation location, const SceneNodeDesc *base) noexcept {

    if (tag == SceneNodeTag::INTERNAL ||
        tag == SceneNodeTag::DECLARATION) [[unlikely]] {
        LUISA_ERROR(
            "Defining internal or declaration node "
            "as a global node is not allowed. [{}]",
            location.string());
    }

    std::scoped_lock lock{_mutex};
    auto [iter, _] = _global_nodes.emplace(
        lazy_construct([identifier, tag] {
            return luisa::make_unique<SceneNodeDesc>(
                luisa::string{identifier}, tag);
        }));
    auto node = iter->get();
    if (node->is_defined()) [[unlikely]] {
        LUISA_ERROR(
            "Redefinition of node '{}' ({}::{}) "
            "in scene description. [{}]",
            node->identifier(),
            scene_node_tag_description(node->tag()),
            node->impl_type(),
            location.string());
    }
    node->define(tag, impl_type, location, base);

    if (tag == SceneNodeTag::ROOT) {
        if (_root) [[unlikely]] {
            LUISA_ERROR(
                "Redefinition of root node "
                "in scene description. [{}]",
                location.string());
        }
        _root = node;
    }

    return node;
}

const std::filesystem::path *SceneDesc::register_path(std::filesystem::path path) noexcept {
    auto p = luisa::make_unique<std::filesystem::path>(std::move(path));
    std::scoped_lock lock{_mutex};
    return _paths.emplace_back(std::move(p)).get();
}

}// namespace luisa::render
