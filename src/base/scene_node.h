//
// Created by Mike on 2021/12/13.
//

#pragma once

#include <cstddef>
#include <string_view>

#include <luisa/core/basic_types.h>
#include <luisa/dsl/syntax.h>
#include <sdl/scene_node_desc.h>

namespace luisa::compute {
class Device;
class Stream;
class CommandBuffer;
}// namespace luisa::compute

namespace luisa::render {

using compute::Device;
using compute::Stream;

using compute::Expr;
using compute::Float;
using compute::Float2;
using compute::Float3;
using compute::Float4;
using compute::Var;

class Scene;
class Pipeline;

class SceneNode {

public:
    class Instance {

    private:
        Pipeline &_pipeline;

    public:
        explicit Instance(Pipeline &pipeline) noexcept: _pipeline{pipeline} {}
        Instance(const Instance &) noexcept = delete;
        Instance(Instance &&another) noexcept = default;
        Instance &operator=(const Instance &) noexcept = delete;
        Instance &operator=(Instance &&another) noexcept = default;
        virtual ~Instance() noexcept = default;
        [[nodiscard]] auto &pipeline() noexcept { return _pipeline; }
        [[nodiscard]] const auto &pipeline() const noexcept { return _pipeline; }
    };

private:
    const Scene * _scene;
    SceneNodeTag _tag;
    bool _dirty;

public:
    template <typename T>
    [[nodiscard]] static bool update_value(T &value, T new_value) noexcept {
        if (value == new_value) return false;
        else {
            value = new_value;
            return true;
        }
    }

public:
    SceneNode(const Scene *scene, const SceneNodeDesc *desc, SceneNodeTag tag) noexcept;
    SceneNode(SceneNode &&) noexcept = delete;
    SceneNode(const SceneNode &) noexcept = delete;
    virtual void update(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::string info() const noexcept;
    SceneNode &operator=(SceneNode &&) noexcept = delete;
    SceneNode &operator=(const SceneNode &) noexcept = delete;
    virtual ~SceneNode() noexcept = default;
    [[nodiscard]] auto scene() const noexcept { return _scene; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto dirty() const noexcept { return _dirty; }
    void set_updated(bool updated) noexcept { _dirty |= updated; }
    void clear_dirty() noexcept { _dirty = false; }
    [[nodiscard]] virtual luisa::string_view impl_type() const noexcept = 0;
};

}// namespace luisa::render

#define LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(cls)                   \
    LUISA_EXPORT_API luisa::render::SceneNode *create(             \
        luisa::render::Scene *scene,                               \
        const luisa::render::SceneNodeDesc *desc) LUISA_NOEXCEPT { \
        return luisa::new_with_allocator<cls>(scene, desc);        \
    }                                                              \
    LUISA_EXPORT_API void destroy(                                 \
        luisa::render::SceneNode *node) LUISA_NOEXCEPT {           \
        luisa::delete_with_allocator(node);                        \
    }
