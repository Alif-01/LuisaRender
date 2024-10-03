//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/stl.h>
#include <luisa/runtime/buffer.h>
#include <base/scene_node.h>

namespace luisa::render {

class Transform : public SceneNode {
public:
    Transform(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual luisa::string info() const noexcept override;
    [[nodiscard]] virtual bool is_static() const noexcept = 0;
    [[nodiscard]] virtual bool is_identity() const noexcept = 0;
    [[nodiscard]] virtual float4x4 matrix(float time) const noexcept = 0;
};

class TransformTree {

public:
    class Node {

    private:
        const Node *_parent;
        const Transform *_transform;

    public:
        Node(const Node *parent, const Transform *t) noexcept;
        [[nodiscard]] auto transform() const noexcept { return _transform; }
        [[nodiscard]] float4x4 matrix(float time) const noexcept;
    };

private:
    luisa::vector<luisa::unique_ptr<Node>> _nodes;
    luisa::vector<const Node *> _node_stack;
    luisa::vector<bool> _static_stack;

public:
    TransformTree() noexcept;
    [[nodiscard]] auto size() const noexcept { return _nodes.size(); }
    [[nodiscard]] auto empty() const noexcept { return _nodes.empty(); }
    void push(const Transform *t) noexcept;
    void pop(const Transform *t) noexcept;
    [[nodiscard]] std::pair<const Node *, bool /* is_static */> leaf(const Transform *t) noexcept;
};

}// namespace luisa::render
