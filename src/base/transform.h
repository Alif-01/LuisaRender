//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/stl.h>
#include <luisa/runtime/buffer.h>
#include <base/scene_node.h>

namespace luisa::render {

/* Keep the constructing methods (SRT or matrix) on  */
struct RawTransform {
    RawTransform() noexcept = default;
    RawTransform(PyFloatArr transform) noexcept: empty{false} {
        auto vec = pyarray_to_vector<float>(transform);
        for (auto row = 0u; row < 4u; ++row) {
            for (auto col = 0u; col < 4u; ++col) {
                transform[col][row] = vec[row * 4u + col];
            }
        }
    }
    RawTransform(float4x4 transform) noexcept: transform{transform}, empty{false} {}
    RawTransform(PyFloatArr translate, PyFloatArr rotate, PyFloatArr scale) noexcept:
        translate{pyarray_to_pack<float, 3>(translate)},
        rotate{pyarray_to_pack<float, 4>(rotate)},
        scale{pyarray_to_pack<float, 3>(scale)}, empty{false} {}
    RawTransform(float3 translate, float4 rotate, float3 scale) noexcept:
        translate{translate}, rotate{rotate}, scale{scale}, empty{false} {}
    [[nodiscard]] bool is_matrix() noexcept {
        return any(transform[0] != make_float4(1.0f, 0.0f, 0.0f, 0.0f)) ||
               any(transform[1] != make_float4(0.0f, 1.0f, 0.0f, 0.0f)) ||
               any(transform[2] != make_float4(0.0f, 0.0f, 1.0f, 0.0f)) ||
               any(transform[3] != make_float4(0.0f, 0.0f, 0.0f, 1.0f));
    }
    [[nodiscard]] bool is_srt() noexcept {
        return any(translate != make_float3(0.f)) ||
               any(rotate != make_float4(0.f)) ||
               any(scale != make_float3(1.f));
    }
    
    bool empty{true};
    float4x4 transform{make_float4x4(1.f)};
    float3 translate{make_float3(0.f)};
    float4 rotate{make_float4(0.f)};
    float3 scale{make_float3(1.f)};
};

class Transform : public SceneNode {
public:
    Transform(Scene *scene, const SceneNodeDesc *desc) noexcept;
    Transform(Scene *scene) noexcept;
    virtual void update_transform(Scene *scene, const RawTransform &trans) noexcept;
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

class InstancedTransform {

private:
    const TransformTree::Node *_node;
    size_t _instance_id;

public:
    InstancedTransform(const TransformTree::Node *node, size_t inst) noexcept
        : _node{node}, _instance_id{inst} {}
    [[nodiscard]] auto instance_id() const noexcept { return _instance_id; }
    [[nodiscard]] auto matrix(float time) const noexcept {
        return _node == nullptr ? make_float4x4(1.0f) : _node->matrix(time);
    }
};

}// namespace luisa::render
