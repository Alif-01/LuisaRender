//
// Created by Mike Smith on 2022/11/8.
//

#include <future>

#include <base/shape.h>
#include <util/spheres_base.h>

namespace luisa::render {

class SphereGroup : public Shape {

private:
    std::shared_future<SpheresProceduralGeometry> _geometry;
    uint _subdiv;

public:
    SphereGroup(Scene *scene, const SceneNodeDesc *desc) noexcept :
        Shape{scene, desc},
        _subdiv{desc->property_uint_or_default("subdivision", 0u)} { 
        _geometry = SpheresProceduralGeometry::create(
            desc->property_float_list("centers"),
            desc->property_float_list("radii")
        );
    }

    void update(Scene *scene, const SceneNodeDesc *desc) noexcept override {
        _geometry = SpheresProceduralGeometry::create(
            desc->property_float_list("centers"),
            desc->property_float_list("radii")
        );
        set_updated(true);
    }

    [[nodiscard]] luisa::string info() const noexcept override {
        return luisa::format("{} geometry=[{}]", Shape::info(), _geometry.get().info());
    }

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_spheres() const noexcept override { return true; }
    [[nodiscard]] bool empty() const noexcept override { 
        return _geometry.get().aabbs().empty();
    }
    [[nodiscard]] SpheresView spheres() const noexcept override {
        return { _geometry.get().aabbs() };
    }
    [[nodiscard]] uint vertex_properties() const noexcept override { 
        return Shape::property_flag_has_vertex_normal |
               Shape::property_flag_has_vertex_uv;
    }
};

using SphereGroupWrapper = VisibilityShapeWrapper<SphereGroup>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SphereGroupWrapper)