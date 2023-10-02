#include <future>

#include <base/shape.h>
#include <shapes/plane_base.h>

namespace luisa::render {

class Plane : public Shape {

private:
    std::shared_future<PlaneGeometry> _geometry;

public:
    Plane(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc},
          _geometry{PlaneGeometry::create(
              std::min(desc->property_uint_or_default("subdivision", 0u),
                       plane_max_subdivision_level))} {}
    
    Plane(Scene *scene, const RawShapeInfo &shape_info) noexcept
        : Shape{scene, shape_info} {

        if (shape_info.plane_info == nullptr) [[unlikely]]
            LUISA_ERROR_WITH_LOCATION("Invalid plane info!");
        auto plane_info = shape_info.plane_info.get();
        _geometry = PlaneGeometry::create(plane_info->subdivision);
        _geometry.wait();
    }

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    // [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] MeshView mesh() const noexcept override { return _geometry.get().mesh(); }
    [[nodiscard]] uint vertex_properties() const noexcept override {
        return Shape::property_flag_has_vertex_normal |
               Shape::property_flag_has_vertex_uv;
    }
};

using PlaneWrapper = VisibilityShapeWrapper<ShadingShapeWrapper<Plane>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PlaneWrapper)

LUISA_EXPORT_API luisa::render::SceneNode *create_raw(
    luisa::render::Scene *scene,
    const luisa::render::RawShapeInfo &shape_info) LUISA_NOEXCEPT {
    return luisa::new_with_allocator<luisa::render::PlaneWrapper>(scene, shape_info);
}