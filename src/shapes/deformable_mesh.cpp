//
// Created by Mike on 2022/2/18.
//

#include <base/shape.h>
#include <shapes/mesh_base.h>

namespace luisa::render {

class DeformableMesh : public Shape {

private:
    MeshLoader _loader;

public:
    DeformableMesh(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc} {

        auto triangles = desc->property_uint_list("indices");
        auto positions = desc->property_float_list("positions");
        auto normals = desc->property_float_list_or_default("normals");
        auto uvs = desc->property_float_list_or_default("uvs");

        _loader.build_mesh(positions, triangles, normals, uvs);
    }

    DeformableMesh(Scene *scene, const RawShapeInfo &shape_info) noexcept
        : Shape{scene, shape_info} {

        if (shape_info.get_type() != "deformablemesh") [[unlikely]]
            LUISA_ERROR_WITH_LOCATION("Invalid deformable info!");
        auto mesh_info = shape_info.mesh_info.get();

        _loader.build_mesh(
            mesh_info->vertices, 
            mesh_info->triangles,
            mesh_info->normals,
            mesh_info->uvs
        );
    }

    void update_shape(Scene *scene, const RawShapeInfo &shape_info) noexcept override {
        Shape::update_shape(scene, shape_info);

        if (shape_info.get_type() != "deformablemesh") [[unlikely]]
            LUISA_ERROR_WITH_LOCATION("Invalid deformable info!");
        auto mesh_info = shape_info.mesh_info.get();

        _loader.build_mesh(
            mesh_info->vertices,
            mesh_info->triangles,
            mesh_info->normals,
            mesh_info->uvs
        );
    }

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool deformable() const noexcept override { return true; }
    [[nodiscard]] MeshView mesh() const noexcept override { return _loader.mesh(); }
    [[nodiscard]] uint vertex_properties() const noexcept override { return _loader.properties(); }

};

using DeformableMeshWrapper = VisibilityShapeWrapper<ShadingShapeWrapper<DeformableMesh>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DeformableMeshWrapper)

LUISA_EXPORT_API luisa::render::SceneNode *create_raw(
    luisa::render::Scene *scene,
    const luisa::render::RawShapeInfo &shape_info) LUISA_NOEXCEPT {
    return luisa::new_with_allocator<luisa::render::DeformableMeshWrapper>(scene, shape_info);
}