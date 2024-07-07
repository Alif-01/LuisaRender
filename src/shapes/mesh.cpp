//
// Created by Mike on 2022/1/7.
//

#include <base/shape.h>
#include <util/mesh_base.h>

namespace luisa::render {

class Mesh : public Shape {

private:
    std::shared_future<MeshGeometry> _geometry;

public:
    Mesh(Scene *scene, const SceneNodeDesc *desc) noexcept:
        Shape{scene, desc} {

        if (desc->property_string_or_default("file").empty()) {
            // auto positions = desc->property_float_list("positions");
            // auto indices = desc->property_uint_list("indices");
            // auto normals = desc->property_float_list_or_default("normals");
            // auto uvs = desc->property_float_list_or_default("uvs");
            // _geometry = MeshGeometry::create(positions, indices, normals, uvs);
            _geometry = MeshGeometry::create(
                desc->property_float_list("positions"),
                desc->property_uint_list("indices"),
                desc->property_float_list_or_default("normals"),
                desc->property_float_list_or_default("uvs"));
            // _geometry.wait();
        } else {
            _geometry = MeshGeometry::create(
                desc->property_path("file"),
                desc->property_uint_or_default("subdivision", 0u),
                desc->property_bool_or_default("flip_uv", false),
                desc->property_bool_or_default("drop_normal", false),
                desc->property_bool_or_default("drop_uv", false));
            // _geometry.wait();
        }
    }

    Mesh(Scene *scene, const RawShapeInfo &shape_info) noexcept:
        Shape{scene, shape_info} {
        LUISA_ASSERT(shape_info.get_type() == "mesh", "Invalid rigid info.");

        if (shape_info.file_info != nullptr) {
            auto file_info = shape_info.file_info.get();
            _geometry = MeshGeometry::create(file_info->file, 0u, false, false, false);
        } else if (shape_info.mesh_info != nullptr) {
            auto mesh_info = shape_info.mesh_info.get();
            _geometry = MeshGeometry::create(
                mesh_info->vertices, 
                mesh_info->triangles,
                mesh_info->normals,
                mesh_info->uvs
            );
            _geometry.wait();
        } else {
            LUISA_ERROR_WITH_LOCATION("Invalid rigid info!");
        }
    }

    [[nodiscard]] bool update(Scene *scene, const SceneNodeDesc *desc) noexcept override {
        return Shape::update(scene, desc);
    }

    void update_shape(Scene *scene, const RawShapeInfo &shape_info) noexcept override {
        Shape::update_shape(scene, shape_info);
    }
    
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] bool empty() const noexcept override { 
        const MeshGeometry &g = _geometry.get();
        return g.vertices().empty() || g.triangles().empty();
    }
    [[nodiscard]] MeshView mesh() const noexcept override {
        const MeshGeometry &g = _geometry.get();
        return { g.vertices(), g.triangles() }; 
    }
    [[nodiscard]] uint vertex_properties() const noexcept override {
        const MeshGeometry &g = _geometry.get();
        return (g.has_normal() ? Shape::property_flag_has_vertex_normal : 0u) | 
               (g.has_uv() ? Shape::property_flag_has_vertex_uv : 0u);
    }
};

using MeshWrapper = VisibilityShapeWrapper<ShadingShapeWrapper<Mesh>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MeshWrapper)

LUISA_EXPORT_API luisa::render::SceneNode *create_raw(
    luisa::render::Scene *scene,
    const luisa::render::RawShapeInfo &shape_info) LUISA_NOEXCEPT {
    return luisa::new_with_allocator<luisa::render::MeshWrapper>(scene, shape_info);
}