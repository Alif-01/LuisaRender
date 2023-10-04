//
// Created by Mike on 2022/1/7.
//

#include <base/shape.h>
#include <shapes/mesh_base.h>

namespace luisa::render {

class Mesh : public Shape {

private:
    std::shared_future<MeshLoader> _loader;

public:
    Mesh(Scene *scene, const SceneNodeDesc *desc) noexcept :
        Shape{scene, desc},
        _loader{MeshLoader::load(desc->property_path("file"),
                                 desc->property_uint_or_default("subdivision", 0u),
                                 desc->property_bool_or_default("flip_uv", false),
                                 desc->property_bool_or_default("drop_normal", false),
                                 desc->property_bool_or_default("drop_uv", false))} {
        _loader.wait();
    }

    Mesh(Scene *scene, const RawShapeInfo &shape_info) noexcept :
        Shape{scene, shape_info} {

        if (shape_info.type() != "mesh") [[unlikely]]
            LUISA_ERROR_WITH_LOCATION("Invalid rigid info!");

        if (shape_info.file_info != nullptr) {
            auto file_info = shape_info.file_info.get();
            _loader = MeshLoader::load(file_info->file, 0u, false, false, false);
            _loader.wait();
        } else if (shape_info.mesh_info != nullptr) {
            auto mesh_info = shape_info.mesh_info.get();

            _loader = MeshLoader::load(
                mesh_info->vertices, 
                mesh_info->triangles,
                mesh_info->normals,
                mesh_info->uvs
            );
            _loader.wait();
        } else {
            LUISA_ERROR_WITH_LOCATION("Invalid rigid info!");
        }
    }

    void update_shape(Scene *scene, const RawShapeInfo& shape_info) noexcept override {
        Shape::update_shape(scene, shape_info);
    }
    
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] MeshView mesh() const noexcept override { return _loader.get().mesh(); }
    [[nodiscard]] uint vertex_properties() const noexcept override { return _loader.get().properties(); }
};

using MeshWrapper = VisibilityShapeWrapper<ShadingShapeWrapper<Mesh>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MeshWrapper)

LUISA_EXPORT_API luisa::render::SceneNode *create_raw(
    luisa::render::Scene *scene,
    const luisa::render::RawShapeInfo &shape_info) LUISA_NOEXCEPT {
    return luisa::new_with_allocator<luisa::render::MeshWrapper>(scene, shape_info);
}