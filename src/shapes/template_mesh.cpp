//
// Modified from mesh.cpp
//

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/mesh.h>
#include <assimp/scene.h>
#include <assimp/Subdivision.h>

#include <luisa/core/clock.h>
#include <util/thread_pool.h>
#include <base/shape.h>

namespace luisa::render {

class TemplateMesh : public Shape {

private:
    luisa::string _template_id;

public:
    TemplateMesh(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc} {
        _template_id = desc->property_string_or_default("template_id", "");
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] bool is_template_mesh() const noexcept override { return true; }
    [[nodiscard]] luisa::string template_id() const noexcept override { return _template_id; }
};

using TemplateMeshWrapper =
    VisibilityShapeWrapper<
        ShadowTerminatorShapeWrapper<
            IntersectionOffsetShapeWrapper<TemplateMesh>>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::TemplateMeshWrapper)
