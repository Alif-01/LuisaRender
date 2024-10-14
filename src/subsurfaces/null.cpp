//
// Created by Mike Smith on 2022/1/12.
//

#include <base/subsurface.h>

namespace luisa::render {

struct NullSubsurface final : public Subsurface {

public:
    NullSubsurface(Scene *scene, const SceneNodeDesc *desc) noexcept : Subsurface{scene, desc} {}
    [[nodiscard]] bool is_null() const noexcept override { return true; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }

private:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        LUISA_ERROR_WITH_LOCATION("NullSubsurface cannot be instantiated.");
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NullSubsurface)