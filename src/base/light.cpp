//
// Created by Mike on 2021/12/15.
//

#include <base/light.h>

namespace luisa::render {

Light::Light(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::LIGHT} {}

Light::Light(Scene *scene) noexcept
    : SceneNode{scene, SceneNodeTag::LIGHT} {}

}// namespace luisa::render
