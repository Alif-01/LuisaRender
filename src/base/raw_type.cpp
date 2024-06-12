#include <luisa/dsl/syntax.h>
#include <base/raw_type.h>

namespace luisa::render {

using namespace omit;

void RawTextureInfo::build_constant(FloatArr constant) noexcept {
    constant_info = luisa::make_unique<RawConstantInfo>(std::move(constant));
}
void RawTextureInfo::build_image(
    StringArr image, FloatArr scale,
    FloatArr image_data, uint2 resolution, uint channel
) noexcept {
    image_info = luisa::make_unique<RawImageInfo>(
        std::move(image), std::move(scale),
        std::move(image_data), std::move(resolution), channel
    );
}

void RawTextureInfo::build_checker(RawTextureInfo on, RawTextureInfo off, float scale) noexcept {
    checker_info = luisa::make_unique<RawCheckerInfo>(std::move(on), std::move(off), scale);
}

StringArr RawShapeInfo::get_type_info() const noexcept {
    return spheres_info != nullptr ? spheres_info->get_info() :
           mesh_info != nullptr ? mesh_info->get_info() : 
           file_info != nullptr ? file_info->get_info() :
           plane_info != nullptr ? plane_info->get_info() : "";
}

StringArr RawShapeInfo::get_type() const noexcept {
    return spheres_info != nullptr ? "spheregroup" :
           mesh_info != nullptr ? mesh_info->get_type() : 
           file_info != nullptr ? "mesh" :
           plane_info != nullptr ? "plane" : "None";
}

StringArr RawCameraInfo::get_type_info() const noexcept {
    return pinhole_info != nullptr ? pinhole_info->get_info() :
           thinlens_info != nullptr ? thinlens_info->get_info() : "";
}

} // namespace luisa::render