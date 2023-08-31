#include <luisa/dsl/syntax.h>
#include <base/raw_type.h>

namespace luisa::render {

using namespace omit;
// RawTextureInfo::RawTextureInfo() noexcept {}

void RawTextureInfo::build_constant(FloatArr constant) noexcept {
    // constant_info = get_ptr<RawConstantInfo>(std::move(constant));
    constant_info = luisa::make_unique<RawConstantInfo>(std::move(constant));
}
void RawTextureInfo::build_image(StringArr image, float3 scale) noexcept {
    // image_info = get_ptr<RawImageInfo>(std::move(image), std::move(scale));
    image_info = luisa::make_unique<RawImageInfo>(std::move(image), std::move(scale));
}
void RawTextureInfo::build_checker(RawTextureInfo on, RawTextureInfo off, float scale) noexcept {
    // checker_info = get_ptr<RawCheckerInfo>(std::move(on), std::move(off), scale);
    checker_info = luisa::make_unique<RawCheckerInfo>(std::move(on), std::move(off), scale);
}

StringArr RawShapeInfo::get_type_info() const noexcept {
    return spheres_info != nullptr ? spheres_info->get_info() :
           mesh_info != nullptr ? mesh_info->get_info() : 
           file_info != nullptr ? file_info->get_info() : "";
}
// template <typename T, typename... Args>
// UniquePtr<T> get_ptr(Args&&... args) {
//     return luisa::make_unique<T>(std::forward<Args>(args)...);
// }
// const luisa::string RawSurfaceInfo::mat_string[5] = {
//     "null", "metal", "substrate", "matte", "glass"
// };

} // namespace luisa::render