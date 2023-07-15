#include <base/denoise.h>

namespace luisa::render {

Denoiser::Denoiser(CUDADevice *device) : _device(device), _denoiser(nullptr) {
    auto optix_ctx = _device->handle().optix_context();
    auto model_kind = optix::DENOISER_MODEL_KIND_LDR;

    optix::DenoiserOptions options = {};
    options.guideAlbedo = 0;
    options.guideNormal = 0;

    LUISA_CHECK_OPTIX(optix::api().denoiserCreate(optix_ctx, model_kind, &options, &_denoiser));
}

Denoiser::~Denoiser() {
    LUISA_CHECK_OPTIX(optix::api().denoiserDestroy(_denoiser));
}

}