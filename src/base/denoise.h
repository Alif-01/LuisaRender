#pragma once
#include <luisa/backends/cuda/optix_api.h>
#include <luisa/backends/cuda/cuda_device.h>
#include <luisa/backends/cuda/cuda_buffer.h>
#include <luisa/runtime/stream.h>

namespace luisa::compute::cuda {

class Denoiser {
private:
    CUDADevice *_device;
    optix::Denoiser _denoiser;
public:
    Denoiser(CUDADevice *device);
    ~Denoiser();
};

}