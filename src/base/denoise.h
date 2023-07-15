#pragma once
#include <backends/cuda/optix_api.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_buffer.h>
#include <runtime/stream.h>

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