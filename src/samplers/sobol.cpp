//
// Created by Mike Smith on 2022/2/9.
//

#include <bit>

#include <dsl/sugar.h>
#include <util/rng.h>
#include <util/sobolmatrices.h>
#include <base/sampler.h>
#include <base/pipeline.h>

namespace luisa::render {

class SobolSampler final : public Sampler {

public:
    SobolSampler(Scene *scene, const SceneNodeDesc *desc) noexcept : Sampler{scene, desc} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
};

using namespace luisa::compute;

class SobolSamplerInstance final : public Sampler::Instance {

private:
    uint2 _resolution;
    uint _scale{};
    luisa::optional<UInt2> _pixel;
    luisa::optional<UInt> _dimension;
    luisa::optional<ULong> _sobol_index;
    Buffer<uint> _sobol_matrices;
    Buffer<ulong> _vdc_sobol_matrices;
    Buffer<ulong> _vdc_sobol_matrices_inv;
    Buffer<uint4> _state_buffer;

private:
    [[nodiscard]] static auto _fast_owen_scramble(Expr<uint> seed, UInt v) noexcept {
        v = reverse(v);
        v ^= v * 0x3d20adeau;
        v += seed;
        v *= (seed >> 16u) | 1u;
        v ^= v * 0x05526c56u;
        v ^= v * 0x53a22864u;
        return reverse(v);
    }

    template<bool scramble>
    [[nodiscard]] auto _sobol_sample(ULong a, Expr<uint> dimension, Expr<uint> hash) const noexcept {
        static Callable impl = [](ULong a, UInt dimension, BufferVar<uint> sobol_matrices, UInt hash) noexcept {
            auto v = def(0u);
            auto i = def(dimension * SobolMatrixSize);
            $while (a != 0ull) {
                v = ite((a & 1ull) != 0ull, v ^ sobol_matrices.read(i), v);
                a = a >> 1ull;
                i = i + 1u;
            };
            if constexpr (scramble) { v = _fast_owen_scramble(hash, v); }
            return cast<float>(v) * 0x1p-32f;
        };
        return impl(a, dimension, _sobol_matrices.view(), hash);
    }

    [[nodiscard]] auto _sobol_interval_to_index(uint m, UInt frame, Expr<uint2> p) const noexcept {
        if (m == 0u) { return cast<ulong>(frame); }
        static Callable impl = [](UInt m, UInt frame, UInt2 p, BufferVar<ulong> vdc, BufferVar<ulong> vdc_inv) noexcept {
            auto c = def(0u);
            auto m2 = m << 1u;
            auto index = cast<ulong>(frame) << cast<ulong>(m2);
            auto delta = def<ulong>(0ull);
            $while (frame != 0u) {
                $if ((frame & 1u) != 0u) {
                    auto v = vdc.read(c);
                    delta = delta ^ v;
                };
                frame >>= 1u;
                c += 1u;
            };
            // flipped b
            auto b = delta ^ ((cast<ulong>(p.x) << m) | cast<ulong>(p.y));
            auto d = def(0u);
            $while (b != 0ull) {
                $if ((b & 1ull) != 0ull) {
                    auto v = vdc_inv.read(d);
                    index = index ^ v;
                };
                b = b >> 1ull;
                d += 1u;
            };
            return index;
        };
        return impl(m, frame, p, _vdc_sobol_matrices.view(), _vdc_sobol_matrices_inv.view());
    }

public:
    explicit SobolSamplerInstance(
        Pipeline &pipeline, CommandBuffer &command_buffer, const SobolSampler *s) noexcept:
        Sampler::Instance{pipeline, s} {
        _sobol_matrices = pipeline.device().create_buffer<uint>(SobolMatrixSize * NSobolDimensions);
        _vdc_sobol_matrices = pipeline.device().create_buffer<ulong>(SobolMatrixSize);
        _vdc_sobol_matrices_inv = pipeline.device().create_buffer<ulong>(SobolMatrixSize);
        command_buffer << _sobol_matrices.copy_from(SobolMatrices32);
    }
    void reset(CommandBuffer &command_buffer, uint2 resolution, uint state_count, uint spp) noexcept override {
        if (spp != next_pow2(spp)) {
            LUISA_WARNING_WITH_LOCATION(
                "Non power-of-two samples per pixel "
                "is not optimal for Sobol' sampler.");
        }
        if (!_state_buffer || _state_buffer.size() < state_count) {
            _state_buffer = pipeline().device().create_buffer<uint4>(
                next_pow2(state_count));
        }
        _resolution = resolution;
        _scale = next_pow2(std::max(resolution.x, resolution.y));
        LUISA_ASSERT(_scale <= 0xffffu, "Sobol sampler scale is too large.");
        auto m = std::bit_width(_scale) - 1u;
        std::array<ulong, SobolMatrixSize> vdc_sobol_matrices{};
        std::array<ulong, SobolMatrixSize> vdc_sobol_matrices_inv{};
        for (auto i = 0u; i < SobolMatrixSize; i++) {
            vdc_sobol_matrices[i] = VdCSobolMatrices[m - 1u][i];
            vdc_sobol_matrices_inv[i] = VdCSobolMatricesInv[m - 1u][i];
        }
        command_buffer << _vdc_sobol_matrices.copy_from(vdc_sobol_matrices.data())
                       << _vdc_sobol_matrices_inv.copy_from(vdc_sobol_matrices_inv.data())
                       << commit();
    }
    void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept override {
        _dimension.emplace(2u);
        _sobol_index.emplace(_sobol_interval_to_index(
            std::bit_width(_scale) - 1u, sample_index, pixel));
        _pixel.emplace(pixel);
    }
    void save_state(Expr<uint> state_id) noexcept override {
        auto sobol_index_hi = cast<uint>(_sobol_index.value() >> 32ull);
        auto sobol_index_lo = cast<uint>(_sobol_index.value());
        auto pixel_index = _pixel.value().y * _resolution.x + _pixel.value().x;
        auto state = make_uint4(sobol_index_hi, sobol_index_lo, *_dimension, pixel_index);
        _state_buffer->write(state_id, state);
    }
    void load_state(Expr<uint> state_id) noexcept override {
        auto state = _state_buffer->read(state_id);
        auto sobol_index_hi = cast<ulong>(state.x) << 32ull;
        auto sobol_index_lo = cast<ulong>(state.y);
        _sobol_index.emplace(sobol_index_hi | sobol_index_lo);
        _dimension.emplace(state.z);
        _pixel.emplace(make_uint2(state.w / _resolution.x, state.w % _resolution.x));
    }
    [[nodiscard]] Float generate_1d() noexcept override {
        *_dimension = ite(*_dimension >= NSobolDimensions, 2u, *_dimension);
        auto hash = xxhash32(make_uint2(*_dimension, node()->seed()));
        auto u = _sobol_sample<true>(*_sobol_index, *_dimension, hash);
        *_dimension += 1u;
        return clamp(u, 0.f, one_minus_epsilon);
    }
    [[nodiscard]] Float2 generate_2d() noexcept override {
        *_dimension = ite(*_dimension + 1u >= NSobolDimensions, 2u, *_dimension);
        auto hx = xxhash32(make_uint2(*_dimension, node()->seed()));
        auto hy = xxhash32(make_uint2(*_dimension + 1u, node()->seed()));
        auto ux = _sobol_sample<true>(*_sobol_index, *_dimension, hx);
        auto uy = _sobol_sample<true>(*_sobol_index, *_dimension + 1u, hy);
        *_dimension += 2u;
        return clamp(make_float2(ux, uy), 0.f, one_minus_epsilon);
    }
    [[nodiscard]] Float2 generate_pixel_2d() noexcept override {
        auto ux = _sobol_sample<false>(*_sobol_index, 0u, 0u);
        auto uy = _sobol_sample<false>(*_sobol_index, 1u, 0u);
        auto s = static_cast<float>(_scale);
        return clamp(make_float2(ux, uy) * s - make_float2(*_pixel),
                     0.f, one_minus_epsilon);
    }
};

luisa::unique_ptr<Sampler::Instance> SobolSampler::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<SobolSamplerInstance>(pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SobolSampler)
