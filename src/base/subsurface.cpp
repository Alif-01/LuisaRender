//
// Created by Mike on 2021/12/14.
//

#include <base/subsurface.h>
#include <base/scene.h>
#include <base/interaction.h>
#include <base/pipeline.h>

namespace luisa::render {

Subsurface::Subsurface(Scene *scene, const SceneNodeDesc *desc) noexcept:
    SceneNode{scene, desc, SceneNodeTag::SUBSURFACE} {}

void Subsurface::update(Scene *scene, const SceneNodeDesc *desc) noexcept {
    // TODO: update subsurface
}

luisa::unique_ptr<Subsurface::Instance> Subsurface::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return _build(pipeline, command_buffer);
}

void Subsurface::Instance::closure(
    PolymorphicCall<Closure> &call, const Interaction &it,
    const SampledWavelengths &swl, Expr<float> time
) const noexcept {
    auto cls = call.collect(closure_identifier(), [&] {
        return create_closure(swl, time);
    });
    populate_closure(cls, it);
}

Subsurface::Evaluation Subsurface::Closure::evaluate(
    const Interaction &it_i, TransportMode mode
) const noexcept {
    // auto &&ctx = context<Context>();
    // auto wo_local = ctx.it.shading().world_to_local(wo);
    // auto wi_local = pi.shading().world_to_local(wi);
    // Float ft = fresnel_dielectric(cos_theta(wo_local), 1, ctx.eta);
    // SampledSpectrum f = (1 - Ft) * sp(it_i) * sw(wi);
    // return {.f = f * abs_cos_theta(wi_local), .pdf = pdf};
    auto &it = it();
    auto d = it.p() - it_i.p();
    auto d_local = it.shading().world_to_local(d);
    ArrayFloat<3> r_project = {
        sqrt(d_local.y * d_local.y + d_local.z * d_local.z),
        sqrt(d_local.z * d_local.z + d_local.x * d_local.x),
        sqrt(d_local.x * d_local.x + d_local.y * d_local.y)
    };
    static const ArrayFloat<3> axis_prob = {.25f, .25f, .5f};

    auto pdf = def(0.f);
    for (int axis = 0; axis < 3; ++axis)
        pdf += pdf_sr(r_project[axis]) * axis_prob[axis];

    return {
        .f = sr(distance(it().p(), pi.p())),
        .pdf = pdf;
    };
}

Subsurface::Sample Subsurface::Closure::sample(
    Expr<float> u_lobe, Expr<float2> u, TransportMode mode
) const noexcept {
    // auto &&ctx = context<Context>();
    auto r = sample_r(u.x);
    auto r_max = sample_r(1.f);
    auto phi = 2 * pi * u.y;
    
    auto l = sqrt(r_max * r_max - r * r);
    auto &it = it();
    auto &fr = it.shading();
    auto test_origin = make_float3(0.f);
    auto test_dir = make_float3(0.f);
    auto test_tmin = def(0.f);
    auto test_tmax = l * 2.f;

    $if (u_lobe < .5f) {
        test_origin = it.p() + (fr.s() * cos(phi) + fr.t() * sin(phi)) * r - fr.n() * l;
        test_dir = fr.n();
        u_lobe *= 2;
    }
    $elif (u_lobe < .75f) {
        test_origin = it.p() + (fr.t() * cos(phi) + fr.n() * sin(phi)) * r - fr.s() * l;
        test_dir = fr.s();
        u_lobe = (u_lobe - .5f) * 4.f;
    }
    $else {
        test_origin = it.p() + (fr.n() * cos(phi) + fr.s() * sin(phi)) * r - fr.t() * l;
        test_dir = fr.t();
        u_lobe = (u_lobe - .75f) * 4.f;
    };
    // auto test_target = test_origin + test_dir * test_tmax;

    // Choose spectral channel for BSSRDF sampling
    // int ch = Clamp((int)(u1 * Spectrum::nSamples), 0, Spectrum::nSamples - 1);
    // u1 = u1 * Spectrum::nSamples - ch;

    ArrayVar<CommittedHit, sample_capacity> sample_hit;
    auto n_found = def(0u);
    $loop {
        test_min += min((test_max - test_min) * 0.001f, 1e-6f);
        $if (test_min > test_max) { $break; };
        auto test_ray = make_ray(test_origin, test_dir, test_tmin, test_tmax);
        auto test_hit = pipeline().geometry().trace_closest(test_ray);
        $if (test_hit->miss()) { $break; };
        auto test_inst = pipeline().geometry().instance(test_hit.inst);

        $if (test_inst.has_subsurface() &
             test_inst.subsurface_tag() == it.shape().subsurface_tag()) {
            sample_hit[n_found] = test_hit;
            test_min = test_hit->distance();
            n_found += 1;
        };
    };

    auto select = cast<uint>(clamp(u_lobe * cast<float>(n_found), 0.f, cast<float>(n_found) - 1.f));
    auto sample = Subsurface::Sample::zero(swl().dimension());
    $if (r >= 0.f & r < r_max & n_found > 0) {
        auto test_sel_tmin = ite(select == 0u, 0.f,
            (sample_hit[select - 1]->distance() + sample_hit[select - 1]->distance()) * 0.5f);
        auto test_ray = make_ray(test_origin, test_dir, test_sel_tmin, test_tmax);
        sample.it = pipeline().geometry().interaction(test_ray, sample_hit[select]);
        $if (sample.it.back_facing()) {
            sample.dist = ite(select == 0u, sample_hit[select]->distance(),
                (sample_hit[select]->distance() - sample_hit[select - 1]->distance()) * 0.5f
            );
        }
        $else {
            sample.dist = ite(select == n_found - 1, test_tmax - sample_hit[select]->distance(),
                (sample_hit[select + 1]->distance() - sample_hit[select]->distance()) * 0.5f
            );
        };
        sample.eval = evaluate(sample.it);
        sample.eval.pdf /= cast<float>(n_found);
    };

    return sample;
}

}// namespace luisa::render
