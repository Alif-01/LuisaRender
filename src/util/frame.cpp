//
// Created by Mike Smith on 2022/1/13.
//

#include <luisa/dsl/sugar.h>
#include <util/scattering.h>
#include <util/frame.h>

namespace luisa::render {

Frame::Frame(Expr<float3> s,
             Expr<float3> t,
             Expr<float3> n) noexcept
    : _s{s}, _t{t}, _n{n} {}

Frame::Frame() noexcept
    : _s{make_float3(1.f, 0.f, 0.f)},
      _t{make_float3(0.f, 1.f, 0.f)},
      _n{make_float3(0.f, 0.f, 1.f)} {}

Frame Frame::make(Expr<float3> n) noexcept {
    auto sgn = sign(n.z);
    auto a = -1.f / (sgn + n.z);
    auto b = n.x * n.y * a;
    auto s = make_float3(1.f + sgn * sqr(n.x) * a, sgn * b, -sgn * n.x);
    auto t = make_float3(b, sgn + sqr(n.y) * a, -n.y);
    return {normalize(s), normalize(t), n};
}

Frame Frame::make(Expr<float3> n, Expr<float3> s) noexcept {
    auto ss = normalize(s - n * dot(n, s));
    auto tt = normalize(cross(n, ss));
    return {ss, tt, n};
}

Float3 Frame::local_to_world(Expr<float3> d) const noexcept {
    return normalize(d.x * _s + d.y * _t + d.z * _n);
}

Float3 Frame::world_to_local(Expr<float3> d) const noexcept {
    return normalize(make_float3(dot(d, _s), dot(d, _t), dot(d, _n)));
}

void Frame::flip() noexcept {
    _n = -_n;
    _t = -_t;
}

/* If the shading normal results in specular reflection in the lower hemisphere, raise the shading
 * normal towards the geometry normal so that the specular reflection is just above the surface.
 * Only used for glossy materials. */
Float3 clamp_shading_normal(Expr<float3> ns, Expr<float3> ng, Expr<float3> wo) noexcept {
    auto wi_refl = reflect(-wo, ns);
    auto wo_z = dot(wo, ng);
    // assert(wo_z >= 0);

    /* Reflection rays may always be at least as shallow as the incoming ray. */
    auto threshold = ite(wo_z > 0.f, min(0.9f * wo_z, 0.01f),
                                     max(0.9f * wo_z, -0.01f));

    /* Form coordinate system with Ng as the Z axis and N inside the X-Z-plane.
     * The X axis is found by normalizing the component of N that's orthogonal to Ng.
     * The Y axis isn't actually needed.
     */
    auto X = normalize(ns - dot(ns, ng) * ng);

    /* Calculate N.z and N.x in the local coordinate system.
     *
     * The goal of this computation is to find a N' that is rotated towards Ng just enough
     * to lift R' above the threshold (here called t), therefore dot(R', Ng) = t.
     *
     * According to the standard reflection equation,
     * this means that we want dot(2*dot(N', I)*N' - I, Ng) = t.
     *
     * Since the Z axis of our local coordinate system is Ng, dot(x, Ng) is just x.z, so we get
     * 2*dot(N', I)*N'.z - I.z = t.
     *
     * The rotation is simple to express in the coordinate system we formed -
     * since N lies in the X-Z-plane, we know that N' will also lie in the X-Z-plane,
     * so N'.y = 0 and therefore dot(N', I) = N'.x*I.x + N'.z*I.z .
     *
     * Furthermore, we want N' to be normalized, so N'.x = sqrt(1 - N'.z^2).
     *
     * With these simplifications, we get the equation
     * 2*(sqrt(1 - N'.z^2)*I.x + N'.z*I.z)*N'.z - I.z = t,
     * or
     * 2*sqrt(1 - N'.z^2)*I.x*N'.z = t + I.z * (1 - 2*N'.z^2),
     * after rearranging terms.
     * Raise both sides to the power of two and substitute terms with
     * a = I.x^2 + I.z^2,
     * b = 2*(a + Iz*t),
     * c = (Iz + t)^2,
     * we obtain
     * 4*a*N'.z^4 - 2*b*N'.z^2 + c = 0.
     *
     * The only unknown here is N'.z, so we can solve for that.
     *
     * The equation has four solutions in general, two can immediately be discarded because they're
     * negative so N' would lie in the lower hemisphere; one solves
     * 2*sqrt(1 - N'.z^2)*I.x*N'.z = -(t + I.z * (1 - 2*N'.z^2))
     * instead of the original equation (before squaring both sides).
     * Therefore only one root is valid.
     */

    auto wo_x = dot(wo, X);
    auto a = sqr(wo_x) + sqr(wo_z);
    auto b = (a + wo_z * threshold) * 2.0f;
    auto c = sqr(threshold + wo_z);

    /* In order that the root formula solves 2*sqrt(1 - N'.z^2)*I.x*N'.z = t + I.z - 2*I.z*N'.z^2,
     * Ix and (t + I.z * (1 - 2*N'.z^2)) must have the same sign (the rest terms are non-negative by
     * definition). */
    auto nz2 = ite((wo_x > 0.f) ^ (wo_z > 0.f),
        0.25f * (b + sqrt(sqr(b) - 4.0f * a * c)) / a,
        0.25f * (b - sqrt(sqr(b) - 4.0f * a * c)) / a);

    auto nx = sqrt(1.0f - nz2);
    auto nz = sqrt(nz2);
    return ite((wo_z > 0.f) ^ (dot(ng, wi_refl) >= threshold),
               nx * X + nz * ng, ns);
}

}// namespace luisa::render
