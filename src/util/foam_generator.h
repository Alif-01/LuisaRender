

namespace luisa::render {

struct FoamSettings {
    float wc_min, wc_max;
    float ta_min, ta_max;
    float k_mid, k_max;
};

void generateFoamParticles(
    const luisa::vector<float> &positions,
    const luisa::vector<float> &velocities,
	const FoamSettings *foam_settings
) noexcept {
    if (positions.size() % 3u != 0u)
        LUISA_ERROR_WITH_LOCATION("Invalid particle count.");
	Vector3r e1, e2;
	const Vector3r v = v0[index]; 
	Vector3r vn = v; 
	vn.normalize();
	getOrthogonalVectors(vn, e1, e2);

	e1 = particleRadius * e1;
	e2 = particleRadius * e2;

	for (unsigned int i = 0; i < numParticles; i++)
	{
		// Generate a random distribution of the foam particles in a cylinder.
		const Real Xr = uniform_distr_0_1(random_generator);
		const Real Xtheta = uniform_distr_0_1(random_generator);
		const Real Xh = uniform_distr_0_1(random_generator);

		const Real r = particleRadius * sqrt(Xr);
		const Real theta = Xtheta * static_cast<Real>(2.0 * M_PI);
		const Real h = (Xh- static_cast<Real>(0.5)) * timeStepSize * v.norm();

		const Vector3r xd = x0[index] + r * cos(theta) * e1 + r * sin(theta) * e2 + h * vn;
		const Vector3r vd = r * cos(theta) * e1 + r * sin(theta) * e2 + v;
		
		unsigned char generatorType = GeneratedType::TrappedAir;
		if (i >= numTrappedAir)
			generatorType = GeneratedType::WaveCrest;
		if (i >= (numTrappedAir + numWaveCrest))
			generatorType = GeneratedType::Vorticity;


#ifdef _OPENMP
		int tid = omp_get_thread_num();
#else
		int tid = 0;
#endif
		fxPerThread[tid].push_back(xd);
		fvPerThread[tid].push_back(vd);
		if (splitGenerators)
		{
			particleTypePerThread[tid].push_back(generatorType);
			numParticlesOfTypePerThread[tid][generatorType]++;
		}
		else
		{
			particleTypePerThread[tid].push_back(ParticleType::Foam); // default, actual value will be set during advection
		}
		Real lt = lifetimeMin + I_ke / keMax * uniform_distr_0_1(random_generator) *(lifetimeMax - lifetimeMin);
		flifetimePerThread[tid].push_back(lt);
	}
}

}  // namespace luisa::render