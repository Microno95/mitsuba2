#include <random>
#include <enoki/stl.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>


NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class RaymarchingPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {

public:
MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
MTS_IMPORT_TYPES(Scene, Sampler, Emitter, EmitterPtr, BSDF, BSDFPtr,
                    Medium, MediumPtr, PhaseFunctionContext)

RaymarchingPathIntegrator(const Properties &props) : Base(props) {
    m_volume_step_size   = props.float_("volume_step_size", 0.025f);
    int strat_sample_init = props.int_("stratified_samples", 1);
    if (strat_sample_init <= 0) {
        std::stringstream oss;
        oss << "stratified_samples cannot be <= 0, was given " << strat_sample_init << ", set stratified_samples to 1";
        Log(Warn, "%s", oss.str());
    }
    m_stratified_samples    = std::max(strat_sample_init, 1);
    m_atol                  = props.float_("absolute_tolerance", 1e-5f);
    m_rtol                  = props.float_("relative_tolerance", 1e-5f);
    m_use_adaptive_sampling = props.bool_("adaptive_stepping", false);
    m_use_bisection         = props.bool_("exact_termination", false);
}

MTS_INLINE
Float index_spectrum(const UnpolarizedSpectrum &spec, const UInt32 &idx) const {
    Float m = spec[0];
    if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
        masked(m, eq(idx, 1u)) = spec[1];
        masked(m, eq(idx, 2u)) = spec[2];
    } else {
        ENOKI_MARK_USED(idx);
    }
    return m;
}

MediumInteraction3f sample_raymarched_interaction(const Ray3f &ray,
                                                  MediumPtr medium, 
                                                  UInt32 channel,
                                                  Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::MediumSample, active);

    // initialize basic medium interaction fields
    MediumInteraction3f mi = zero<MediumInteraction3f>();
    mi.sh_frame    = Frame3f(ray.d);
    mi.wi          = -ray.d;
    mi.time        = ray.time;
    mi.wavelengths = ray.wavelengths;

    auto [aabb_its, mint, maxt] = medium->intersect_aabb(ray);
    aabb_its &= (enoki::isfinite(mint) || enoki::isfinite(maxt));
    active &= aabb_its;
    masked(mint, !active) = 0.f;
    masked(maxt, !active) = math::Infinity<Float>;

    mint = max(ray.mint, mint);
    maxt = min(ray.maxt, maxt);

    if (any(!enoki::isfinite(mint) && active)) {
        std::ostringstream oss;
        oss << "[medium interaction error]: " << ray << std::endl << medium << std::endl << channel << std::endl << active << std::endl;
        Log(Error, "%s", oss.str());
    }

    // Sampling based on infinite homogeneous medium assumption
    Float sampled_t = mint;
    Mask valid_mi   = active && (sampled_t <= maxt);
    mi.t            = select(valid_mi, sampled_t, math::Infinity<Float>);
    mi.p            = ray(sampled_t);
    mi.medium       = medium;
    mi.mint         = mint;
    mi.maxt         = maxt;
    std::tie(mi.sigma_s, mi.sigma_n, mi.sigma_t) = medium->get_scattering_coefficients(mi, active);
    mi.radiance            = medium->get_radiance(mi, active);
    mi.combined_extinction = medium->get_combined_extinction(mi, active);
    return mi;
}

std::tuple<Spectrum, Float, Float>
integration_step(std::function<Spectrum(Float, Mask)>& df, Float dt, Mask active, Mask use_adaptive_sampling) const {
    if (m_use_adaptive_sampling) {
        Spectrum a1(0.f), a2(0.f), a3(0.f), a4(0.f), a5(0.f), est1(0.f), est2(0.f);
        Float err_estimate(0.f), curr_dt(dt), next_dt(dt);
        Mask step_rejected = true;
        // --------------------- RK45CK -------------------------- //
        // In order to accelerate the numerical integration via marching, we can use an adaptive stepping algorithm
        // We utilise the embedded Runge-Kutta 4(5) method with Cash-Karp coefficients
        // With this we can find the total optical depth of the ray segment that starts at
        // current_flight_distance and ends at current_flight_distance + dt
        // Since the ODE is linear and the right hand side is independent of the dependent variable (optical_depth)
        // We can simply accumulate the results of each segment as we step through them
        //
        // There is no 'a0' as the optical depth of a segment of length 0 is 0
        while (any(step_rejected)) {
            masked(a1, active && step_rejected) = df(0.2f   * curr_dt, active && step_rejected);
            masked(a2, active && step_rejected) = df(0.3f   * curr_dt, active && step_rejected);
            masked(a3, active && step_rejected) = df(0.6f   * curr_dt, active && step_rejected);
            masked(a4, active && step_rejected) = df(1.0f   * curr_dt, active && step_rejected);
            masked(a5, active && step_rejected) = df(0.875f * curr_dt, active && step_rejected);

            // 4th order estimate of optical_step
            masked(est1, active && step_rejected) = curr_dt * (0.40257648953301128f * a2 + 0.21043771043771045f * a3 + 0.28910220214568039f * a5);
            // 5th order estimate of optical_step
            masked(est2, active && step_rejected) = curr_dt * (0.38390790343915343f * a2 + 0.24459273726851852f * a3 + 0.01932198660714286f * a4 + 0.25000000000000000f * a5);

            // Error estimate from difference between 5th and 4th order estimates
            masked(err_estimate, active && step_rejected) = hmax(abs(detach(est2) - detach(est1)));
            // Based on scipy scaling of error
            auto scale = m_atol + max(hmax(abs(detach(est1))), hmax(abs(detach(est2)))) * m_rtol;
            auto error_norm = err_estimate / scale;

            Float corr = select(err_estimate > 0.f, 0.8f * enoki::pow(error_norm, -0.20f), 1.25f);

            masked(next_dt, active && step_rejected) = max(100.0f * math::RayEpsilon<Float>, 0.8f * curr_dt * min(10.f, max(0.2f, corr)));
            step_rejected &= (use_adaptive_sampling && (error_norm >= 1.0f) && (next_dt >= 200.0f * math::RayEpsilon<Float>)) && (next_dt < curr_dt);
            masked(curr_dt, active && step_rejected) = next_dt;
        }
        return std::make_tuple(est2, curr_dt, next_dt);
    } else {
        return std::make_tuple(dt * (df(0.f * dt, active) + df(dt, active)) * 0.5f, dt, dt);
    }
}

std::tuple<std::pair<Spectrum, Spectrum>, Float, Float>
integration_step_radiance_paired(std::function<std::pair<Spectrum, Spectrum>(Spectrum, Float, Mask)>& df_paired, Spectrum optical_depth, Float dt, Mask active, Mask use_adaptive_sampling) const {
    if (m_use_adaptive_sampling) {
        Spectrum a1_opt(0.f), a2_opt(0.f), a3_opt(0.f), a4_opt(0.f), a5_opt(0.f), est1_opt(0.f), est2_opt(0.f);
        Spectrum a1_rad(0.f), a2_rad(0.f), a3_rad(0.f), a4_rad(0.f), a5_rad(0.f), est1_rad(0.f), est2_rad(0.f);
        Float err_estimate(0.f), curr_dt(dt), next_dt(dt);
        Spectrum opt_depth1(0.f), rad1(0.f);
        Mask step_rejected = true;
        // --------------------- RK45CK -------------------------- //
        // In order to accelerate the numerical integration via marching, we can use an adaptive stepping algorithm
        // We utilise the embedded Runge-Kutta 4(5) method with Cash-Karp coefficients
        // With this we can find the total optical depth of the ray segment that starts at
        // current_flight_distance and ends at current_flight_distance + dt
        // Since the ODE is linear and the right hand side is independent of the dependent variable (optical_depth)
        // We can simply accumulate the results of each segment as we step through them
        //
        // There is no 'a0' as the optical depth of a segment of length 0 is 0
        while (any(step_rejected)) {
            std::tie(opt_depth1, rad1) = df_paired(optical_depth, 0.2f   * curr_dt, active && step_rejected);
            masked(a1_opt, active && step_rejected) = opt_depth1;
            masked(a1_rad, active && step_rejected) = rad1;
            std::tie(opt_depth1, rad1) = df_paired(optical_depth, 0.3f   * curr_dt, active && step_rejected);
            masked(a2_opt, active && step_rejected) = opt_depth1;
            masked(a2_rad, active && step_rejected) = rad1;
            std::tie(opt_depth1, rad1) = df_paired(optical_depth, 0.6f   * curr_dt, active && step_rejected);
            masked(a3_opt, active && step_rejected) = opt_depth1;
            masked(a3_rad, active && step_rejected) = rad1;
            std::tie(opt_depth1, rad1) = df_paired(optical_depth, 1.0f   * curr_dt, active && step_rejected);
            masked(a4_opt, active && step_rejected) = opt_depth1;
            masked(a4_rad, active && step_rejected) = rad1;
            std::tie(opt_depth1, rad1) = df_paired(optical_depth, 0.875f   * curr_dt, active && step_rejected);
            masked(a5_opt, active && step_rejected) = opt_depth1;
            masked(a5_rad, active && step_rejected) = rad1;

            // 4th order estimate of optical_step
            masked(est1_opt, active && step_rejected) = curr_dt * (0.40257648953301128f * a2_opt + 0.21043771043771045f * a3_opt + 0.28910220214568039f * a5_opt);
            masked(est1_rad, active && step_rejected) = curr_dt * (0.40257648953301128f * a2_rad + 0.21043771043771045f * a3_rad + 0.28910220214568039f * a5_rad);
            // 5th order estimate of optical_step
            masked(est2_opt, active && step_rejected) = curr_dt * (0.38390790343915343f * a2_opt + 0.24459273726851852f * a3_opt + 0.01932198660714286f * a4_opt + 0.25000000000000000f * a5_opt);
            masked(est2_rad, active && step_rejected) = curr_dt * (0.38390790343915343f * a2_rad + 0.24459273726851852f * a3_rad + 0.01932198660714286f * a4_rad + 0.25000000000000000f * a5_rad);

            // Error estimate from difference between 5th and 4th order estimates
            masked(err_estimate, active && step_rejected) = max(hmax(abs(detach(est2_opt) - detach(est1_opt))), hmax(abs(detach(est2_rad) - detach(est1_rad))));
            // Based on scipy scaling of error
            auto max_diff1 = max(hmax(abs(detach(est1_opt))), hmax(abs(detach(est1_rad))));
            auto max_diff2 = max(hmax(abs(detach(est2_opt))), hmax(abs(detach(est2_rad))));
            auto scale = m_atol + max(max_diff1, max_diff2) * m_rtol;
            auto error_norm = err_estimate / scale;

            Float corr = select(err_estimate > 0.f, 0.8f * enoki::pow(error_norm, -0.20f), 1.25f);

            masked(next_dt, active && step_rejected) = max(100.0f * math::RayEpsilon<Float>, 0.8f * curr_dt * min(10.f, max(0.2f, corr)));
            step_rejected &= (use_adaptive_sampling && (error_norm >= 1.0f) && (next_dt >= 200.0f * math::RayEpsilon<Float>)) && (next_dt < curr_dt);
            masked(curr_dt, active && step_rejected) = next_dt;
        }
        return std::make_tuple(std::make_pair(est2_opt, est2_rad), curr_dt, next_dt);
    } else {
        Spectrum opt_depth1(0.f), rad1(0.f);
        Spectrum a1_opt(0.f), a2_opt(0.f);
        Spectrum a1_rad(0.f), a2_rad(0.f);
        std::tie(opt_depth1, rad1) = df_paired(optical_depth, 0.0f * dt, active);
        masked(a1_opt, active) = opt_depth1;
        masked(a1_rad, active) = rad1;
        std::tie(opt_depth1, rad1) = df_paired(optical_depth, 1.0f * dt, active);
        masked(a2_opt, active) = opt_depth1;
        masked(a2_rad, active) = rad1;
        return std::make_tuple(
            std::make_pair(dt * (a1_opt + a2_opt) * 0.5f, dt * (a1_rad + a2_rad) * 0.5f), dt, dt
        );
    }
}

std::pair<Spectrum, Mask> sample(const Scene *scene,
                                    Sampler *sampler,
                                    const RayDifferential3f &ray_,
                                    const Medium *initial_medium,
                                    Float * /* aovs */,
                                    Mask active) const override {
    MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

    // If there is an environment emitter and emitters are visible: all rays will be valid
    // Otherwise, it will depend on whether a valid interaction is sampled
    Mask valid_ray = !m_hide_emitters && neq(scene->environment(), nullptr);

    // For now, don't use ray differentials
    Ray3f ray = ray_;

    // Tracks radiance scaling due to index of refraction changes
    Float eta(1.f);

    Spectrum throughput(1.f), result(0.f);
    MediumPtr medium = initial_medium;
    MediumInteraction3f mi = zero<MediumInteraction3f>();
    mi.t = math::Infinity<Float>;
    Mask specular_chain = active && !m_hide_emitters, use_adaptive_sampling = m_use_adaptive_sampling;
    UInt32 depth = 0;

    UInt32 channel = 0;
    if (is_rgb_v<Spectrum>) {
        uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
        channel = (UInt32) min(sampler->next_1d(active) * n_channels, n_channels - 1);
    }

    SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
    si.t = math::Infinity<Float>;
    Mask needs_intersection = true;
    for (int bounce = 0;; ++bounce) {
        // ----------------- Handle termination of paths ------------------

        // Russian roulette: try to keep path weights equal to one, while accounting for the
        // solid angle compression at refractive index boundaries. Stop with at least some
        // probability to avoid  getting stuck (e.g. due to total internal reflection)

        active &= any(neq(depolarize(throughput), 0.f));
        Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
        Mask perform_rr = (depth > (uint32_t) m_rr_depth);
        active &= sampler->next_1d(active) < q || !perform_rr;
        masked(throughput, perform_rr) *= rcp(detach(q));

        Mask exceeded_max_depth = depth >= (uint32_t) m_max_depth;

        if (none(active) || all(exceeded_max_depth))
            break;

        // ----------------------- Sampling the RTE -----------------------
        Mask active_medium  = active && neq(medium, nullptr);
        Mask active_surface = active && !active_medium;
        Mask act_null_scatter = false, act_medium_scatter = false, escaped_medium = false;

        if (any_or<true>(active_medium)) {
            mi = sample_raymarched_interaction(ray, medium, channel, active_medium);
            masked(ray.maxt, active_medium && medium->is_homogeneous() && mi.is_valid()) = mi.maxt;
            Mask intersect = needs_intersection && active_medium;

            if (any_or<true>(intersect)) {
                masked(si, intersect) = scene->ray_intersect(ray, intersect);
            }

            needs_intersection &= !intersect;

            // Get maximum flight distance of ray
            Float max_flight_distance      = min(si.t, mi.maxt) - mi.mint;
            Float current_flight_distance  = max(mi.t - mi.mint, 0.f);
            Float dt                       = max_flight_distance - current_flight_distance;
            UInt32 stratified_sample_count = 0;
            Float max_throughput           = (sampler->next_1d(active_medium) + (m_stratified_samples - stratified_sample_count - 1.0f)) / m_stratified_samples;
            Float desired_density          = -enoki::log(max_throughput);

            // Instantiate masks that track which rays are able to continue marching
            Mask iteration_mask = active_medium;
            masked(mi.t, eq(max_flight_distance, math::Infinity<Float>)) = math::Infinity<Float>;
            iteration_mask &= mi.is_valid() && current_flight_distance < max_flight_distance;
            Mask reached_density = false;

            // Instantiate tracking of optical depth, this will be used to estimate throughput
            Spectrum optical_depth(0.f), optical_step(0.f), tr(1.0f), total_radiance(0.f);

            // Create variables for local interaction parameters
            auto [local_ss, local_sn, local_st] = medium->get_scattering_coefficients(mi, iteration_mask);
            Spectrum local_radiance = medium->get_radiance(mi, iteration_mask);

            // Initialise step size
            // For inhomogeneous media we will use RK45 to do the integration
            // For inhomogeneous media we need to take integration steps, here we jitter the initial step and impose that a step is at least 1 ray epsilon
            masked(dt, iteration_mask && !medium->is_homogeneous()) = max(math::RayEpsilon<Float>, min(dt, m_volume_step_size) * sampler->next_1d(iteration_mask));
            // For homogeneous media we can simply sample the correct distance to achieve the desired density
            masked(dt, iteration_mask &&  medium->is_homogeneous()) = min(dt, desired_density / index_spectrum(local_st, channel));

            // RK45 parameters
            //     Optical Depth Function
            std::function<Spectrum(Float, Mask)> df_opt = [&ray, &mi, &medium, &current_flight_distance](Float dt_in, Mask active) {
                masked(mi.p, active) = ray(current_flight_distance + dt_in + mi.mint);
                auto [local_ss, local_sn, local_st] = medium->get_scattering_coefficients(mi, active);
                return local_st;
            };

            // {
            //     std::ostringstream oss;
            //     oss << "[main volume transport init]: " << dt << ", " << desired_density << ", " << iteration_mask << ", " << max_flight_distance << ", " << current_flight_distance << ", " << mi.t << ", " << si.t << ", " << mi.mint << ", " << mi.maxt;
            //     Log(Debug, "%s", oss.str());
            // }

            while (any(iteration_mask)) {
                MTS_MASKED_FUNCTION(ProfilerPhase::MediumRaymarch, iteration_mask);
            
                auto [next_depth, curr_dt, next_dt] = integration_step(df_opt, dt, iteration_mask, use_adaptive_sampling);
                masked(optical_step, iteration_mask) = next_depth;

                // --------------------- Correction ---------------------- //
                // Check if we exceed the desired density by taking this step
                // If we do, find the point between this point and the last point that reaches the desired density
                // The desired density and the obtained density may differ since we sample the density at the point we interpolate to
                // And if the step size is larger than 2x the mean grid spacing then this can be quite different
                Mask optical_depth_needs_correction = iteration_mask && (index_spectrum(optical_depth + optical_step, channel) >= desired_density);
                Spectrum old_optical_step = optical_step, old_optical_depth = optical_depth;
                Float old_desired_density = desired_density;

                if (any_or<true>(optical_depth_needs_correction)) {
                    if (m_use_bisection) {
                        Mask not_found_depth = optical_depth_needs_correction;
                        UInt32 iters_used = 0;
                        // Bisection method to find the point at which the optical thickness matches the target
                        auto [a_depth, a_dt, a_ndt] = integration_step(df_opt, curr_dt * 1.f,    optical_depth_needs_correction, false);
                        auto [b_depth, b_dt, b_ndt] = integration_step(df_opt, curr_dt * 0.f,    optical_depth_needs_correction, false);
                        while (any(not_found_depth)) {
                            Float half_step = (detach(b_dt) + detach(a_dt)) * 0.5f;
                            // Float half_step = (index_spectrum(desired_density - (optical_depth + a_depth), channel) / index_spectrum(b_depth - a_depth, channel)) * (detach(b_dt) - detach(a_dt));
                            auto [m_depth, m_dt, m_ndt] = integration_step(df_opt, half_step, not_found_depth, use_adaptive_sampling);
                            Mask upper_half = index_spectrum(m_depth, channel) > index_spectrum(desired_density - (optical_depth + a_depth), channel);
                            upper_half |= m_dt <= half_step;
                            std::tie(a_depth, a_dt, a_ndt) = std::make_tuple(
                                select(upper_half, m_depth, a_depth),
                                select(upper_half, m_dt,    a_dt),
                                select(upper_half, m_ndt,   a_ndt)
                            );
                            std::tie(b_depth, b_dt, b_ndt) = std::make_tuple(
                                select(upper_half, b_depth, m_depth),
                                select(upper_half, b_dt,    m_dt),
                                select(upper_half, b_ndt,   m_ndt)
                            );
                            masked(iters_used, not_found_depth) += 1;
                            not_found_depth &= index_spectrum(enoki::abs(b_depth - a_depth), channel) >= m_atol && iters_used < 4;
                            // std::ostringstream oss;
                            // oss << "[main volume bisection iter]: " << a_dt - b_dt << ", " << m_atol * 2.f;
                            // Log(Debug, "%s", oss.str());
                        }
                        masked(optical_step, optical_depth_needs_correction) = a_depth;
                        masked(curr_dt,      optical_depth_needs_correction) = a_dt;
                        masked(next_dt,      optical_depth_needs_correction) = a_ndt;
                    } else {
                        Float interp(1.0f);
                        // Simple linear interpolation assuming constant optical thickness
                        masked(interp, optical_depth_needs_correction && index_spectrum(optical_step, channel) > 0.f) = index_spectrum(desired_density - optical_depth, channel) / index_spectrum(optical_step, channel);
                        // masked(interp, optical_depth_needs_correction && !medium->is_homogeneous()) *= 0.95f;
                        interp = clamp(interp, 0.f, 1.f);

                        // Sample new points based on updated step
                        auto [corr_next_depth, corr_curr_dt, corr_next_dt]   = integration_step(df_opt, curr_dt * interp, optical_depth_needs_correction, use_adaptive_sampling);
                        masked(optical_step, optical_depth_needs_correction) = corr_next_depth;
                        masked(next_dt,      optical_depth_needs_correction) = corr_next_dt;
                        masked(curr_dt,      optical_depth_needs_correction) = corr_curr_dt;
                    }
                }
                // ------------------------------------------------------- //
                masked(dt, iteration_mask) = curr_dt;

                // Update accumulators and ray position
                masked(current_flight_distance, iteration_mask) += dt;
                masked(optical_depth, iteration_mask)           += optical_step;
                masked(mi.t, iteration_mask)                    += dt;
                masked(mi.p, iteration_mask)                     = ray(current_flight_distance + mi.mint);
                Mask sample_radiance                             = optical_depth_needs_correction && (desired_density - index_spectrum(optical_depth, channel) < math::RayEpsilon<Float>);

                if (any_or<true>(sample_radiance)) {
                    std::tie(local_ss, local_sn, local_st)            = medium->get_scattering_coefficients(mi, sample_radiance);
                    local_radiance                                    = medium->get_radiance(mi, sample_radiance);
                    tr                                                = exp(-optical_depth);
                    Spectrum path_pdf                                 = select(mi.t < max_flight_distance, tr * local_st, tr) * m_stratified_samples;
                    Float tr_pdf                                      = index_spectrum(path_pdf, channel);
                    masked(result, sample_radiance)                  += select(tr_pdf > 0.f, throughput * tr * local_radiance * (local_st - local_ss) / tr_pdf, 0.f);
                    masked(stratified_sample_count, sample_radiance) += 1;
                    masked(max_throughput, sample_radiance)           = (sampler->next_1d(sample_radiance) + (m_stratified_samples - stratified_sample_count - 1.0f)) / m_stratified_samples;
                    masked(desired_density, sample_radiance)          = -enoki::log(max_throughput);
                }
                
                reached_density |= sample_radiance && (stratified_sample_count == m_stratified_samples);

                // {
                //     std::ostringstream oss;
                //     oss << "[main volume transport iter]: " << desired_density << ", " << stratified_sample_count << ", " << optical_depth_needs_correction << ", " << reached_density << ", " << next_dt << ", " << curr_dt << ", " << optical_step << ", " << channel << ", " << iteration_mask << ", " << current_flight_distance << ", " << optical_depth;
                //     Log(Debug, "%s", oss.str());
                // }

                // Update step size for next iteration
                // If the volume is inhomogeneous, we use the smaller of the adaptive step or the remaining distance
                // If the volume is homogeneous, then next_dt is infinite and therefore we either
                //     terminate the iteration (current_flight_distance == max_flight_distance)
                //     or we take a step to close the gap between max_flight_distance and current_flight_distance
                if (m_use_adaptive_sampling) {
                    masked(dt, iteration_mask) = max_flight_distance - current_flight_distance;
                    masked(dt, iteration_mask && !medium->is_homogeneous()) = min(dt, next_dt);
                    masked(dt, iteration_mask &&  medium->is_homogeneous()) = min(dt, desired_density / index_spectrum(local_st, channel));
                } else {
                    masked(dt, iteration_mask) = max_flight_distance - current_flight_distance;
                    masked(dt, iteration_mask && !medium->is_homogeneous()) = min(dt, m_volume_step_size);
                    masked(dt, iteration_mask &&  medium->is_homogeneous()) = min(dt, desired_density / index_spectrum(local_st, channel));
                }

                if (any(optical_step != optical_step)) {
                        std::ostringstream oss;
                        oss << "[main volume transport error]: " << old_desired_density << ", " << desired_density << ", " << optical_depth_needs_correction << ", " << reached_density << ", " << dt << ", " << next_dt << ", " << curr_dt << ", " << old_optical_step << ", " << optical_step << ", " << channel << ", " << iteration_mask << ", " << current_flight_distance << ", " << old_optical_depth << ", " << optical_depth;
                        Log(Warn, "%s", oss.str());
                }

                // Update iteration mask
                // Marching should end when either throughput falls below sampled throughput or we exceed distance to the surface
                iteration_mask &= (current_flight_distance < max_flight_distance - math::RayEpsilon<Float>) && !reached_density;
            }

            masked(mi.p, active_medium) = ray(current_flight_distance + mi.mint);
            std::tie(local_ss, local_sn, local_st) = medium->get_scattering_coefficients(mi, active_medium);
            local_radiance = medium->get_radiance(mi, active_medium);
            tr = exp(-optical_depth);
            Spectrum path_pdf = select(mi.t < max_flight_distance, tr * local_st, tr) * m_stratified_samples;
            Float tr_pdf      = index_spectrum(path_pdf, channel);

            masked(result, active_medium)     += select(tr_pdf > 0.f, (m_stratified_samples - stratified_sample_count) * throughput * tr * local_radiance * (local_st - local_ss) / tr_pdf, 0.f);
            masked(throughput, active_medium) *= select(tr_pdf > 0.f, (m_stratified_samples - stratified_sample_count + 1.0f) * tr / tr_pdf, 0.f);

            masked(mi.t, active_medium && mi.t >= max_flight_distance) = math::Infinity<Float>;
            masked(mi.t, active_medium && mi.t >= si.t)                = math::Infinity<Float>;
            
            if (any(tr != tr)) {
                    std::ostringstream oss;
                    oss << "[main volume transport error]: " << desired_density << ", " << reached_density << ", " << optical_step << ", " << channel << ", " << iteration_mask << ", " << current_flight_distance << ", " << optical_depth;
                    Log(Warn, "%s", oss.str());
            }

            needs_intersection &= !active_medium;

            escaped_medium = active_medium && !mi.is_valid();
            active_medium &= mi.is_valid();

            act_medium_scatter |= active_medium;

            masked(depth, act_medium_scatter) += 1;
            masked(throughput, act_medium_scatter) *= local_ss;

            masked(ray.o,    act_medium_scatter) = ray(mi.t);
            masked(ray.mint, act_medium_scatter) = 0.f;
            masked(si.t,     act_medium_scatter) = si.t - mi.t;

            // {
            //     std::ostringstream oss;
            //     oss << "[main volume transport exit]: " << index_spectrum(optical_depth, channel) << ", " << mi.is_valid() << ", " << act_medium_scatter << ", " << current_flight_distance << ", " << mi.t << ", " << si.t;
            //     Log(Debug, "%s", oss.str());
            // }
        }

        // Dont estimate lighting if we exceeded number of bounces
        active &= depth < (uint32_t) m_max_depth;
        active_medium &= active;

        if (any_or<true>(act_medium_scatter)) {
           PhaseFunctionContext phase_ctx(sampler);
           auto phase = mi.medium->phase_function();

           // --------------------- Emitter sampling ---------------------
           Mask sample_emitters = mi.medium->use_emitter_sampling();
           valid_ray |= act_medium_scatter;
           specular_chain &= !act_medium_scatter;
           specular_chain |= act_medium_scatter && !sample_emitters;

           Mask active_e = act_medium_scatter && sample_emitters;
           if (any_or<true>(active_e)) {
               auto [emitted, ds] = sample_emitter(mi, true, scene, sampler, medium, channel, active_e);
               Float phase_val = phase->eval(phase_ctx, mi, ds.d, active_e);
               masked(result, active_e) += throughput * phase_val * emitted;
           }

           // ------------------ Phase function sampling -----------------
           masked(phase, !act_medium_scatter) = nullptr;
           auto [wo, phase_pdf] = phase->sample(phase_ctx, mi, sampler->next_2d(act_medium_scatter), act_medium_scatter);
           Ray3f new_ray  = mi.spawn_ray(wo);
           new_ray.mint = 0.0f;
           masked(ray, act_medium_scatter) = new_ray;
           needs_intersection |= act_medium_scatter;
        }

        // --------------------- Surface Interactions ---------------------
        active_surface |= escaped_medium;
        Mask intersect = active_surface && needs_intersection;
        if (any_or<true>(intersect))
            masked(si, intersect) = scene->ray_intersect(ray, intersect);

        if (any_or<true>(active_surface)) {
            // ---------------- Intersection with emitters ----------------
            EmitterPtr emitter = si.emitter(scene);
            Mask use_emitter_contribution = active_surface && specular_chain && neq(emitter, nullptr);
            if (any_or<true>(use_emitter_contribution)) {
                masked(result, use_emitter_contribution) += throughput * emitter->eval(si, use_emitter_contribution);
            }
        }
        active_surface &= si.is_valid();
        if (any_or<true>(active_surface)) {
            // --------------------- Emitter sampling ---------------------
            BSDFContext ctx;
            BSDFPtr bsdf  = si.bsdf(ray);
            Mask active_e = active_surface && has_flag(bsdf->flags(), BSDFFlags::Smooth) && (depth + 1 < (uint32_t) m_max_depth);

            if (likely(any_or<true>(active_e))) {
                auto [emitted, ds] = sample_emitter(si, false, scene, sampler, medium, channel, active_e);

                // Query the BSDF for that emitter-sampled direction
                Vector3f wo       = si.to_local(ds.d);
                Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Determine probability of having sampled that same
                // direction using BSDF sampling.
                Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_e);
                masked(result, active_e) += throughput * bsdf_val * mis_weight(ds.pdf, select(ds.delta, 0.f, bsdf_pdf)) * emitted;

            }

            // ----------------------- BSDF sampling ----------------------
            auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active_surface),
                                                sampler->next_2d(active_surface), active_surface);
            bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

            masked(throughput, active_surface) *= bsdf_val;

            masked(eta, active_surface) *= bs.eta;

            Ray bsdf_ray                = si.spawn_ray(si.to_world(bs.wo));
            masked(ray, active_surface) = bsdf_ray;
            needs_intersection |= active_surface;

            Mask non_null_bsdf = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Null);
            masked(depth, non_null_bsdf) += 1;

            valid_ray |= non_null_bsdf;
            specular_chain |= non_null_bsdf && has_flag(bs.sampled_type, BSDFFlags::Delta);
            specular_chain &= !(active_surface && has_flag(bs.sampled_type, BSDFFlags::Smooth));

            Mask add_emitter = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Delta) &&
                                any(neq(depolarize(throughput), 0.f)) && (depth < (uint32_t) m_max_depth);
            act_null_scatter |= active_surface && has_flag(bs.sampled_type, BSDFFlags::Null);

            // Intersect the indirect ray against the scene
            Mask intersect2 = active_surface && needs_intersection && add_emitter;
            SurfaceInteraction3f si_new = si;
            if (any_or<true>(intersect2))
                masked(si_new, intersect2) = scene->ray_intersect(ray, intersect2);
            needs_intersection &= !intersect2;

            auto [emitted, emitter_pdf] = evaluate_direct_light(si, scene, sampler,
                                                                medium, ray, si_new, channel, add_emitter);
            result += select(add_emitter && neq(emitter_pdf, 0),
                            mis_weight(bs.pdf, emitter_pdf) * throughput * emitted, 0.0f);

            Mask has_medium_trans            = active_surface && si.is_medium_transition();
            masked(medium, has_medium_trans) = si.target_medium(ray.d);

            masked(si, intersect2) = si_new;
        }
        active &= (active_surface | active_medium);
    }
    return { result, valid_ray };
}


/// Samples an emitter in the scene and evaluates it's attenuated contribution
std::tuple<Spectrum, DirectionSample3f>
sample_emitter(const Interaction3f &ref_interaction, Mask is_medium_interaction, const Scene *scene,
                        Sampler *sampler, MediumPtr medium, UInt32 channel, Mask active) const {
    using EmitterPtr = replace_scalar_t<Float, const Emitter *>;
    Spectrum transmittance(1.0f);

    auto [ds, emitter_val] = scene->sample_emitter_direction(ref_interaction, sampler->next_2d(active), false, active);
    masked(emitter_val, eq(ds.pdf, 0.f)) = 0.f;
    active &= neq(ds.pdf, 0.f);

    if (none_or<false>(active)) {
        return { emitter_val, ds };
    }

    Ray3f ray = ref_interaction.spawn_ray(ds.d);
    masked(ray.mint, is_medium_interaction) = 0.f;

    Float total_dist = 0.f;
    SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
    si.t = math::Infinity<Float>;
    Mask needs_intersection = true;
    while (any(active)) {
        Float remaining_dist = ds.dist * (1.f - math::ShadowEpsilon<Float>) - total_dist;
        ray.maxt = remaining_dist;
        active &= remaining_dist > 0.f;
        if (none(active))
            break;

        Mask escaped_medium = false;
        Mask active_medium  = active && neq(medium, nullptr);
        Mask active_surface = active && !active_medium;

        if (any_or<true>(active_medium)) {
            // Get raymarching interaction
            auto mi = sample_raymarched_interaction(ray, medium, channel, active_medium);
            masked(ray.maxt, active_medium && medium->is_homogeneous()) = mi.maxt;
            Mask intersect = needs_intersection && active_medium;

            if (any_or<true>(intersect)) {
                masked(si, intersect) = scene->ray_intersect(ray, intersect);
            }
            
            needs_intersection &= !intersect;

            // Get maximum flight distance of ray
            // If there is a surface then we should target that instead of the throughput limit
            Float max_flight_distance     = min(remaining_dist, min(si.t, mi.maxt)) - mi.mint;
            Float current_flight_distance = max(mi.t - mi.mint, 0.f);
            Float dt                      = m_volume_step_size;

            // Instantiate masks that track which rays are able to continue marching
            Mask iteration_mask = active_medium;
            masked(mi.t, eq(max_flight_distance, math::Infinity<Float>)) = math::Infinity<Float>;
            iteration_mask &= mi.is_valid() && current_flight_distance < max_flight_distance;

            // Instantiate tracking of optical depth, this will be used to estimate throughput
            Spectrum optical_depth(0.f), optical_step(0.f);

            // Create variables for local interaction parameters
            auto [local_ss, local_sn, local_st] = medium->get_scattering_coefficients(mi, iteration_mask);

            // Initialise step size
            // For inhomogeneous media we will use RK45 to do the integration
            // If the medium is homogeneous, we can just skip ahead
            masked(dt, iteration_mask) = max_flight_distance - current_flight_distance;
            masked(dt, iteration_mask && !medium->is_homogeneous()) = min(dt, m_volume_step_size);

            // Skip integration for homogeneous media
            masked(current_flight_distance, iteration_mask && medium->is_homogeneous()) += dt;
            masked(optical_depth, iteration_mask) += dt * local_st;
            masked(mi.t, iteration_mask && medium->is_homogeneous()) += dt;
            iteration_mask &= !medium->is_homogeneous();

            // RK45 parameters
            //     Optical Depth Function
            std::function<Spectrum(Float, Mask)> df_opt = [&ray, &mi, &medium, &current_flight_distance](Float dt_in, Mask active) {
                masked(mi.p, active) = ray(current_flight_distance + dt_in + mi.mint);
                auto [local_ss, local_sn, local_st] = medium->get_scattering_coefficients(mi, active);
                return local_st;
            };

            // {
            //     std::ostringstream oss;
            //     oss << "[emitter transport init]: " << iteration_mask << ", " << remaining_dist << ", " << max_flight_distance << ", " << current_flight_distance << ", " << mi.t << ", " << si.t << ", " << mi.mint << ", " << mi.maxt;
            //     Log(Debug, "%s", oss.str());
            // }

            while (any(iteration_mask)) {
                MTS_MASKED_FUNCTION(ProfilerPhase::MediumRaymarch, iteration_mask);
            
                auto [next_depth, curr_dt, next_dt] = integration_step(df_opt, dt, iteration_mask, m_use_adaptive_sampling);
                masked(optical_step, iteration_mask) = next_depth;

                // Update accumulators and ray position
                masked(current_flight_distance, iteration_mask) += dt;
                masked(optical_depth, iteration_mask) += optical_step;
                masked(mi.t, iteration_mask) += dt;
                masked(mi.p, iteration_mask) = ray(current_flight_distance + mi.mint);

                // {
                //     std::ostringstream oss;
                //     oss << "[emitter transport iter]: " << iteration_mask << ", " << dt << ", " << optical_depth << ", " << current_flight_distance;
                //     Log(Debug, "%s", oss.str());
                // }

                // Update step size for next iteration
                // If the volume is inhomogeneous, we use the smaller of the adaptive step or the remaining distance
                // If the volume is homogeneous, then next_dt is infinite and therefore we either
                //     terminate the iteration (current_flight_distance == max_flight_distance)
                //     or we take a step to close the gap between max_flight_distance and current_flight_distance
                if (m_use_adaptive_sampling) {
                    masked(dt, iteration_mask) = min(max_flight_distance - current_flight_distance, next_dt);
                } else {
                    masked(dt, iteration_mask) = min(max_flight_distance - current_flight_distance, m_volume_step_size);
                }

                // Update iteration mask
                // Marching should end when either throughput falls below sampled throughput or we exceed distance to the surface
                iteration_mask &= (current_flight_distance < max_flight_distance - math::RayEpsilon<Float>);
            }

            Spectrum tr = exp(-optical_depth);

            masked(mi.t, active_medium && mi.t >= max_flight_distance) = math::Infinity<Float>;
            masked(mi.t, active_medium && mi.t >= si.t) = math::Infinity<Float>;

            masked(transmittance, active_medium) *= tr;

            // Handle exceeding the maximum distance by medium sampling
            masked(total_dist, active_medium && (mi.t > remaining_dist) && mi.is_valid()) = ds.dist;
            masked(mi.t, active_medium && (mi.t > remaining_dist)) = math::Infinity<Float>;
            
            escaped_medium = active_medium && !mi.is_valid();
            active_medium &= mi.is_valid();

            masked(total_dist, active_medium) += mi.t;

            masked(ray.o, active_medium)    = ray(mi.t);
            masked(ray.mint, active_medium) = 0.f;
            masked(si.t, active_medium)     = si.t - mi.t;

            // {
            //     std::ostringstream oss;
            //     oss << "[emitter transport exit]: " << optical_depth << ", " << transmittance << ", " << exp(-optical_depth) << ", " << current_flight_distance << ", " << mi.t << ", " << si.t;
            //     Log(Debug, "%s", oss.str());
            // }
        }

        // Handle interactions with surfaces
        Mask intersect = active_surface && needs_intersection;
        if (any_or<true>(intersect))
            masked(si, intersect)    = scene->ray_intersect(ray, intersect);
        needs_intersection &= !intersect;
        active_surface |= escaped_medium;
        masked(total_dist, active_surface) += si.t;

        active_surface &= si.is_valid() && active && !active_medium;
        if (any_or<true>(active_surface)) {
            auto bsdf         = si.bsdf(ray);
            Spectrum bsdf_val = bsdf->eval_null_transmission(si, active_surface);
            bsdf_val = si.to_world_mueller(bsdf_val, si.wi, si.wi);
            masked(transmittance, active_surface) *= bsdf_val;
        }

        // Update the ray with new origin & t parameter
        masked(ray, active_surface) = si.spawn_ray(ray.d);
        ray.maxt = remaining_dist;
        needs_intersection |= active_surface;

        // Continue tracing through scene if non-zero weights exist
        active &= (active_medium || active_surface) && any(neq(depolarize(transmittance), 0.f));

        // If a medium transition is taking place: Update the medium pointer
        Mask has_medium_trans = active_surface && si.is_medium_transition();
        if (any_or<true>(has_medium_trans)) {
            masked(medium, has_medium_trans) = si.target_medium(ray.d);
        }
    }
    return { emitter_val * transmittance, ds };
}


std::pair<Spectrum, Float>
evaluate_direct_light(const Interaction3f &ref_interaction, const Scene *scene,
                        Sampler *sampler, MediumPtr medium, Ray3f ray,
                        const SurfaceInteraction3f &si_ray,
                        UInt32 channel, Mask active) const {
    using EmitterPtr = replace_scalar_t<Float, const Emitter *>;

    Spectrum emitter_val(0.0f);

    // Assumes the ray was alread intersected to compute si_ray before calling this method
    Mask needs_intersection = false, use_adaptive_sampling = m_use_adaptive_sampling;

    Spectrum transmittance(1.0f);
    Float emitter_pdf(0.0f);
    SurfaceInteraction3f si = si_ray;
    while (any(active)) {
        Mask escaped_medium = false;
        Mask active_medium  = active && neq(medium, nullptr);
        Mask active_surface = active && !active_medium;
        SurfaceInteraction3f si_medium = zero<SurfaceInteraction3f>();

        if (any_or<true>(active_medium)) {
            auto mi = sample_raymarched_interaction(ray, medium, channel, active_medium);
            masked(ray.maxt, active_medium && medium->is_homogeneous()) = mi.maxt;
            Mask intersect = needs_intersection && active_medium;

            if (any_or<true>(intersect)) {
                masked(si, intersect) = scene->ray_intersect(ray, intersect);
            }
            
            needs_intersection &= !intersect;

            // Get maximum flight distance of ray
            // If there is a surface then we should target that instead of the throughput limit
            Float max_flight_distance     = min(si.t, mi.maxt) - mi.mint;
            Float current_flight_distance = max(mi.t - mi.mint, 0.f);
            Float dt                      = m_volume_step_size;

            // Instantiate masks that track which rays are able to continue marching
            Mask iteration_mask = active_medium;
            masked(mi.t, eq(max_flight_distance, math::Infinity<Float>)) = math::Infinity<Float>;
            iteration_mask &= mi.is_valid() && current_flight_distance < max_flight_distance;

            // Instantiate tracking of optical depth, this will be used to estimate throughput
            Spectrum optical_depth(0.f), optical_step(0.f);

            // Create variables for local interaction parameters
            auto [local_ss, local_sn, local_st] = medium->get_scattering_coefficients(mi, iteration_mask);

            // Initialise step size
            // For inhomogeneous media we will use RK45 to do the integration
            // If the medium is homogeneous, we can just skip ahead
            masked(dt, iteration_mask) = max_flight_distance - current_flight_distance;
            masked(dt, iteration_mask && !medium->is_homogeneous()) = min(dt, m_volume_step_size);

            // Skip integration for homogeneous media
            masked(current_flight_distance, iteration_mask && medium->is_homogeneous()) += dt;
            masked(optical_depth, iteration_mask) += dt * local_st;
            masked(mi.t, iteration_mask && medium->is_homogeneous()) += dt;
            iteration_mask &= !medium->is_homogeneous();

            // RK45 parameters
            //     Optical Depth Function
            std::function<Spectrum(Float, Mask)> df_opt = [&ray, &mi, &medium, &current_flight_distance](Float dt_in, Mask active) {
                masked(mi.p, active) = ray(current_flight_distance + dt_in + mi.mint);
                auto [local_ss, local_sn, local_st] = medium->get_scattering_coefficients(mi, active);
                return local_st;
            };

            // {
            //     std::ostringstream oss;
            //     oss << "[direct transport init]: " << iteration_mask << ", " << max_flight_distance << ", " << current_flight_distance << ", " << mi.t << ", " << si.t << ", " << mi.mint << ", " << mi.maxt;
            //     Log(Debug, "%s", oss.str());
            // }

            while (any(iteration_mask)) {
                MTS_MASKED_FUNCTION(ProfilerPhase::MediumRaymarch, iteration_mask);
                auto [next_depth, curr_dt, next_dt] = integration_step(df_opt, dt, iteration_mask, use_adaptive_sampling);
                masked(optical_step, iteration_mask) = next_depth;

                // Update accumulators and ray position
                masked(current_flight_distance, iteration_mask) += dt;
                masked(optical_depth, iteration_mask) += optical_step;
                masked(mi.t, iteration_mask) += dt;
                masked(mi.p, iteration_mask) = ray(current_flight_distance + mi.mint);

                // {
                //     std::ostringstream oss;
                //     oss << "[direct transport iter]: " << iteration_mask << ", " << dt << ", " << optical_depth << ", " << current_flight_distance;
                //     Log(Debug, "%s", oss.str());
                // }

                // Update step size for next iteration
                // If the volume is inhomogeneous, we use the smaller of the adaptive step or the remaining distance
                // If the volume is homogeneous, then next_dt is infinite and therefore we either
                //     terminate the iteration (current_flight_distance == max_flight_distance)
                //     or we take a step to close the gap between max_flight_distance and current_flight_distance
                if (m_use_adaptive_sampling) {
                    masked(dt, iteration_mask) = min(max_flight_distance - current_flight_distance, next_dt);
                } else {
                    masked(dt, iteration_mask) = min(max_flight_distance - current_flight_distance, m_volume_step_size);
                }

                // Update iteration mask
                // Marching should end when either throughput falls below sampled throughput or we exceed distance to the surface
                iteration_mask &= (current_flight_distance < max_flight_distance - math::RayEpsilon<Float>);
            }

            Spectrum tr = exp(-optical_depth);

            masked(mi.t, active_medium && mi.t >= max_flight_distance) = math::Infinity<Float>;
            masked(mi.t, active_medium && mi.t >= si.t) = math::Infinity<Float>;

            masked(transmittance, active_medium) *= tr;

            needs_intersection &= !active_medium;
            
            escaped_medium = active_medium && !mi.is_valid();
            active_medium &= mi.is_valid();

            // {
            //     std::ostringstream oss;
            //     oss << "[direct transport exit]: " << exp(-optical_depth) << ", " << current_flight_distance << ", " << mi.t << ", " << si.t;
            //     Log(Debug, "%s", oss.str());
            // }

            masked(ray.o, active_medium)    = ray(mi.t);
            masked(ray.mint, active_medium) = 0.f;
            masked(si.t, active_medium)     = si.t - mi.t;
        }

        // Handle interactions with surfaces
        Mask intersect = active_surface && needs_intersection;
        masked(si, intersect)    = scene->ray_intersect(ray, intersect);
        needs_intersection &= !intersect;
        active_surface |= escaped_medium;

        // Check if we hit an emitter and add illumination if needed
        EmitterPtr emitter = si.emitter(scene, active_surface);
        Mask emitter_hit   = neq(emitter, nullptr) && active_surface;
        if (any_or<true>(emitter_hit)) {
            DirectionSample3f ds(si, ref_interaction);
            ds.object                        = emitter;
            masked(emitter_val, emitter_hit) = emitter->eval(si, emitter_hit);
            masked(emitter_pdf, emitter_hit) = scene->pdf_emitter_direction(ref_interaction, ds, emitter_hit);
            active &= !emitter_hit; // disable lanes which found an emitter
            active_surface &= active;
            active_medium &= active;
        }

        active_surface &= si.is_valid() && !active_medium;
        if (any_or<true>(active_surface)) {
            auto bsdf         = si.bsdf(ray);
            Spectrum bsdf_val = bsdf->eval_null_transmission(si, active_surface);
            bsdf_val = si.to_world_mueller(bsdf_val, si.wi, si.wi);

            masked(transmittance, active_surface) *= bsdf_val;
        }

        // Update the ray with new origin & t parameter
        masked(ray, active_surface) = si.spawn_ray(ray.d);
        needs_intersection |= active_surface;

        // Continue tracing through scene if non-zero weights exist
        active &= (active_medium || active_surface) && any(neq(depolarize(transmittance), 0.f));

        // If a medium transition is taking place: Update the medium pointer
        Mask has_medium_trans = active_surface && si.is_medium_transition();
        if (any_or<true>(has_medium_trans)) {
            masked(medium, has_medium_trans) = si.target_medium(ray.d);
        }
    }
    return { transmittance * emitter_val, emitter_pdf };
}


//! @}
// =============================================================

std::string to_string() const override {
    return tfm::format("RaymarchingSimplePathIntegrator[\n"
                        "  volume_step_size = %g"
                        "  relative_tolerance = %g"
                        "  absolute_tolerance = %g"
                        "  adaptive_stepping = %s"
                        "  max_depth = %i,\n"
                        "  rr_depth = %i\n"
                        "]",
                        m_volume_step_size, m_rtol, m_atol, (m_use_adaptive_sampling ? "true" : "false"), m_max_depth, m_rr_depth);
}

Float mis_weight(Float pdf_a, Float pdf_b) const {
    pdf_a *= pdf_a;
    pdf_b *= pdf_b;
    return select(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), Float(0.0f));
};

MTS_DECLARE_CLASS()

protected:
float m_volume_step_size, m_rtol, m_atol;
int m_stratified_samples;
bool m_use_adaptive_sampling, m_use_bisection;
};

MTS_IMPLEMENT_CLASS_VARIANT(RaymarchingPathIntegrator, MonteCarloIntegrator);
MTS_EXPORT_PLUGIN(RaymarchingPathIntegrator, "Raymarching Path Tracer integrator");
NAMESPACE_END(mitsuba)