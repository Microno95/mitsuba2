#include <enoki/stl.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/records.h>
#include <random>

NAMESPACE_BEGIN(mitsuba)

// Forward declaration of specialized integrator
template <typename Float, typename Spectrum, bool SpectralMis>
class RaymarchingMisIntegratorImpl;

/**!

.. _integrator-volpath:

Volumetric path tracer with null scattering (:monosp:`volpath`)
---------------------------------------------------------------

.. todo:: Not documented yet.
*/
template <typename Float, typename Spectrum>
class RaymarchingMisPathIntegrator final
    : public MonteCarloIntegrator<Float, Spectrum> {

public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth,
                    m_hide_emitters)
    MTS_IMPORT_TYPES(Scene, Sampler, Emitter, EmitterPtr, BSDF, BSDFPtr, Medium,
                     MediumPtr, PhaseFunctionContext)

    RaymarchingMisPathIntegrator(const Properties &props) : Base(props) {
        m_use_spectral_mis = props.bool_("use_spectral_mis", true);
        m_props            = props;
    }

    template <bool SpectralMis>
    using Impl = RaymarchingMisIntegratorImpl<Float, Spectrum, SpectralMis>;

    std::vector<ref<Object>> expand() const override {
        ref<Object> result;
        if (m_use_spectral_mis)
            result = (Object *) new Impl<true>(m_props);
        else
            result = (Object *) new Impl<false>(m_props);
        return { result };
    }
    MTS_DECLARE_CLASS()

protected:
    Properties m_props;
    bool m_use_spectral_mis;
};

template <typename Float, typename Spectrum, bool SpectralMis>
class RaymarchingMisIntegratorImpl final
    : public MonteCarloIntegrator<Float, Spectrum> {

public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth,
                    m_hide_emitters)
    MTS_IMPORT_TYPES(Scene, Sampler, Emitter, EmitterPtr, BSDF, BSDFPtr, Medium,
                     MediumPtr, PhaseFunctionContext)

    using WeightMatrix =
        std::conditional_t<SpectralMis,
                           Matrix<Float, array_size_v<UnpolarizedSpectrum>>,
                           UnpolarizedSpectrum>;

    RaymarchingMisIntegratorImpl(const Properties &props) : Base(props) {
        m_volume_step_size   = props.float_("volume_step_size", 0.025f);
        int strat_sample_init = props.int_("stratified_samples", 1);
        if (strat_sample_init <= 0) {
            std::stringstream oss;
            oss << "stratified_samples cannot be <= 0, was given " << strat_sample_init << ", set stratified_samples to 1";
            Log(Warn, "%s", oss.str());
        }
        m_stratified_samples    = std::max(strat_sample_init, 1);
        if (props.has_property("tolerance") && !props.has_property("absolute_tolerance") && !props.has_property("relative_tolerance")) {
            m_atol = props.float_("tolerance");
            m_rtol = m_atol;
        } else {
            m_atol = props.float_("absolute_tolerance", 1e-5f);
            m_rtol = props.float_("relative_tolerance", 1e-5f);
        }
        m_use_adaptive_sampling = props.bool_("adaptive_stepping", false);
        m_use_bisection         = props.bool_("exact_termination", true);
    }

    MTS_INLINE
    Float index_spectrum(const UnpolarizedSpectrum &spec,
                         const UInt32 &idx) const {
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

        // if (any(!enoki::isfinite(mint) && active)) {
        //     std::ostringstream oss;
        //     oss << "[medium interaction error]: " << ray << std::endl << medium << std::endl << channel << std::endl << active << std::endl;
        //     Log(Error, "%s", oss.str());
        // }

        // Sampling based on infinite homogeneous medium assumption
        mi.uniformly_sampled = active;
        mi.m            = 1.f;
        Float sampled_t = mint;
        Mask valid_mi   = active && (sampled_t <= maxt);
        mi.t            = select(valid_mi, sampled_t, math::Infinity<Float>);
        mi.p            = ray(sampled_t);
        mi.medium       = medium;
        mi.mint         = mint;
        mi.maxt         = maxt;
        std::tie(mi.sigma_s, mi.sigma_n, mi.sigma_t) = medium->get_scattering_coefficients(mi, active);
        mi.combined_extinction = medium->get_combined_extinction(mi, active);
        mi.radiance     = medium->get_radiance(mi, active);
        return mi;
    }

    std::tuple<Spectrum, Float, Float>
    integration_step(std::function<Spectrum(Float, Mask)>& df, Float dt, Mask active, Mask use_adaptive_sampling) const {
        MTS_MASKED_FUNCTION(ProfilerPhase::MediumRaymarch, active);
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

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium *initial_medium,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);
        if constexpr (is_polarized_v<Spectrum>) {
            Throw("This integrator currently does not support polarized mode!");
        }

        // If there is an environment emitter and emitters are visible: all rays
        // will be valid Otherwise, it will depend on whether a valid
        // interaction is sampled
        Mask valid_ray = !m_hide_emitters && neq(scene->environment(), nullptr);

        // For now, don't use ray differentials
        Ray3f ray = ray_;

        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);

        Spectrum result(0.f);

        MediumPtr medium       = initial_medium;
        MediumInteraction3f mi = zero<MediumInteraction3f>();
        mi.t                   = math::Infinity<Float>;

        Mask specular_chain       = active && !m_hide_emitters;
        UInt32 depth              = 0;
        WeightMatrix p_over_f     = full<WeightMatrix>(1.f);
        WeightMatrix p_over_f_nee = full<WeightMatrix>(1.f);

        UInt32 channel = 0;
        if (is_rgb_v<Spectrum>) {
            uint32_t n_channels = (uint32_t) array_size_v<Spectrum>;
            channel = (UInt32) min(sampler->next_1d(active) * n_channels, n_channels - 1);
        }

        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.t                    = math::Infinity<Float>;

        Mask needs_intersection = true, last_event_was_null = false;
        Interaction3f last_scatter_event = zero<Interaction3f>();
        last_scatter_event.t             = math::Infinity<Float>;
        for (int bounce = 0;; ++bounce) {
            // ----------------- Handle termination of paths ------------------

            // Russian roulette: try to keep path weights equal to one, while
            // accounting for the solid angle compression at refractive index
            // boundaries. Stop with at least some probability to avoid  getting
            // stuck (e.g. due to total internal reflection)
            Spectrum mis_throughput = mis_weight(p_over_f);
            Float q = min(hmax(depolarize(mis_throughput)) * sqr(eta), .95f);
            Mask perform_rr = active && !last_event_was_null && (depth > (uint32_t) m_rr_depth);
            active &= !(sampler->next_1d(active) >= q && perform_rr);
            update_weights(p_over_f, detach(q), 1.0f, channel, perform_rr);

            last_event_was_null = false;

            Mask exceeded_max_depth = depth >= (uint32_t) m_max_depth;
            active &= !exceeded_max_depth;
            active &= any(neq(depolarize(mis_weight(p_over_f)), 0.f));

            if (none(active))
                break;

            // ----------------------- Sampling the RTE -----------------------
            Mask active_medium  = active && neq(medium, nullptr);
            Mask active_surface = active && !active_medium;
            Mask act_medium_scatter = false, escaped_medium = false;

            // If the medium does not have a spectrally varying extinction,
            // we can perform a few optimizations to speed up rendering            
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
                Float max_throughput           = sampler->next_1d(active_medium);
                Float desired_density          = -enoki::log(max_throughput);

                // Instantiate masks that track which rays are able to continue marching
                Mask iteration_mask = active_medium;
                masked(mi.t, eq(max_flight_distance, math::Infinity<Float>)) = math::Infinity<Float>;
                iteration_mask &= mi.is_valid() && current_flight_distance < max_flight_distance;
                Mask reached_density = false, non_scattering_media = false; // iteration_mask && !medium->has_scattering();

                // Instantiate tracking of optical depth, this will be used to estimate throughput
                Spectrum optical_depth(0.f), optical_step(0.f), tr(1.0f), total_radiance(0.f);

                // Create variables for local interaction parameters
                auto [local_ss, local_sn, local_st] = medium->get_scattering_coefficients(mi, iteration_mask);
                Spectrum local_radiance = medium->get_radiance(mi, iteration_mask), accumulated_radiance = 0.f, intermediate_sum = 0.f;

                // Initialise step size
                // For inhomogeneous media we will use RK45 to do the integration
                // For inhomogeneous media we need to take integration steps, here we jitter the initial step and impose that a step is at least 1 ray epsilon
                masked(dt, iteration_mask && !medium->is_homogeneous()) = max(math::RayEpsilon<Float>, min(dt, m_volume_step_size) * sampler->next_1d(iteration_mask));
                // For homogeneous media we can simply sample the correct distance to achieve the desired density
                masked(dt, iteration_mask &&  medium->is_homogeneous() && !non_scattering_media) = min(dt, desired_density / index_spectrum(local_st, channel));

                // // If the medium is homogeneous, we don't have to integrate the radiance nor the optical depth
                // // Update accumulators and ray position
                // masked(current_flight_distance, medium->is_homogeneous()) += dt;
                // masked(optical_depth,           medium->is_homogeneous())  = dt * local_st;
                // masked(mi.t,                    medium->is_homogeneous()) += dt;
                // masked(mi.p,                    medium->is_homogeneous())  = ray(mi.mint + dt);
                // // If the medium has absorption/out-scattering, then the radiance is modulated by the transmittance
                // masked(accumulated_radiance, medium->is_homogeneous() &&  non_scattering_media &&  medium->has_absorption()) = (local_radiance / local_st) * (1 - exp(-optical_depth));
                // // If the medium has no absorption/out-scattering, then the radiance is just the linear sum of the radiating elements
                // masked(accumulated_radiance, medium->is_homogeneous() &&  non_scattering_media && !medium->has_absorption()) = local_radiance * max_flight_distance;
                // masked(accumulated_radiance, medium->is_homogeneous() && !non_scattering_media) = local_radiance;

                // iteration_mask &= !medium->is_homogeneous() || !non_scattering_media;

                // RK45 parameters
                //     Optical Depth Function
                std::function<Spectrum(Float, Mask)> df_opt = [&ray, &mi, &medium, &current_flight_distance](Float dt_in, Mask active) {
                    masked(mi.p, active) = ray(current_flight_distance + dt_in + mi.mint);
                    auto [local_ss, local_sn, local_st] = medium->get_scattering_coefficients(mi, active);
                    return local_st;
                };

                while (any(iteration_mask)) {           
                    auto [next_depth, curr_dt, next_dt] = integration_step(df_opt, dt, iteration_mask, m_use_adaptive_sampling);
                    masked(optical_step, iteration_mask) = next_depth;

                    // --------------------- Correction ---------------------- //
                    // Check if we exceed the desired density by taking this step
                    // If we do, find the point between this point and the last point that reaches the desired density
                    // The desired density and the obtained density may differ since we sample the density at the point we interpolate to
                    // And if the step size is larger than 2x the mean grid spacing then this can be quite different
                    Mask optical_depth_needs_correction = iteration_mask && (index_spectrum(optical_depth + optical_step, channel) >= desired_density) && !non_scattering_media;

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
                                auto [m_depth, m_dt, m_ndt] = integration_step(df_opt, half_step, not_found_depth, m_use_adaptive_sampling);
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
                            auto [corr_next_depth, corr_curr_dt, corr_next_dt]   = integration_step(df_opt, curr_dt * interp, optical_depth_needs_correction, m_use_adaptive_sampling);
                            masked(optical_step, optical_depth_needs_correction) = corr_next_depth;
                            masked(next_dt,      optical_depth_needs_correction) = corr_next_dt;
                            masked(curr_dt,      optical_depth_needs_correction) = corr_curr_dt;
                        }
                    }
                    // ------------------------------------------------------- //
                    masked(dt, iteration_mask) = curr_dt;

                    // Accumulate radiance on first half of step for non-scattering media
                    if (any_or<true>(non_scattering_media)) {
                        masked(local_radiance, non_scattering_media)    = medium->get_radiance(mi, non_scattering_media);
                        masked(intermediate_sum, non_scattering_media) += exp(-optical_depth) * local_radiance;
                    }

                    // Update accumulators and ray position
                    masked(current_flight_distance, iteration_mask) += dt;
                    masked(optical_depth, iteration_mask)           += optical_step;
                    masked(mi.t, iteration_mask)                    += dt;
                    masked(mi.p, iteration_mask)                     = ray(current_flight_distance + mi.mint);
                    Mask sample_radiance                             = optical_depth_needs_correction && (desired_density - index_spectrum(optical_depth, channel) < math::RayEpsilon<Float>);

                    // Sample Emission at the Integrate Point
                    if (any_or<true>(sample_radiance)) {
                        local_radiance                                    = medium->get_radiance(mi, sample_radiance);
                        masked(accumulated_radiance, sample_radiance)    += local_radiance;
                        reached_density |= sample_radiance;
                    }

                    // Accumulate radiance on second half of step for non-scattering media
                    // and add radiance to the accumulated radiance
                    if (any_or<true>(non_scattering_media)) {
                        masked(local_radiance, non_scattering_media)        = medium->get_radiance(mi, non_scattering_media);
                        masked(intermediate_sum, non_scattering_media)     += exp(-optical_depth) * local_radiance;
                        masked(accumulated_radiance, non_scattering_media) += intermediate_sum * dt * 0.5f;
                        intermediate_sum                                    = 0.f;
                    }

                    // Update step size for next iteration
                    // If the volume is inhomogeneous, we use the smaller of the adaptive step or the remaining distance
                    // If the volume is homogeneous, then next_dt is infinite and therefore we either
                    //     terminate the iteration (current_flight_distance == max_flight_distance)
                    //     or we take a step to close the gap between max_flight_distance and current_flight_distance
                    if (m_use_adaptive_sampling) {
                        masked(dt, iteration_mask) = max_flight_distance - current_flight_distance;
                        masked(dt, iteration_mask && !medium->is_homogeneous()) = min(dt, next_dt);
                        masked(dt, iteration_mask &&  medium->is_homogeneous() && !non_scattering_media) = min(dt, desired_density / index_spectrum(local_st, channel));
                    } else {
                        masked(dt, iteration_mask) = max_flight_distance - current_flight_distance;
                        masked(dt, iteration_mask && !medium->is_homogeneous()) = min(dt, m_volume_step_size);
                        masked(dt, iteration_mask &&  medium->is_homogeneous() && !non_scattering_media) = min(dt, desired_density / index_spectrum(local_st, channel));
                    }

                    // Update iteration mask
                    // Marching should end when either throughput falls below sampled throughput or we exceed distance to the surface
                    iteration_mask &= (current_flight_distance < max_flight_distance - math::RayEpsilon<Float>) && !reached_density;
                }

                masked(mi.p, active_medium) = ray(current_flight_distance + mi.mint);
                std::tie(local_ss, local_sn, local_st) = medium->get_scattering_coefficients(mi, active_medium);
                local_radiance = medium->get_radiance(mi, active_medium);
                tr = exp(-optical_depth);
                Spectrum path_pdf = select(mi.t < max_flight_distance, tr * local_st, tr);

                masked(result, active_medium &&  non_scattering_media) += mis_weight(p_over_f) * accumulated_radiance;

                update_weights(p_over_f,     1.f,    tr, channel, active_medium &&  non_scattering_media);
                update_weights(p_over_f_nee, 1.f,    tr, channel, active_medium &&  non_scattering_media);
                update_weights(p_over_f,     path_pdf, tr, channel, active_medium && !non_scattering_media);
                update_weights(p_over_f_nee, path_pdf, tr, channel, active_medium && !non_scattering_media);
                
                masked(result, active_medium && !non_scattering_media) += mis_weight(p_over_f) * accumulated_radiance;

                masked(mi.t, active_medium && mi.t >= max_flight_distance) = math::Infinity<Float>;
                masked(mi.t, active_medium && mi.t >= si.t)                = math::Infinity<Float>;

                needs_intersection &= !active_medium;

                escaped_medium = active_medium && !mi.is_valid();
                active_medium &= mi.is_valid();

                act_medium_scatter |= active_medium;

                // Count this as a bounce
                masked(depth, act_medium_scatter)      += 1;
                masked(last_scatter_event, act_medium_scatter) = mi;

                masked(ray.o,    act_medium_scatter) = ray(mi.t);
                masked(ray.mint, act_medium_scatter) = 0.f;
                masked(si.t,     act_medium_scatter) = si.t - mi.t;

                Mask sample_emitters = mi.medium->use_emitter_sampling();

                // Dont estimate lighting if we exceeded number of bounces
                active &= depth < (uint32_t) m_max_depth;
                act_medium_scatter &= active;
                specular_chain = specular_chain && !(act_medium_scatter && sample_emitters);

                if (any_or<true>(act_medium_scatter)) {
                    update_weights(p_over_f, 1.f, local_ss, channel, act_medium_scatter);

                    PhaseFunctionContext phase_ctx(sampler);
                    auto phase = mi.medium->phase_function();

                    // --------------------- Emitter sampling ---------------------
                    valid_ray |= act_medium_scatter;
                    Mask active_e = act_medium_scatter && sample_emitters;
                    if (any_or<true>(active_e)) {
                        auto [p_over_f_nee_end, p_over_f_end, emitted, ds] = sample_emitter(mi, true, scene, sampler, medium, p_over_f, channel, active_e);
                        Float phase_val = phase->eval(phase_ctx, mi, ds.d, active_e);
                        update_weights(p_over_f_nee_end, 1.0f, phase_val, channel, active_e);
                        update_weights(p_over_f_end, select(ds.delta, 0.f, phase_val), phase_val, channel, active_e);
                        masked(result, active_e) += mis_weight(p_over_f_nee_end, p_over_f_end) * emitted;
                    }

                    // In a real interaction: reset p_over_f_nee
                    masked(p_over_f_nee, act_medium_scatter) = p_over_f;

                    // ------------------ Phase function sampling -----------------
                    masked(phase, !act_medium_scatter) = nullptr;
                    auto [wo, phase_pdf]               = phase->sample(phase_ctx, mi, sampler->next_2d(act_medium_scatter), act_medium_scatter);
                    Ray3f new_ray                   = mi.spawn_ray(wo);
                    new_ray.mint                    = 0.0f;
                    masked(ray, act_medium_scatter) = new_ray;
                    needs_intersection |= act_medium_scatter;

                    update_weights(p_over_f, phase_pdf, phase_pdf, channel, act_medium_scatter);
                    update_weights(p_over_f_nee, 1.f, phase_pdf, channel, act_medium_scatter);
                }
            }

            // --------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium;
            Mask intersect = active_surface && needs_intersection;
            if (any_or<true>(intersect))
                masked(si, intersect) = scene->ray_intersect(ray, intersect);

            if (any_or<true>(active_surface)) {
                // ---------------- Intersection with emitters ----------------
                Mask ray_from_camera = active_surface && eq(depth, 0u);
                Mask count_direct    = ray_from_camera || specular_chain;
                EmitterPtr emitter   = si.emitter(scene);
                Mask active_e = active_surface && neq(emitter, nullptr) &&
                                !(eq(depth, 0u) && m_hide_emitters);
                if (any_or<true>(active_e)) {
                    if (any_or<true>(active_e && !count_direct)) {
                        // Get the PDF of sampling this emitter using next event
                        // estimation
                        DirectionSample3f ds(si, last_scatter_event);
                        ds.object         = emitter;
                        Float emitter_pdf = scene->pdf_emitter_direction(last_scatter_event, ds, active_e);
                        update_weights(p_over_f_nee, emitter_pdf, 1.f, channel, active_e);
                    }
                    Spectrum emitted = emitter->eval(si, active_e);
                    Spectrum contrib = select(count_direct, mis_weight(p_over_f) * emitted, mis_weight(p_over_f, p_over_f_nee) * emitted);
                    masked(result, active_e) += contrib;
                }
            }

            active_surface &= si.is_valid();
            if (any_or<true>(active_surface)) {

                // --------------------- Emitter sampling ---------------------
                BSDFContext ctx;
                BSDFPtr bsdf  = si.bsdf(ray);
                Mask active_e = active_surface && has_flag(bsdf->flags(), BSDFFlags::Smooth) && (depth + 1 < (uint32_t) m_max_depth);
                if (likely(any_or<true>(active_e))) {
                    auto [p_over_f_nee_end, p_over_f_end, emitted, ds] = sample_emitter(si, false, scene, sampler, medium, p_over_f, channel, active_e);
                    Vector3f wo_local = si.to_local(ds.d);
                    Spectrum bsdf_val = bsdf->eval(ctx, si, wo_local, active_e);
                    Float bsdf_pdf    = bsdf->pdf(ctx, si, wo_local, active_e);
                    update_weights(p_over_f_nee_end, 1.0f, depolarize(bsdf_val), channel, active_e);
                    update_weights(p_over_f_end, select(ds.delta, 0.f, bsdf_pdf), depolarize(bsdf_val), channel, active_e);
                    masked(result, active_e) += mis_weight(p_over_f_nee_end, p_over_f_end) * emitted;
                }

                // ----------------------- BSDF sampling ----------------------
                auto [bs, bsdf_weight] = bsdf->sample(ctx, si, sampler->next_1d(active_surface), sampler->next_2d(active_surface), active_surface);
                Mask invalid_bsdf_sample = active_surface && eq(bs.pdf, 0.f);
                active_surface &= bs.pdf > 0.f;
                masked(eta, active_surface) *= bs.eta;

                Ray bsdf_ray                = si.spawn_ray(si.to_world(bs.wo));
                masked(ray, active_surface) = bsdf_ray;
                needs_intersection |= active_surface;

                Mask non_null_bsdf = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Null);
                valid_ray |= non_null_bsdf || invalid_bsdf_sample;
                specular_chain |= non_null_bsdf && has_flag(bs.sampled_type, BSDFFlags::Delta);
                specular_chain &= !(active_surface && has_flag(bs.sampled_type, BSDFFlags::Smooth));
                masked(depth, non_null_bsdf) += 1;
                masked(last_scatter_event, non_null_bsdf) = si;

                // Update NEE weights only if the BSDF is not null
                masked(p_over_f_nee, non_null_bsdf) = p_over_f;
                update_weights(p_over_f, bs.pdf, depolarize(bsdf_weight * bs.pdf), channel, active_surface);
                update_weights(p_over_f_nee, 1.f, depolarize(bsdf_weight * bs.pdf), channel, non_null_bsdf);

                Mask has_medium_trans = active_surface && si.is_medium_transition();
                masked(medium, has_medium_trans) = si.target_medium(ray.d);
            }
            active &= (active_surface | active_medium);
        }

        return { result, valid_ray };
    }

    std::tuple<WeightMatrix, WeightMatrix, Spectrum, DirectionSample3f>
    sample_emitter(const Interaction3f &ref_interaction,
                   Mask is_medium_interaction, const Scene *scene,
                   Sampler *sampler, MediumPtr medium,
                   const WeightMatrix &p_over_f, UInt32 channel,
                   Mask active) const {
        using EmitterPtr          = replace_scalar_t<Float, const Emitter *>;
        WeightMatrix p_over_f_nee = p_over_f, p_over_f_uni = p_over_f;

        // Float vol_or_shape_emitter = sampler->next_1d();
        DirectionSample3f ds, ds_vol;
        // Float vol_prob = scene->get_volume_emitter_probability();
        Spectrum emitter_sample_weight, emitter_sample_weight_vol;
        // Mask is_vol_emitter = false; // vol_or_shape_emitter < vol_prob;

        std::tie(ds, emitter_sample_weight) = scene->sample_emitter_direction(ref_interaction, sampler->next_2d(active), false, active);
        // std::tie(ds_vol, emitter_sample_weight_vol) = scene->sample_volume_emitter_direction(ref_interaction, sampler->next_2d(active), false, active && is_vol_emitter, channel);
        
        // masked(ds, is_vol_emitter) = ds_vol;
        // masked(emitter_sample_weight, is_vol_emitter) = emitter_sample_weight_vol;

        Spectrum emitter_val                 = emitter_sample_weight * ds.pdf;
        masked(emitter_val, eq(ds.pdf, 0.f)) = 0.f;
        active &= neq(ds.pdf, 0.f);

        update_weights(p_over_f_nee, ds.pdf, 1.0f, channel, active);

        if (none_or<false>(active)) {
            return { p_over_f_nee, p_over_f_uni, emitter_val, ds };
        }

        Ray3f ray = ref_interaction.spawn_ray(ds.d);
        masked(ray.mint, is_medium_interaction) = 0.f;

        Float total_dist        = 0.f;
        SurfaceInteraction3f si = zero<SurfaceInteraction3f>();
        si.t                    = math::Infinity<Float>;

        Mask needs_intersection = true;
        while (any(active)) {
            Float remaining_dist = ds.dist * (1.f - math::ShadowEpsilon<Float>) -total_dist;
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

                while (any(iteration_mask)) {            
                    auto [next_depth, curr_dt, next_dt] = integration_step(df_opt, dt, iteration_mask, m_use_adaptive_sampling);
                    masked(optical_step, iteration_mask) = next_depth;

                    // Update accumulators and ray position
                    masked(current_flight_distance, iteration_mask) += dt;
                    masked(optical_depth, iteration_mask)           += optical_step;
                    masked(mi.t, iteration_mask)                    += dt;
                    masked(mi.p, iteration_mask)                     = ray(current_flight_distance + mi.mint);

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

                UnpolarizedSpectrum tr = exp(-optical_depth);

                masked(mi.t, active_medium && mi.t >= max_flight_distance) = math::Infinity<Float>;
                masked(mi.t, active_medium && mi.t >= si.t) = math::Infinity<Float>;

                update_weights(p_over_f_nee, 1.f, tr, channel, active_medium);
                update_weights(p_over_f_uni, 1.f, tr, channel, active_medium);
                
                escaped_medium = active_medium && !mi.is_valid();
                active_medium &= mi.is_valid();

                masked(total_dist, escaped_medium) += mi.t;

                masked(ray.o, active_medium)    = ray(mi.t);
                masked(ray.mint, active_medium) = 0.f;
                masked(si.t, active_medium)     = si.t - mi.t;
            }

            // Handle interactions with surfaces
            Mask intersect = active_surface && needs_intersection;
            if (any_or<true>(intersect))
                masked(si, intersect) = scene->ray_intersect(ray, intersect);
            active_surface |= escaped_medium;
            masked(total_dist, active_surface) += si.t;

            active_surface &= si.is_valid() && active && !active_medium;
            if (any_or<true>(active_surface)) {
                auto bsdf = si.bsdf(ray);
                Spectrum bsdf_val = bsdf->eval_null_transmission(si, active_surface);
                update_weights(p_over_f_nee, 1.0f, depolarize(bsdf_val), channel, active_surface);
                update_weights(p_over_f_uni, 1.0f, depolarize(bsdf_val), channel, active_surface);
            }

            masked(ray, active_surface) = si.spawn_ray(ray.d);
            ray.maxt                    = remaining_dist;
            needs_intersection |= active_surface;

            // Continue tracing through scene if non-zero weights exist
            if constexpr (SpectralMis)
                active &= (active_medium || active_surface) && any(neq(mis_weight(p_over_f_uni), 0.f));
            else
                active &= (active_medium || active_surface) && (any(neq(depolarize(p_over_f_uni), 0.f)) || any(neq(depolarize(p_over_f_nee), 0.f)));

            // If a medium transition is taking place: Update the medium pointer
            Mask has_medium_trans = active_surface && si.is_medium_transition();
            if (any_or<true>(has_medium_trans)) {
                masked(medium, has_medium_trans) = si.target_medium(ray.d);
            }
        }

        return { p_over_f_nee, p_over_f_uni, emitter_val, ds };
    }

    MTS_INLINE
    void update_weights(WeightMatrix &p_over_f, const UnpolarizedSpectrum &p,
                        const UnpolarizedSpectrum &f, UInt32 channel,
                        Mask active) const {
        // For two spectra p and f, computes all the ratios of the individual
        // components and multiplies them to the current values in p_over_f
        if constexpr (SpectralMis) {
            ENOKI_MARK_USED(channel);
            for (size_t i = 0; i < array_size_v<Spectrum>; ++i) {
                UnpolarizedSpectrum ratio = p / f.coeff(i);
                ratio = select(enoki::isfinite(ratio), ratio, 0.f);
                ratio *= p_over_f[i];
                masked(p_over_f[i], active) = select(neq(ratio, ratio), 0.f, ratio);
            }
        } else {
            // If we don't do spectral MIS: We need to use a specific channel of
            // the spectrum "p" as the PDF
            Float pdf  = index_spectrum(p, channel);
            auto ratio = p_over_f * (pdf / f);
            masked(p_over_f, active) = select(enoki::isfinite(ratio), ratio, 0.f);
        }
    }

    UnpolarizedSpectrum mis_weight(const WeightMatrix &p_over_f) const {
        if constexpr (SpectralMis) {
            constexpr size_t n = array_size_v<Spectrum>;
            UnpolarizedSpectrum weight(0.0f);
            for (size_t i = 0; i < n; ++i) {
                Float sum = hsum(p_over_f[i]);
                weight[i] = select(eq(sum, 0.f), 0.0f, n / sum);
            }
            return weight;
        } else {
            Mask invalid = eq(hmin(abs(p_over_f)), 0.f);
            return select(invalid, 0.f, 1.f / p_over_f);
        }
    }

    // returns MIS'd throughput/pdf of two full paths represented by p_over_f1
    // and p_over_f2
    UnpolarizedSpectrum mis_weight(const WeightMatrix &p_over_f1,
                                   const WeightMatrix &p_over_f2) const {
        UnpolarizedSpectrum weight(0.0f);
        if constexpr (SpectralMis) {
            constexpr size_t n = array_size_v<Spectrum>;
            auto sum_matrix    = p_over_f1 + p_over_f2;
            for (size_t i = 0; i < n; ++i) {
                Float sum = hsum(sum_matrix[i]);
                weight[i] = select(eq(sum, 0.f), 0.0f, n / sum);
            }
        } else {
            auto sum = p_over_f1 + p_over_f2;
            weight   = select(eq(hmin(abs(sum)), 0.f), 0.0f, 1.f / sum);
        }
        return weight;
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("RaymarchingMisPathIntegrator[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }

    MTS_DECLARE_CLASS()

    protected:
    float m_volume_step_size, m_rtol, m_atol;
    int m_stratified_samples;
    bool m_use_adaptive_sampling, m_use_bisection;
};

MTS_IMPLEMENT_CLASS_VARIANT(RaymarchingMisPathIntegrator,
                            MonteCarloIntegrator);
MTS_EXPORT_PLUGIN(RaymarchingMisPathIntegrator,
                  "Raymarching MIS Path Tracer integrator");

NAMESPACE_BEGIN(detail)
template <bool SpectralMis> constexpr const char *volpath_class_name() {
    if constexpr (SpectralMis) {
        return "Volpath_spectral_mis";
    } else {
        return "Volpath_no_spectral_mis";
    }
}
NAMESPACE_END(detail)

template <typename Float, typename Spectrum, bool SpectralMis>
Class *RaymarchingMisIntegratorImpl<Float, Spectrum, SpectralMis>::m_class =
    new Class(detail::volpath_class_name<SpectralMis>(), "MonteCarloIntegrator",
              ::mitsuba::detail::get_variant<Float, Spectrum>(), nullptr,
              nullptr);

template <typename Float, typename Spectrum, bool SpectralMis>
const Class *
RaymarchingMisIntegratorImpl<Float, Spectrum, SpectralMis>::class_() const {
    return m_class;
}

NAMESPACE_END(mitsuba)