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
        m_volume_step_size = props.float_("volume_step_size", 0.0625f);
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
        MediumInteraction3f mi;
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

        auto combined_extinction = medium->get_combined_extinction(mi, active);
        Float m                  = combined_extinction[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            masked(m, eq(channel, 1u)) = combined_extinction[1];
            masked(m, eq(channel, 2u)) = combined_extinction[2];
        } else {
            ENOKI_MARK_USED(channel);
        }

        auto radiant_emission = medium->get_radiance(mi, active);
        Float m2              = radiant_emission[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            masked(m2, eq(channel, 1u)) = radiant_emission[1];
            masked(m2, eq(channel, 2u)) = radiant_emission[2];
        } else {
            ENOKI_MARK_USED(channel);
        }

        Float sampled_t = mint + m_volume_step_size;
        Mask valid_mi   = active && (sampled_t <= maxt);
        mi.t            = select(valid_mi, sampled_t, math::Infinity<Float>);
        mi.p            = ray(sampled_t);
        mi.medium       = medium;
        mi.mint         = mint;
        std::tie(mi.sigma_s, mi.sigma_n, mi.sigma_t) =
            medium->get_scattering_coefficients(mi, valid_mi);
        mi.radiance            = medium->get_radiance(mi, valid_mi);
        mi.combined_extinction = combined_extinction;
        return mi;
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
        Mask specular_chain = active && !m_hide_emitters;
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

            /*if (any(select(active, throughput, 0.f) > 1.f)) {
                Throw("Throughput is greater than 1.f!");
            }*/

            active &= any(neq(depolarize(throughput), 0.f));
            Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
            Mask perform_rr = (depth > (uint32_t) m_rr_depth);
            active &= sampler->next_1d(active) < q || !perform_rr;
            masked(throughput, perform_rr) *= rcp(detach(q));

            /*if (any(select(active, throughput, 0.f) > 1.f)) {
                Throw("Throughput is greater than 1.f!");
            }*/

            Mask exceeded_max_depth = depth >= (uint32_t) m_max_depth;
            if (none(active) || all(exceeded_max_depth))
                break;

            // ----------------------- Sampling the RTE -----------------------
            Mask active_medium  = active && neq(medium, nullptr);
            Mask active_surface = active && !active_medium;
            Mask act_absorption = false, act_null_scatter = false, 
				 act_medium_scatter = false, escaped_medium = false;

            if (any_or<true>(active_medium)) {
                mi = sample_raymarched_interaction(ray, medium, channel, active_medium);
                masked(ray.maxt, active_medium && medium->is_homogeneous() && mi.is_valid()) = mi.t;
                Mask intersect = needs_intersection && active_medium;
                if (any_or<true>(intersect)) {
                    masked(si, intersect) = scene->ray_intersect(ray, intersect);
                }
                needs_intersection &= !active_medium;

                masked(mi.t, active_medium && (si.t < mi.t)) = math::Infinity<Float>;

                auto [tr, eps, eps_int, free_flight_pdf] = medium->eval_tr_eps_and_pdf(mi, si, active_medium);
                masked(result, active_medium) += throughput * eps_int;
                masked(throughput, active_medium) *= tr;

                masked(ray.o, active_medium)    = mi.p;
                masked(ray.mint, active_medium) = 0.f;
                masked(si.t, active_medium)     = si.t - mi.t;
                masked(depth, active_medium) += 1;

                escaped_medium      = active_medium && !mi.is_valid();
                active_medium      &= mi.is_valid();

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

                    /*if (any(!enoki::isfinite(result))) {
                        std::ostringstream oss;
                        oss << "[4] Invalid result value: [";
                        for (uint32_t i = 0; i < array_size_v<Spectrum>; ++i) {
                            oss << result.coeff(i);
                            if (i + 1 < array_size_v<Spectrum>)
                                oss << ", ";
                        }
                        oss << "]";
                        Log(Warn, "%s", oss.str());
                    }*/
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

                    /*if (any(!enoki::isfinite(result))) {
                        std::ostringstream oss;
                        oss << "[5] Invalid result value: [";
                        for (uint32_t i = 0; i < array_size_v<Spectrum>; ++i) {
                            oss << result.coeff(i);
                            if (i + 1 < array_size_v<Spectrum>)
                                oss << ", ";
                        }
                        oss << "]";
                        Log(Warn, "%s", oss.str());
                    }*/
                }

                // ----------------------- BSDF sampling ----------------------
                auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active_surface),
                                                   sampler->next_2d(active_surface), active_surface);
                bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

                masked(throughput, active_surface) *= bsdf_val;

               /* if (any(!enoki::isfinite(throughput))) {
                    std::ostringstream oss;
                    oss << "[6] Invalid throughput value: [";
                    for (uint32_t i = 0; i < array_size_v<Spectrum>; ++i) {
                        oss << throughput.coeff(i);
                        if (i + 1 < array_size_v<Spectrum>)
                            oss << ", ";
                    }
                    oss << "]";
                    Log(Warn, "%s", oss.str());
                }*/

                /*if (any(select(active, throughput, 0.f) > 1.f)) {
                    Throw("Throughput is greater than 1.f!");
                }*/

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
                /*if (any(!enoki::isfinite(result))) {
                    std::ostringstream oss;
                    oss << "[7] Invalid result value: [";
                    for (uint32_t i = 0; i < array_size_v<Spectrum>; ++i) {
                        oss << result.coeff(i);
                        if (i + 1 < array_size_v<Spectrum>)
                            oss << ", ";
                    }
                    oss << "]";
                    Log(Warn, "%s", oss.str());
                }*/

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
                auto mi = sample_raymarched_interaction(ray, medium, channel, active_medium);
                masked(ray.maxt, active_medium && medium->is_homogeneous() && mi.is_valid()) = min(mi.t, remaining_dist);
                Mask intersect = needs_intersection && active_medium;
                if (any_or<true>(intersect))
                    masked(si, intersect) = scene->ray_intersect(ray, intersect);

                masked(mi.t, active_medium && (si.t < mi.t)) = si.t;
                needs_intersection &= !active_medium;

                Mask is_spectral = medium->has_spectral_extinction() && active_medium;
                Mask not_spectral = !is_spectral && active_medium;
                Float t = min(remaining_dist, min(mi.t, si.t)) - mi.mint;

                if (any_or<true>(is_spectral)) {
                    UnpolarizedSpectrum tr  = exp(-t * mi.combined_extinction);
                    masked(transmittance, is_spectral) *= tr;
                }

                // Handle exceeding the maximum distance by medium sampling
                masked(total_dist, active_medium && (mi.t > remaining_dist) && mi.is_valid()) = ds.dist;
                masked(mi.t, active_medium && (mi.t > remaining_dist)) = math::Infinity<Float>;

                masked(total_dist, active_medium) += mi.mint + t;
                masked(ray.o, active_medium)    = ray(t);
                masked(ray.mint, active_medium) = 0.f;
                masked(si.t, active_medium && mi.is_valid()) = si.t - t;

                escaped_medium = active_medium && !mi.is_valid();
                active_medium &= mi.is_valid();
                is_spectral &= active_medium;
                not_spectral &= active_medium;
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
                /*if (any(!enoki::isfinite(transmittance))) {
                    std::ostringstream oss;
                    oss << "[5.3] Invalid transmittance value: [";
                    for (uint32_t i = 0; i < array_size_v<Spectrum>; ++i) {
                        oss << transmittance.coeff(i);
                        if (i + 1 < array_size_v<Spectrum>)
                            oss << ", ";
                    }
                    oss << "]";
                    Log(Warn, "%s", oss.str());
                }*/
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
        Mask needs_intersection = false;

        Spectrum transmittance(1.0f);
        Float emitter_pdf(0.0f);
        SurfaceInteraction3f si = si_ray;
        while (any(active)) {
            Mask escaped_medium = false;
            Mask active_medium  = active && neq(medium, nullptr);
            Mask active_surface = active && !active_medium;
            SurfaceInteraction3f si_medium;
            if (any_or<true>(active_medium)) {
                auto mi = sample_raymarched_interaction(ray, medium, channel, active_medium);
                masked(ray.maxt, active_medium && medium->is_homogeneous() && mi.is_valid()) = mi.t;
                Mask intersect = needs_intersection && active_medium;
                if (any_or<true>(intersect))
                    masked(si, intersect) = scene->ray_intersect(ray, intersect);

                masked(mi.t, active_medium && (si.t < mi.t)) = si.t;

                Mask is_spectral = medium->has_spectral_extinction() && active_medium;
                Mask not_spectral = !is_spectral && active_medium;
                if (any_or<true>(is_spectral)) {
                    auto [tr, eps, eps_int, free_flight_pdf] = medium->eval_tr_eps_and_pdf(mi, si, is_spectral);
                    masked(transmittance, is_spectral) *= tr;
                    masked(ray.o, active_medium)    = mi.p;
                    masked(ray.mint, active_medium) = 0.f;
                    masked(si.t, active_medium)     = si.t - mi.t;
                }

                masked(mi.t, si.t - mi.t <= 1e-5f) = math::Infinity<Float>; 

                needs_intersection &= !active_medium;
                escaped_medium = active_medium && !mi.is_valid();
                active_medium &= mi.is_valid();
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
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_volume_step_size, m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), Float(0.0f));
    };

    MTS_DECLARE_CLASS()

protected:
    float m_volume_step_size;
};

MTS_IMPLEMENT_CLASS_VARIANT(RaymarchingPathIntegrator, MonteCarloIntegrator);
MTS_EXPORT_PLUGIN(RaymarchingPathIntegrator, "Raymarching Path Tracer integrator");
NAMESPACE_END(mitsuba)