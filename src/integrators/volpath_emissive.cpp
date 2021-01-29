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
class EmissiveVolumetricPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {

public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MTS_IMPORT_TYPES(Scene, Sampler, Emitter, EmitterPtr, BSDF, BSDFPtr,
                     Medium, MediumPtr, PhaseFunctionContext)

    EmissiveVolumetricPathIntegrator(const Properties &props) : Base(props) {
        std::string sampling_type = props.string("probability_type", "analog");
        if (sampling_type == "analog") {
            m_sampling_type = 0;
        } else if (sampling_type == "max") {
            m_sampling_type = 1;
        } else if (sampling_type == "average") {
            m_sampling_type = 2;
        } else {
            Log(Warn, "Sampling Probability Type %s not recognised, defaulting to \"analog\" sampling", sampling_type);
            m_sampling_type = 0;
        }
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

    MTS_INLINE std::tuple<Float, Float, Float, Spectrum, Spectrum, Spectrum>
    medium_probabilities(const MediumInteraction3f &mi,
                         const Spectrum &throughput,
                         UInt32 &channel,
                         uint32_t probability_type = 0) const {
        Float prob_emission, prob_scatter, prob_null, c;
        Spectrum weight_emission(0.0f), weight_scatter(0.0f), weight_null(0.0f);
        if (probability_type == 0) {
            std::tie(prob_emission, prob_scatter, prob_null) = medium_probabilities_analog(mi.sigma_t, mi.sigma_s, mi.sigma_n, mi.combined_extinction, channel);
        } else {
            if (probability_type == 1) {
                std::tie(prob_emission, prob_scatter, prob_null) = medium_probabilities_max(mi.sigma_t, mi.sigma_s, mi.sigma_n, throughput);
            } else if (probability_type == 2) {
                std::tie(prob_emission, prob_scatter, prob_null) = medium_probabilities_average(mi.sigma_t, mi.sigma_s, mi.sigma_n, throughput);
            } else {
                Throw("Invalid probability type:", "%i", probability_type);
            }
        }
        Mask natural_medium  = mi.medium->is_natural();

        masked(prob_emission, mi.is_valid() && !natural_medium) = 1.f;
        
        c = prob_emission + prob_scatter + prob_null;
        masked(c, eq(c, 0.f)) = 1.0f;
        prob_emission /= c;
        prob_scatter  /= c;
        prob_null     /= c;

        masked(weight_emission, prob_emission > 0.f &&  natural_medium) = (mi.sigma_t - mi.sigma_s) / prob_emission;
        masked(weight_emission, prob_emission > 0.f && !natural_medium) = 1.f / prob_emission;
        masked(weight_scatter,  prob_scatter > 0.f)  =  mi.sigma_s / prob_scatter;
        masked(weight_null,     prob_null > 0.f)     =  mi.sigma_n / prob_null;
        
        masked(weight_emission, neq(weight_emission, weight_emission) || !(weight_emission < math::Infinity<Float>)) = 0.f;
        masked(weight_scatter, neq(weight_scatter, weight_scatter) || !(weight_scatter < math::Infinity<Float>)) = 0.f;
        masked(weight_null, neq(weight_null, weight_null) || !(weight_null < math::Infinity<Float>)) = 0.f;
        
        return { prob_emission, prob_scatter, prob_null, weight_emission, weight_scatter, weight_null };
    }

    MTS_INLINE std::tuple<Float, Float, Float>
    medium_probabilities_analog(const Spectrum &sigma_t, 
                                const Spectrum &sigma_s,
                                const Spectrum &sigma_n,
                                const Spectrum &combined_extinction,
                                UInt32 &channel) const {
        Float prob_e = 0.f, prob_s = 0.f, prob_n = 0.f;
        prob_e = (index_spectrum(sigma_t, channel) - index_spectrum(sigma_s, channel)) / index_spectrum(combined_extinction, channel);
		prob_s =  index_spectrum(sigma_s, channel) / index_spectrum(combined_extinction, channel);
		prob_n =  index_spectrum(sigma_n, channel) / index_spectrum(combined_extinction, channel);
        return { prob_e, prob_s, prob_n };
    }

    MTS_INLINE std::tuple<Float, Float, Float>
    medium_probabilities_max(const Spectrum &sigma_t, 
                             const Spectrum &sigma_s,
                             const Spectrum &sigma_n,
                             const Spectrum &throughput) const {
        Float prob_e = 0.f, prob_s = 0.f, prob_n = 0.f;
        prob_e = hmax(abs((sigma_t - sigma_s) * throughput));
        prob_s = hmax(abs( sigma_s * throughput));
        prob_n = hmax(abs( sigma_n * throughput));
        return { prob_e, prob_s, prob_n};
    }

    MTS_INLINE std::tuple<Float, Float, Float>
    medium_probabilities_average(const Spectrum &sigma_t,
                                 const Spectrum &sigma_s, 
                                 const Spectrum &sigma_n,
                                 const Spectrum &throughput) const {
        Float prob_e = 0.f, prob_s = 0.f, prob_n = 0.f;
        prob_e = hmean(abs((sigma_t - sigma_s) * throughput));
        prob_s = hmean(abs( sigma_s * throughput));
        prob_n = hmean(abs( sigma_n * throughput));
        return {prob_e, prob_s, prob_n} ;
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

        Spectrum throughput(1.f), result(0.f), prev_throughput(1.f);
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
            Mask act_emission = false, act_null_scatter = false, 
				 act_medium_scatter = false, escaped_medium = false;

            if (any_or<true>(active_medium)) {
                mi = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                masked(ray.maxt, active_medium && medium->is_homogeneous() && mi.is_valid()) = mi.t;
                Mask intersect = needs_intersection && active_medium;
                if (any_or<true>(intersect)) {
                    masked(si, intersect) = scene->ray_intersect(ray, intersect);
                }
                needs_intersection &= !active_medium;

                masked(mi.t, active_medium && (si.t < mi.t)) = math::Infinity<Float>;

                auto [tr, free_flight_pdf] = medium->eval_tr_and_pdf(mi, si, active_medium);
                Float tr_pdf = index_spectrum(free_flight_pdf, channel);
                //Float tr_pdf = hmean(free_flight_pdf);
                prev_throughput = throughput;
                masked(throughput, active_medium) *= select(tr_pdf > 0.f, tr / tr_pdf, 0.f);

                escaped_medium      = active_medium && !mi.is_valid();
                active_medium      &= mi.is_valid();
            }
			
            if (any_or<true>(active_medium)) {
                // Compute emmission, scatter and null event probabilities
                auto [prob_emission, prob_scatter, prob_null, weight_emission, weight_scatter, weight_null] = medium_probabilities(mi, prev_throughput, channel, m_sampling_type);

                // Handle absorption, null and real scatter events
                auto medium_sample_eta   = sampler->next_1d(active_medium);

                Mask emission_interaction =  medium_sample_eta <= prob_emission;
                Mask scatter_interaction  = (medium_sample_eta <= 1 - prob_null) && (medium_sample_eta > prob_emission);
                Mask null_interaction     =  medium_sample_eta > 1 - prob_null;

                act_emission       |= emission_interaction && active_medium;
                act_medium_scatter |= scatter_interaction && active_medium;
                act_null_scatter   |= null_interaction && active_medium;

                if (any_or<true>(act_emission)) {
                    masked(result, act_emission)     += weight_emission * throughput * mi.radiance;
                    masked(throughput, act_emission) *= prob_emission;
                    
                    // Move the ray along
					masked(ray.o, act_emission)    = mi.p;
					masked(ray.mint, act_emission) = 0.f;
					masked(si.t, act_emission)     = si.t - mi.t;
                }

                masked(depth, act_medium_scatter) += 1;

                // Dont estimate lighting if we exceeded number of bounces
                active &= depth < (uint32_t) m_max_depth;

                active &= !act_emission;
				
				act_medium_scatter &= active;
				act_null_scatter   &= active;

				if (any_or<true>(act_null_scatter)) {
                    masked(throughput, act_null_scatter) *= weight_null;

                    // Move the ray along
					masked(ray.o, act_null_scatter)    = mi.p;
					masked(ray.mint, act_null_scatter) = 0.f;
					masked(si.t, act_null_scatter)     = si.t - mi.t;
				}
				
				if (any_or<true>(act_medium_scatter)) {
                    masked(throughput, act_medium_scatter) *= weight_scatter;

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
        Spectrum transmittance(1.0f), prev_transmittance(1.0f);

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
                auto mi = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                masked(ray.maxt, active_medium && medium->is_homogeneous() && mi.is_valid()) = min(mi.t, remaining_dist);
                Mask intersect = needs_intersection && active_medium;
                if (any_or<true>(intersect))
                    masked(si, intersect) = scene->ray_intersect(ray, intersect);

                masked(mi.t, active_medium && (si.t < mi.t)) = math::Infinity<Float>;
                needs_intersection &= !active_medium;

                Mask is_spectral = medium->has_spectral_extinction() && active_medium;
                Mask not_spectral = !is_spectral && active_medium;
                if (any_or<true>(is_spectral)) {
                    Float t      = min(remaining_dist, min(mi.t, si.t)) - mi.mint;
                    UnpolarizedSpectrum tr  = exp(-t * mi.combined_extinction);
                    UnpolarizedSpectrum free_flight_pdf = select(si.t < mi.t || mi.t > remaining_dist, tr, tr * mi.combined_extinction);
                    Float tr_pdf = index_spectrum(free_flight_pdf, channel);
                    prev_transmittance = transmittance;
                    masked(transmittance, is_spectral) *= select(tr_pdf > 0.f, tr / tr_pdf, 0.f);
                }

                // Handle exceeding the maximum distance by medium sampling
                masked(total_dist, active_medium && (mi.t > remaining_dist) && mi.is_valid()) = ds.dist;
                masked(mi.t, active_medium && (mi.t > remaining_dist)) = math::Infinity<Float>;

                escaped_medium = active_medium && !mi.is_valid();
                active_medium &= mi.is_valid();
                is_spectral &= active_medium;
                not_spectral &= active_medium;

                masked(total_dist, active_medium) += mi.t;

                if (any_or<true>(active_medium)) {
                    masked(ray.o, active_medium)    = mi.p;
                    masked(ray.mint, active_medium) = 0.f;
                    masked(si.t, active_medium) = si.t - mi.t;
                    // Compute emmission, scatter and null event probabilities
                    auto [prob_emission, prob_scatter, prob_null, weight_emission, weight_scatter, weight_null] = medium_probabilities(mi, prev_transmittance, channel, m_sampling_type);

                    masked(transmittance, active_medium) *= mi.sigma_n;
                }
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

    /// Samples a volume emitter in the scene and evaluates it's attenuated contribution
    std::tuple<Spectrum, DirectionSample3f>
    sample_volume_emitter(const Interaction3f &ref_interaction, Mask is_medium_interaction, const Scene *scene,
                          Sampler *sampler, MediumPtr medium, UInt32 channel, Mask active) const {
        using EmitterPtr = replace_scalar_t<Float, const Emitter *>;
        Spectrum transmittance(1.0f);

        auto [ds, emitter_val, sampled_medium] = scene->sample_volume_emitter_direction(ref_interaction, sampler->next_2d(active), false, active);
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
            Mask is_sampled_emitter = active && eq(medium, sampled_medium);

            if (any_or<true>(active_medium)) {
                auto mi = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                masked(ray.maxt, active_medium && medium->is_homogeneous() && mi.is_valid()) = min(mi.t, remaining_dist);
                Mask intersect = needs_intersection && active_medium;
                if (any_or<true>(intersect))
                    masked(si, intersect) = scene->ray_intersect(ray, intersect);

                masked(mi.t, active_medium && (si.t < mi.t)) = math::Infinity<Float>;
                needs_intersection &= !active_medium;

                Mask is_spectral = medium->has_spectral_extinction() && active_medium;
                Mask not_spectral = !is_spectral && active_medium;

                auto [prob_emission, prob_scatter, prob_null, weight_emission, weight_scatter, weight_null] = medium_probabilities(mi, transmittance, channel, m_sampling_type);

                if (any_or<true>(is_spectral)) {
                    Float t      = min(remaining_dist, min(mi.t, si.t)) - mi.mint;
                    UnpolarizedSpectrum tr  = exp(-t * mi.combined_extinction);
                    UnpolarizedSpectrum free_flight_pdf = select(si.t < mi.t || mi.t > remaining_dist, tr, tr * mi.combined_extinction);
                    Float tr_pdf = index_spectrum(free_flight_pdf, channel);
                    masked(transmittance, is_spectral) *= select(tr_pdf > 0.f, tr / tr_pdf, 0.f);
                    if (any_or<true>(is_sampled_emitter)) {
                        masked(emitter_val, is_sampled_emitter) += select(tr_pdf > 0.f, weight_emission * transmittance * mi.radiance, 0.f);
                    }
                }

                // Handle exceeding the maximum distance by medium sampling
                masked(total_dist, active_medium && (mi.t > remaining_dist) && mi.is_valid()) = ds.dist;
                masked(mi.t, active_medium && (mi.t > remaining_dist)) = math::Infinity<Float>;

                escaped_medium = active_medium && !mi.is_valid();
                active_medium &= mi.is_valid();
                is_spectral &= active_medium;
                not_spectral &= active_medium;

                masked(total_dist, active_medium) += mi.t;

                if (any_or<true>(active_medium)) {
                    masked(ray.o, active_medium)    = mi.p;
                    masked(ray.mint, active_medium) = 0.f;
                    masked(si.t, active_medium) = si.t - mi.t;
                    // Compute emmission, scatter and null event probabilities

                    masked(transmittance, active_medium) *= mi.sigma_n;
                }
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
        return { emitter_val, ds };
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

        Spectrum transmittance(1.0f), prev_transmittance(1.0f);
        Float emitter_pdf(0.0f);
        SurfaceInteraction3f si = si_ray;
        while (any(active)) {
            Mask escaped_medium = false;
            Mask active_medium  = active && neq(medium, nullptr);
            Mask active_surface = active && !active_medium;
            SurfaceInteraction3f si_medium;
            if (any_or<true>(active_medium)) {
                auto mi = medium->sample_interaction(ray, sampler->next_1d(active_medium), channel, active_medium);
                masked(ray.maxt, active_medium && medium->is_homogeneous() && mi.is_valid()) = mi.t;
                Mask intersect = needs_intersection && active_medium;
                if (any_or<true>(intersect))
                    masked(si, intersect) = scene->ray_intersect(ray, intersect);

                masked(mi.t, active_medium && (si.t < mi.t)) = math::Infinity<Float>;

                Mask is_spectral = medium->has_spectral_extinction() && active_medium;
                Mask not_spectral = !is_spectral && active_medium;
                if (any_or<true>(is_spectral)) {
                    auto [tr, free_flight_pdf] = medium->eval_tr_and_pdf(mi, si, is_spectral);
                    Float tr_pdf = index_spectrum(free_flight_pdf, channel);
                    //Float tr_pdf       = hmin(free_flight_pdf);
                    prev_transmittance = transmittance;
                    masked(transmittance, is_spectral) *= select(tr_pdf > 0.f, tr / tr_pdf, 0.f);
                }

                needs_intersection &= !active_medium;
                escaped_medium = active_medium && !mi.is_valid();
                active_medium &= mi.is_valid();

                if (any_or<true>(active_medium)) {
                    masked(ray.o, active_medium)    = mi.p;
                    masked(ray.mint, active_medium) = 0.f;
                    masked(si.t, active_medium) = si.t - mi.t;
                    // Compute emmission, scatter and null event probabilities
                    auto [prob_emission, prob_scatter, prob_null, weight_emission, weight_scatter, weight_null] = medium_probabilities(mi, prev_transmittance, channel, m_sampling_type);

                    masked(transmittance, active_medium) *= mi.sigma_n;
                }
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
        return tfm::format("EmissiveVolumetricSimplePathIntegrator[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "  sampling_type = %i\n"
                           "]",
                           m_max_depth, m_rr_depth, m_sampling_type);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.0f, pdf_a / (pdf_a + pdf_b), Float(0.0f));
    };

    MTS_DECLARE_CLASS()
protected:
    uint32_t m_sampling_type;
};

MTS_IMPLEMENT_CLASS_VARIANT(EmissiveVolumetricPathIntegrator, MonteCarloIntegrator);
MTS_EXPORT_PLUGIN(EmissiveVolumetricPathIntegrator, "Emissive Volumetric Path Tracer integrator");
NAMESPACE_END(mitsuba)