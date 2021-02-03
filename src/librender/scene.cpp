#include <mitsuba/core/properties.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/kdtree.h>
#include <mitsuba/render/integrator.h>
#include <enoki/stl.h>

#if defined(MTS_ENABLE_EMBREE)
#  include "scene_embree.inl"
#else
#  include "scene_native.inl"
#endif

#if defined(MTS_ENABLE_OPTIX)
#  include "scene_optix.inl"
#endif

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT Scene<Float, Spectrum>::Scene(const Properties &props) {
    for (auto &kv : props.objects()) {
        m_children.push_back(kv.second.get());

        Shape *shape           = dynamic_cast<Shape *>(kv.second.get());
        Emitter *emitter       = dynamic_cast<Emitter *>(kv.second.get());
        Sensor *sensor         = dynamic_cast<Sensor *>(kv.second.get());
        Integrator *integrator = dynamic_cast<Integrator *>(kv.second.get());

        if (shape) {
            if (shape->is_emitter())
                m_emitters.push_back(shape->emitter());
            if (shape->is_sensor())
                m_sensors.push_back(shape->sensor());
            if (shape->is_shapegroup()) {
                m_shapegroups.push_back((ShapeGroup*)shape);
            } else {
                m_bbox.expand(shape->bbox());
                if (shape->interior_medium()) {
                    if (shape->interior_medium()->has_emission()) {
                        m_emissive_mediums.push_back(shape);
                    }
                }
                m_shapes.push_back(shape);
            }
        } else if (emitter) {
            // Surface emitters will be added to the list when attached to a shape
            if (!has_flag(emitter->flags(), EmitterFlags::Surface))
                m_emitters.push_back(emitter);

            if (emitter->is_environment()) {
                if (m_environment)
                    Throw("Only one environment emitter can be specified per scene.");
                m_environment = emitter;
            }
        } else if (sensor) {
            m_sensors.push_back(sensor);
        } else if (integrator) {
            if (m_integrator)
                Throw("Only one integrator can be specified per scene.");
            m_integrator = integrator;
        }
    }

    if (m_sensors.empty()) {
        Log(Warn, "No sensors found! Instantiating a perspective camera..");
        Properties sensor_props("perspective");
        sensor_props.set_float("fov", 45.0f);

        /* Create a perspective camera with a 45 deg. field of view
           and positioned so that it can see the entire scene */
        if (m_bbox.valid()) {
            ScalarPoint3f center = m_bbox.center();
            ScalarVector3f extents = m_bbox.extents();

            ScalarFloat distance =
                hmax(extents) / (2.f * std::tan(45.f * .5f * math::Pi<ScalarFloat> / 180.f));

            sensor_props.set_float("far_clip", hmax(extents) * 5 + distance);
            sensor_props.set_float("near_clip", distance / 100);

            sensor_props.set_float("focus_distance", distance + extents.z() / 2);
            sensor_props.set_transform(
                "to_world", ScalarTransform4f::translate(ScalarVector3f(
                                center.x(), center.y(), m_bbox.min.z() - distance)));
        }

        m_sensors.push_back(
            PluginManager::instance()->create_object<Sensor>(sensor_props));
    }

    if (!m_integrator) {
        Log(Warn, "No integrator found! Instantiating a path tracer..");
        m_integrator = PluginManager::instance()->
            create_object<Integrator>(Properties("path"));
    }

    if constexpr (is_cuda_array_v<Float>)
        accel_init_gpu(props);
    else
        accel_init_cpu(props);

    // Create emitters' shapes (environment luminaires)
    for (Emitter *emitter: m_emitters)
        emitter->set_scene(this);

    m_shapes_grad_enabled = false;
}

MTS_VARIANT Scene<Float, Spectrum>::~Scene() {
    if constexpr (is_cuda_array_v<Float>)
        accel_release_gpu();
    else
        accel_release_cpu();
}

MTS_VARIANT typename Scene<Float, Spectrum>::SurfaceInteraction3f
Scene<Float, Spectrum>::ray_intersect(const Ray3f &ray, Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::RayIntersect, active);

    if constexpr (is_cuda_array_v<Float>)
        return ray_intersect_gpu(ray, HitComputeFlags::All, active);
    else
        return ray_intersect_cpu(ray, HitComputeFlags::All, active);
}

MTS_VARIANT typename Scene<Float, Spectrum>::SurfaceInteraction3f
Scene<Float, Spectrum>::ray_intersect(const Ray3f &ray, HitComputeFlags flags, Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::RayIntersect, active);

    if constexpr (is_cuda_array_v<Float>)
        return ray_intersect_gpu(ray, flags, active);
    else
        return ray_intersect_cpu(ray, flags, active);
}

MTS_VARIANT typename Scene<Float, Spectrum>::PreliminaryIntersection3f
Scene<Float, Spectrum>::ray_intersect_preliminary(const Ray3f &ray, Mask active) const {
    if constexpr (is_cuda_array_v<Float>)
        return ray_intersect_preliminary_gpu(ray, active);
    else
        return ray_intersect_preliminary_cpu(ray, active);
}

MTS_VARIANT typename Scene<Float, Spectrum>::SurfaceInteraction3f
Scene<Float, Spectrum>::ray_intersect_naive(const Ray3f &ray, Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::RayIntersect, active);

#if !defined(MTS_ENABLE_EMBREE)
    if constexpr (!is_cuda_array_v<Float>)
        return ray_intersect_naive_cpu(ray, active);
#endif
    ENOKI_MARK_USED(ray);
    ENOKI_MARK_USED(active);
    NotImplementedError("ray_intersect_naive");
}

MTS_VARIANT typename Scene<Float, Spectrum>::Mask
Scene<Float, Spectrum>::ray_test(const Ray3f &ray, Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::RayTest, active);

    if constexpr (is_cuda_array_v<Float>)
        return ray_test_gpu(ray, active);
    else
        return ray_test_cpu(ray, active);
}

MTS_VARIANT std::pair<typename Scene<Float, Spectrum>::DirectionSample3f, Spectrum>
Scene<Float, Spectrum>::sample_emitter_direction(const Interaction3f &ref, const Point2f &sample_,
                                                 bool test_visibility, Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::SampleEmitterDirection, active);

    using EmitterPtr = replace_scalar_t<Float, Emitter*>;

    Point2f sample(sample_);
    DirectionSample3f ds;
    Spectrum spec;

    if (likely(!m_emitters.empty())) {
        if (m_emitters.size() == 1) {
            // Fast path if there is only one emitter
            std::tie(ds, spec) = m_emitters[0]->sample_direction(ref, sample, active);
        } else {
            ScalarFloat emitter_pdf = 1.f / m_emitters.size();

            // Randomly pick an emitter
            UInt32 index =
                min(UInt32(sample.x() * (ScalarFloat) m_emitters.size()),
                    (uint32_t) m_emitters.size() - 1);

            // Rescale sample.x() to lie in [0,1) again
            sample.x() = (sample.x() - index*emitter_pdf) * m_emitters.size();

            EmitterPtr emitter = gather<EmitterPtr>(m_emitters.data(), index, active);

            // Sample a direction towards the emitter
            std::tie(ds, spec) = emitter->sample_direction(ref, sample, active);

            // Account for the discrete probability of sampling this emitter
            ds.pdf *= emitter_pdf;
            spec *= rcp(emitter_pdf);
        }

        active &= neq(ds.pdf, 0.f);

        // Perform a visibility test if requested
        if (test_visibility && any_or<true>(active)) {
            Ray3f ray(ref.p, ds.d, math::RayEpsilon<Float> * (1.f + hmax(abs(ref.p))), ds.dist * (1.f - math::ShadowEpsilon<Float>), ref.time, ref.wavelengths);
            spec[ray_test(ray, active)] = 0.f;
        }
    } else {
        ds = zero<DirectionSample3f>();
        spec = 0.f;
    }

    // ds.pdf *= (1.f - get_volume_emitter_probability());
    // spec   *= (1.f - get_volume_emitter_probability());

    return { ds, spec };
}

MTS_VARIANT Float
Scene<Float, Spectrum>::pdf_emitter_direction(const Interaction3f &ref,
                                              const DirectionSample3f &ds,
                                              Mask active) const {
    MTS_MASK_ARGUMENT(active);
    using EmitterPtr = replace_scalar_t<Float, const Emitter *>;


    if (m_emitters.size() == 1) {
        // Fast path if there is only one emitter
        return m_emitters[0]->pdf_direction(ref, ds, active) * (1.f - get_volume_emitter_probability());
    } else {
        return reinterpret_array<EmitterPtr>(ds.object)->pdf_direction(ref, ds, active) *
            (1.f / m_emitters.size()) * (1.f - get_volume_emitter_probability());
    }
}

MTS_VARIANT std::pair<typename Scene<Float, Spectrum>::DirectionSample3f, Spectrum>
Scene<Float, Spectrum>::sample_volume_emitter_direction(const Interaction3f &ref,
                                                        const Point2f &sample_,
                                                        bool test_visibility,
                                                        Mask active,
                                                        UInt32 channel) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::SampleEmitterDirection, active);

    using ShapePtr = replace_scalar_t<Float, const Shape*>;
    using MediumPtr = replace_scalar_t<Float, const Medium *>;

    Point2f sample(sample_);
    DirectionSample3f ds;
    Spectrum spec(0.f);
    MediumPtr sampled_medium = nullptr;

    if (likely(!m_emissive_mediums.empty())) {
        if (m_emissive_mediums.size() == 1) {
            // Fast path if there is only one emitter
            ds    = m_emissive_mediums[0]->sample_direction(ref, sample, active);
            sampled_medium = m_emissive_mediums[0]->interior_medium();
        } else {
            ScalarFloat emitter_pdf = 1.f / m_emissive_mediums.size();

            // Randomly pick an emitter
            UInt32 index = min(UInt32(sample.x() * (ScalarFloat) m_emissive_mediums.size()), (uint32_t) m_emissive_mediums.size() - 1);

            // Rescale sample.x() to lie in [0,1) again
            sample.x() = (sample.x() - index * emitter_pdf) * m_emissive_mediums.size();

            ShapePtr emitter = gather<ShapePtr>(m_emissive_mediums.data(), index, active);

            // Sample a direction towards the emitter
            ds = emitter->sample_direction(ref, sample, active);

            // Account for the discrete probability of sampling this emitter
            ds.pdf        *= emitter_pdf * emitter->pdf_direction(ref, ds, active);
            sampled_medium = emitter->interior_medium();
        }

        active &= neq(ds.pdf, 0.f);

        Ray3f ray_vol(ref.p, ds.d,
                  math::RayEpsilon<Float> * (1.f + hmax(abs(ref.p))),
                  ds.dist * (1.f - math::ShadowEpsilon<Float>), ref.time,
                  ref.wavelengths);

        SurfaceInteraction3f si = ray_intersect(ray_vol, active);

        MediumInteraction3f mi = sampled_medium->sample_interaction(ray_vol, si.t, sample.y(), channel, active);
        Float D       = 1.f / (min(si.t, mi.maxt) - mi.mint);
        Float sampled_t = mi.mint + (mi.sample * D);
        Mask valid_mi = active && (sampled_t <= min(si.t, mi.maxt));
        mi.t = select(valid_mi, sampled_t, math::Infinity<Float>);
        mi.p = ray_vol(mi.t);

        UnpolarizedSpectrum tr = exp(-(mi.t - mi.mint) * mi.combined_extinction);

        ds.pdf *= select(si.t < mi.t, 0.f, D);
        spec = mi.radiance * tr * m_emissive_mediums.size();

        // Perform a visibility test if requested
        if (test_visibility && any_or<true>(active)) {
            Ray3f ray(ref.p, ds.d,
                      math::RayEpsilon<Float> * (1.f + hmax(abs(ref.p))),
                      ds.dist * (1.f - math::ShadowEpsilon<Float>), ref.time,
                      ref.wavelengths);
            spec[ray_test(ray, active)] = 0.f;
        }
    } else {
        ds   = zero<DirectionSample3f>();
        spec = 0.f;
        sampled_medium = nullptr;
    }

    ds.pdf *= get_volume_emitter_probability();
    spec   *= rcp(get_volume_emitter_probability());

    return { ds, spec };
}

MTS_VARIANT Float Scene<Float, Spectrum>::pdf_volume_emitter_direction(const Interaction3f &ref, 
                                                                       const DirectionSample3f &ds, 
                                                                       Mask active,
                                                                       UInt32 channel) const {
    MTS_MASK_ARGUMENT(active);
    using ShapePtr = replace_scalar_t<Float, Shape*>;

    Float base_pdf;

    if (m_emissive_mediums.size() == 1) {
        // Fast path if there is only one emitter
        base_pdf = m_emissive_mediums[0]->pdf_direction(ref, ds, active);
    } else {
        base_pdf = reinterpret_array<ShapePtr>(ds.object)->pdf_direction(ref, ds, active) * (1.f / m_emissive_mediums.size());
    }

    MediumPtr sampled_medium = reinterpret_array<ShapePtr>(ds.object)->interior_medium();

    Ray3f ray(ref.p, ds.d,
                math::RayEpsilon<Float> * (1.f + hmax(abs(ref.p))),
                ds.dist * (1.f - math::ShadowEpsilon<Float>), ref.time,
                ref.wavelengths);

    SurfaceInteraction3f si = ray_intersect(ray, active);

    MediumInteraction3f mi = sampled_medium->sample_interaction(ray, si.t, 0.f, channel, active);
    Float D       = 1.f / (min(si.t, mi.maxt) - mi.mint);
    Float sampled_t = mi.mint + (mi.sample * D);
    return base_pdf * select(si.t < sampled_t, 0.f, D) * get_volume_emitter_probability();
}

MTS_VARIANT void Scene<Float, Spectrum>::traverse(TraversalCallback *callback) {
    for (auto& child : m_children) {
        std::string id = child->id();
        if (id.empty() || string::starts_with(id, "_unnamed_"))
            id = child->class_()->name();
        callback->put_object(id, child.get());
    }
}

MTS_VARIANT void Scene<Float, Spectrum>::parameters_changed(const std::vector<std::string> &keys) {
    if (m_environment)
        m_environment->set_scene(this); // TODO use parameters_changed({"scene"})

    bool update_accel = false;
    for (auto &s : m_shapes) {
        if (string::contains(keys, s->id()) || string::contains(keys, s->class_()->name())) {
            update_accel = true;
            break;
        }
    }

    if (update_accel) {
        if constexpr (is_cuda_array_v<Float>)
            accel_parameters_changed_gpu();
        else {
            // TODO update Embree BVH or Mitsuba kdtree if necessary
        }
    }

    // Checks whether any of the shape's parameters require gradient
    m_shapes_grad_enabled = false;
    if constexpr (is_diff_array_v<Float>) {
        for (auto& s : m_shapes) {
            m_shapes_grad_enabled |= s->parameters_grad_enabled();
            if (m_shapes_grad_enabled) break;
        }
    }
}

MTS_VARIANT std::string Scene<Float, Spectrum>::to_string() const {
    std::ostringstream oss;
    oss << "Scene[" << std::endl
        << "  children = [" << std::endl;
    for (size_t i = 0; i < m_children.size(); ++i) {
        oss << "    " << string::indent(m_children[i], 4);
        if (i + 1 < m_children.size())
            oss << ",";
        oss <<  std::endl;
    }
    oss << "  ]"<< std::endl
        << "]";
    return oss.str();
}

void librender_nop() { }

MTS_IMPLEMENT_CLASS_VARIANT(Scene, Object, "scene")
MTS_INSTANTIATE_CLASS(Scene)
NAMESPACE_END(mitsuba)
