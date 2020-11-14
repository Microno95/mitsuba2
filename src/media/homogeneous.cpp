#include <enoki/stl.h>

#include <mitsuba/core/frame.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class HomogeneousMedium final : public Medium<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Medium, m_is_homogeneous, m_has_spectral_extinction, m_has_emission)
    MTS_IMPORT_TYPES(Scene, Sampler, Texture, Volume)

    HomogeneousMedium(const Properties &props) : Base(props) {
        m_is_homogeneous = true;
        m_albedo         = props.volume<Volume>("albedo", 0.75f);
        m_sigmat         = props.volume<Volume>("sigma_t", 1.f);
        m_emissivity     = props.volume<Volume>("emissivity", 0.0f);

        m_scale = props.float_("scale", 1.0f);
        m_emission_scale = props.float_("emission_scale", 1.0f);
        m_has_spectral_extinction = props.bool_("has_spectral_extinction", true);
        m_has_emission = props.bool_("has_emission", true);
    }

    MTS_INLINE auto eval_sigmat(const MediumInteraction3f &mi) const {
        return m_sigmat->eval(mi) * m_scale;
    }

    MTS_INLINE auto eval_emissivity(const MediumInteraction3f &mi) const {
        return m_emissivity->eval(mi) * m_emission_scale;
    }

    UnpolarizedSpectrum get_combined_extinction(const MediumInteraction3f &mi,
                                                Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);
        return eval_sigmat(mi);
    }

    std::tuple<UnpolarizedSpectrum, UnpolarizedSpectrum, UnpolarizedSpectrum>
    get_scattering_coefficients(const MediumInteraction3f &mi,
                                Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);
        auto sigmat                = eval_sigmat(mi);
        auto sigmas                = sigmat * m_albedo->eval(mi, active);
        UnpolarizedSpectrum sigman = 0.f;
        return { sigmas, sigman, sigmat };
    }

    UnpolarizedSpectrum get_emission_coefficient(const MediumInteraction3f &mi,
                                                 Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);
        return eval_emissivity(mi);
    }

    std::tuple<Mask, Float, Float>
    intersect_aabb(const Ray3f & /* ray */) const override {
        return { true, 0.f, math::Infinity<Float> };
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("scale", m_scale);
        callback->put_parameter("emission_scale", m_scale);
        callback->put_object("albedo", m_albedo.get());
        callback->put_object("sigma_t", m_sigmat.get());
        callback->put_object("emissivity", m_emissivity.get());
        Base::traverse(callback);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "HomogeneousMedium[" << std::endl
            << "  albedo         = " << string::indent(m_albedo) << std::endl
            << "  sigma_t        = " << string::indent(m_sigmat) << std::endl
            << "  emissivity     = " << string::indent(m_emissivity) << std::endl
            << "  scale          = " << string::indent(m_scale) << std::endl
            << "  emission_scale = " << string::indent(m_emission_scale) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Volume> m_sigmat, m_albedo, m_emissivity;
    ScalarFloat m_scale, m_emission_scale;
};

MTS_IMPLEMENT_CLASS_VARIANT(HomogeneousMedium, Medium)
MTS_EXPORT_PLUGIN(HomogeneousMedium, "Homogeneous Medium")
NAMESPACE_END(mitsuba)
