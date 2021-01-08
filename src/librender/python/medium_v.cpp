#include <mitsuba/core/properties.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/python/python.h>

/// Trampoline for derived types implemented in Python
MTS_VARIANT class PyMedium : public Medium<Float, Spectrum> {
public:
    MTS_IMPORT_TYPES(Medium, Sampler, Scene)

    PyMedium(const Properties &props) : Medium(props) {}

    std::tuple<Mask, Float, Float>
    intersect_aabb(const Ray3f &ray) const override {
        using Return = std::tuple<Mask, Float, Float>;
        PYBIND11_OVERLOAD_PURE(Return, Medium, intersect_aabb, ray);
    }

    UnpolarizedSpectrum
    get_combined_extinction(const MediumInteraction3f &mi,
                            Mask active = true) const override {
        PYBIND11_OVERLOAD_PURE(UnpolarizedSpectrum, Medium,
                               get_combined_extinction, mi, active);
    }

    UnpolarizedSpectrum get_radiance(const MediumInteraction3f &mi,
                                     Mask active = true) const override {
        PYBIND11_OVERLOAD_PURE(UnpolarizedSpectrum, Medium, get_radiance, mi,
                               active);
    }

    std::tuple<UnpolarizedSpectrum, UnpolarizedSpectrum, UnpolarizedSpectrum>
    get_scattering_coefficients(const MediumInteraction3f &mi,
                                Mask active = true) const override {
        using Return = std::tuple<UnpolarizedSpectrum, UnpolarizedSpectrum,
                                  UnpolarizedSpectrum>;
        PYBIND11_OVERLOAD_PURE(Return, Medium, get_scattering_coefficients, mi,
                               active);
    }

    std::string to_string() const override {
        PYBIND11_OVERLOAD_PURE(std::string, Medium, to_string, );
    }
};

MTS_PY_EXPORT(Medium) {
    MTS_PY_IMPORT_TYPES(Medium, MediumPtr, Scene, Sampler, Ray3f)
    using PyMedium = PyMedium<Float, Spectrum>;

    auto medium =
        py::class_<Medium, PyMedium, Object, ref<Medium>>(m, "Medium",
                                                          D(Medium))
            .def(py::init<const Properties &>())
            .def("intersect_aabb", vectorize(&Medium::intersect_aabb), "ray"_a)
            .def("get_combined_extinction",
                 vectorize(&Medium::get_combined_extinction), "mi"_a,
                 "active"_a = true)
            .def("get_scattering_coefficients",
                 vectorize(&Medium::get_scattering_coefficients), "mi"_a,
                 "active"_a = true)
            .def("get_radiance", vectorize(&Medium::get_radiance), "mi"_a,
                 "active"_a = true)
            .def("sample_interaction", vectorize(&Medium::sample_interaction),
                 "ray"_a, "sample"_a, "channel"_a, "active"_a = true)
            .def("eval_tr_eps_and_pdf", vectorize(&Medium::eval_tr_eps_and_pdf),
                 "mi"_a, "si"_a, "active"_a = true)
            .def_method(Medium, phase_function)
            .def_method(Medium, use_emitter_sampling)
            // .def_method(Medium, is_homogeneous)
            // .def_method(Medium, has_spectral_extinction)
            .def_method(Medium, id)
            .def("__repr__", &Medium::to_string);

    if constexpr (is_cuda_array_v<Float>) {
        pybind11_type_alias<UInt64, MediumPtr>();
    }

    if constexpr (is_array_v<Float>) {
        medium.def_static("intersect_aabb_vec",
                          vectorize([](const MediumPtr &ptr, Ray3f ray) {
                              return ptr->intersect_aabb(ray);
                          }),
                          "ptr"_a, "ray"_a);
        medium.def_static(
            "get_combined_extinction_vec",
            vectorize([](const MediumPtr &ptr, const MediumInteraction3f &mi,
                         Mask active) {
                return ptr->get_combined_extinction(mi, active);
            }),
            "ptr"_a, "mi"_a, "active"_a = true);
        medium.def_static(
            "get_scattering_coefficients_vec",
            vectorize([](const MediumPtr &ptr, const MediumInteraction3f &mi,
                         Mask active) {
                return ptr->get_scattering_coefficients(mi, active);
            }),
            "ptr"_a, "mi"_a, "active"_a = true);
        medium.def_static(
            "get_radiance_vec",
            vectorize(
                [](const MediumPtr &ptr, const MediumInteraction3f &mi,
                   Mask active) { return ptr->get_radiance(mi, active); }),
            "ptr"_a, "mi"_a, "active"_a = true);
        medium.def_static(
            "sample_interaction_vec",
            vectorize([](const MediumPtr &ptr, Ray3f ray, Float sample,
                         UInt32 channel, Mask active) {
                return ptr->sample_interaction(ray, sample, channel, active);
            }),
            "ptr"_a, "ray"_a, "sample"_a, "channel"_a, "active"_a = true);
        medium.def_static(
            "eval_tr_eps_and_pdf_vec",
            vectorize([](const MediumPtr &ptr, const MediumInteraction3f &mi,
                         const SurfaceInteraction3f &si, Mask active) {
                return ptr->eval_tr_eps_and_pdf(mi, si, active);
            }),
            "ptr"_a, "mi"_a, "si"_a, "active"_a = true);
    }

    MTS_PY_REGISTER_OBJECT("register_medium", Medium)
}
