#include <torch/extension.h>
#include <vector>

// the functions are defined in the respective files
// this file only contains the library implementation

#include "gaugemul.hpp"
#include "dirac.hpp"
#include "plaq.hpp"
#include "muladd.hpp"

namespace qcd_ml_accel_dirac{

// Registers _C as a Python extension module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(qcd_ml_accel_dirac, m) {
    m.def("shift_gaugemul(Tensor U2, Tensor Uv, int[] u2shifts, int[] uvshifts) -> Tensor");
    m.def("dirac_wilson_call(Tensor U, Tensor v, float mass) -> Tensor");
    m.def("dirac_wilson_clover_call(Tensor U, Tensor v, Tensor[] F, float mass, float csw) -> Tensor");
    m.def("plaquette_action(Tensor U, float g) -> float");

    m.def("muladd_bench_nopar(Tensor a, Tensor b, Tensor c) -> Tensor");
    m.def("muladd_bench_par(Tensor a, Tensor b, Tensor c) -> Tensor");
    m.def("muladd_bench_nopar_time(Tensor a, Tensor b, Tensor c) -> Tensor");
    m.def("muladd_bench_par_time(Tensor a, Tensor b, Tensor c) -> Tensor");
    m.def("mul_real_bench_nopar(Tensor a, Tensor b) -> Tensor");
    m.def("mul_compl_bench_nopar(Tensor a, Tensor b) -> Tensor");

    m.def("gauge_transform(Tensor U, Tensor v) -> Tensor");
    m.def("gauge_transform_par(Tensor U, Tensor v) -> Tensor");
    m.def("gauge_transform_gamma(Tensor U, Tensor v) -> Tensor");
    m.def("gauge_transform_gamma_shift(Tensor U, Tensor v) -> Tensor");
    m.def("gauge_transform_gamma_2shift(Tensor U, Tensor v) -> Tensor");
    m.def("gauge_transform_gamma_2shift_split(Tensor U, Tensor v) -> Tensor");
    m.def("gauge_transform_gamma_2shift_ysw(Tensor U, Tensor v) -> Tensor");
    m.def("gauge_transform_gamma_2tshift(Tensor U, Tensor v) -> Tensor");
    m.def("gauge_transform_gamma_2ytshift(Tensor U, Tensor v) -> Tensor");
    m.def("gauge_transform_simple_ytshift(Tensor U, Tensor v) -> Tensor");

    m.def("dirac_wilson_call_nohop(Tensor U, Tensor v, float mass) -> Tensor");
    m.def("dirac_wilson_call_nohop_usw(Tensor U, Tensor v, float mass) -> Tensor");
}

// Registers backend implementations
TORCH_LIBRARY_IMPL(qcd_ml_accel_dirac, CPU, m) {
    m.impl("shift_gaugemul", &shift_gaugemul_cpu);
    m.impl("dirac_wilson_call", &dw_call_cpu);
    m.impl("dirac_wilson_clover_call", &dwc_call_cpu);
    m.impl("plaquette_action", &plaq_action_cpu);

    m.impl("muladd_bench_nopar", &muladd_bench_nopar);
    m.impl("muladd_bench_par", &muladd_bench_par);
    m.impl("muladd_bench_nopar_time", &muladd_bench_nopar_time);
    m.impl("muladd_bench_par_time", &muladd_bench_par_time);
    m.impl("mul_real_bench_nopar", &mul_real_bench_nopar);
    m.impl("mul_compl_bench_nopar", &mul_compl_bench_nopar);

    m.impl("gauge_transform", &gauge_transform);
    m.impl("gauge_transform_par", &gauge_transform_par);
    m.impl("gauge_transform_gamma", &gauge_transform_gamma);
    m.impl("gauge_transform_gamma_shift", &gauge_transform_gamma_shift);
    m.impl("gauge_transform_gamma_2shift", &gauge_transform_gamma_2shift);
    m.impl("gauge_transform_gamma_2shift_split", &gauge_transform_gamma_2shift_split);
    m.impl("gauge_transform_gamma_2shift_ysw", &gauge_transform_gamma_2shift_ysw);
    m.impl("gauge_transform_gamma_2tshift", &gauge_transform_gamma_2tshift);
    m.impl("gauge_transform_gamma_2ytshift", &gauge_transform_gamma_2ytshift);
    m.impl("gauge_transform_simple_ytshift", &gauge_transform_simple_ytshift);

    m.impl("dirac_wilson_call_nohop", &dw_call_nohop);
    m.impl("dirac_wilson_call_nohop_usw", &dw_call_nohop_Usw);
}

}
