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

    m.def("dirac_wilson_call_b(Tensor U, Tensor v, float mass, int bls) -> Tensor");
    m.def("dirac_wilson_call_b2(Tensor U, Tensor v, float mass, int bls) -> Tensor");
    m.def("dirac_wilson_clover_call_nopre(Tensor U, Tensor v, float mass, float csw) -> Tensor");
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

    m.impl("dirac_wilson_call_b", &dw_call_block_cpu);
    m.impl("dirac_wilson_call_b2", &dw_call_block2_cpu);
    m.impl("dirac_wilson_clover_call_nopre", &dwc_call_nopre_cpu);
}

}
