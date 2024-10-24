#include <torch/extension.h>
#include <vector>

// the functions are defined in the respective files
// this file only contains the library implementation

#include "gaugemul.hpp"
#include "dirac.hpp"
#include "plaq.hpp"

namespace qcd_ml_accel_dirac{

// Registers _C as a Python extension module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(qcd_ml_accel_dirac, m) {
    m.def("shift_gaugemul(Tensor U2, Tensor Uv, int[] u2shifts, int[] uvshifts) -> Tensor");
    m.def("dirac_wilson_call(Tensor U, Tensor v, float mass) -> Tensor");
    m.def("dirac_wilson_clover_call(Tensor U, Tensor v, Tensor[] F, float mass, float csw) -> Tensor");
    m.def("plaquette_action(Tensor U, float g) -> float");
}

// Registers backend implementations
TORCH_LIBRARY_IMPL(qcd_ml_accel_dirac, CPU, m) {
    m.impl("shift_gaugemul", &shift_gaugemul_cpu);
    m.impl("dirac_wilson_call", &dw_call_cpu);
    m.impl("dirac_wilson_clover_call", &dwc_call_cpu);
    m.impl("plaquette_action", &plaq_action_cpu);
}

}
