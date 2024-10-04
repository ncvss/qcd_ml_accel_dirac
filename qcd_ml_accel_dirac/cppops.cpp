#include <torch/extension.h>

#include <vector>

#include "indexfunc.hpp"
#include "gaugemul.hpp"
#include "dirac.hpp"
#include "plaq.hpp"

namespace qcd_ml_accel_dirac{

// the functions are defined in the respective header files
// indexfunc.hpp only contains helper functions that calculate pointer indices from coordinates
// the shifted gauge field multiplication function is in gaugemul.hpp
// the dirac operator call functions are in dirac.hpp


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
    m.impl("shift_gaugemul", &shift_gaugemul_p_cpu);
    m.impl("dirac_wilson_call", &dw_call_p_cpu);
    m.impl("dirac_wilson_clover_call", &dwc_call_p_cpu);
    m.impl("plaquette_action", &plaq_action_cpu);
}

}
