#include <torch/extension.h>
#include <vector>

// the functions are defined in the respective files
// this file only contains the library implementation

#include "gaugemul.hpp"
#include "dirac.hpp"
#include "plaq.hpp"
#include "dirac_simd.hpp"

namespace qcd_ml_accel_dirac{


// Registers _C as a Python extension module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(qcd_ml_accel_dirac, m) {
    m.def("shift_gaugemul(Tensor U2, Tensor Uv, int[] u2shifts, int[] uvshifts) -> Tensor");
    m.def("dirac_wilson_call(Tensor U, Tensor v, float mass) -> Tensor");
    m.def("dirac_wilson_clover_call(Tensor U, Tensor v, Tensor[] F, float mass, float csw) -> Tensor");
    m.def("plaquette_action(Tensor U, float g) -> float");
    m.def("domain_wall_dirac_call(Tensor U, Tensor v, float mass, float m5) -> Tensor");

    m.def("dw_call_256d_template(Tensor U_tensor, Tensor v_tensor, Tensor hops_tensor, float mass) -> Tensor");
    m.def("dwc_call_256d_template(Tensor U_tensor, Tensor v_tensor, Tensor fs_tensor, Tensor hops_tensor, float mass, float csw) -> Tensor");
}

// Registers backend implementations
TORCH_LIBRARY_IMPL(qcd_ml_accel_dirac, CPU, m) {
    m.impl("shift_gaugemul", &shift_gaugemul_cpu);
    m.impl("dirac_wilson_call", &dw_call_cpu);
    m.impl("dirac_wilson_clover_call", &dwc_call_cpu);
    m.impl("plaquette_action", &plaq_action_cpu);
    m.impl("domain_wall_dirac_call", &domain_wall_call_cpu);

    m.impl("dw_call_256d_template", &dw_call_256d_template);
    m.impl("dwc_call_256d_template", &dwc_call_256d_template);
}

}
