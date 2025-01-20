#include <torch/extension.h>
#include <vector>

// the functions are defined in the respective files
// this file only contains the library implementation

#include "gaugemul.hpp"
#include "dirac.hpp"
#include "plaq.hpp"

namespace qcd_ml_accel_dirac{

at::Tensor convert_complex_to_double (const at::Tensor& complt){
    // in the following way, complex tensors can be accessed as doubles
    at::Tensor result = torch::zeros_like(complt);
    const double* c_ptr = (double*)complt.const_data_ptr<c10::complex<double>>();
    double* res_ptr = (double*)result.mutable_data_ptr<c10::complex<double>>();
    res_ptr[1] = 365;
    res_ptr[0] = c_ptr[1];
    res_ptr[2] = c_ptr[2]-c_ptr[3];
    res_ptr[3] = c_ptr[2]+c_ptr[3];
    return result;
}

// Registers _C as a Python extension module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(qcd_ml_accel_dirac, m) {
    m.def("shift_gaugemul(Tensor U2, Tensor Uv, int[] u2shifts, int[] uvshifts) -> Tensor");
    m.def("dirac_wilson_call(Tensor U, Tensor v, float mass) -> Tensor");
    m.def("dirac_wilson_clover_call(Tensor U, Tensor v, Tensor[] F, float mass, float csw) -> Tensor");
    m.def("plaquette_action(Tensor U, float g) -> float");
    m.def("domain_wall_dirac_call(Tensor U, Tensor v, float mass, float m5) -> Tensor");

    m.def("convert_complex_to_double(Tensor complt) -> Tensor");
#ifdef CPU_IS_AVX_CAPABLE
    m.def("dw_call_lookup_256d(Tensor U_tensor, Tensor v_tensor, Tensor hops_tensor, float mass) -> Tensor");
    m.def("dwc_call_lookup_256d(Tensor U_tensor, Tensor v_tensor, Tensor fs_tensor, Tensor hops_tensor, float mass, float csw) -> Tensor");
    m.def("dw_call_lookup_256d_old(Tensor U_tensor, Tensor v_tensor, Tensor hops_tensor, float mass) -> Tensor");
    m.def("dw_call_256d_om_template(Tensor U_tensor, Tensor v_tensor, Tensor hops_tensor, float mass) -> Tensor");
    m.def("dwc_call_lookup_256d_old(Tensor U_tensor, Tensor v_tensor, Tensor fs_tensor, Tensor hops_tensor, float mass, float csw) -> Tensor");
    m.def("dwc_call_256d_om_template(Tensor U_tensor, Tensor v_tensor, Tensor fs_tensor, Tensor hops_tensor, float mass, float csw) -> Tensor");
#endif
}

// Registers backend implementations
TORCH_LIBRARY_IMPL(qcd_ml_accel_dirac, CPU, m) {
    m.impl("shift_gaugemul", &shift_gaugemul_cpu);
    m.impl("dirac_wilson_call", &dw_call_cpu);
    m.impl("dirac_wilson_clover_call", &dwc_call_cpu);
    m.impl("plaquette_action", &plaq_action_cpu);
    m.impl("domain_wall_dirac_call", &domain_wall_call_cpu);

    m.impl("convert_complex_to_double", &convert_complex_to_double);
#ifdef CPU_IS_AVX_CAPABLE
    m.impl("dw_call_lookup_256d", &dw_call_lookup_256d_cpu);
    m.impl("dwc_call_lookup_256d", &dwc_call_lookup_256d_cpu);
    m.impl("dw_call_lookup_256d_old", &dw_call_lookup_256d_old_layout);
    m.impl("dw_call_256d_om_template", &dw_call_256d_om_template);
    m.impl("dwc_call_lookup_256d_old", &dwc_call_lookup_256d_old_layout);
    m.impl("dwc_call_256d_om_template", &dwc_call_256d_om_template);
#endif
}

}
