#include <torch/extension.h>
#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

// the functions are defined in the respective files
// this file only contains the library implementation

#include "gaugemul.hpp"
#include "dirac.hpp"
#include "plaq.hpp"
#include "dirac_avx.hpp"


extern "C" {
    /* Creates a dummy empty _C module that can be imported from Python.
        The import from Python will load the .so consisting of this file
        in this extension, so that the TORCH_LIBRARY static initializers
        below are run. */
    PyObject* PyInit__C(void){
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",   /* name of module */
            NULL,   /* module documentation, may be NULL */
            -1,     /* size of per-interpreter state of the module,
                        or -1 if the module keeps state in global variables. */
            NULL,   /* methods */
        };
        return PyModule_Create(&module_def);
    }
}

namespace qcd_ml_accel_dirac{


// // Registers _C as a Python extension module
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(qcd_ml_accel_dirac, m) {
    m.def("shift_gaugemul(Tensor U2, Tensor Uv, int[] u2shifts, int[] uvshifts) -> Tensor");
    m.def("dirac_wilson_call(Tensor U, Tensor v, float mass) -> Tensor");
    m.def("dirac_wilson_clover_call(Tensor U, Tensor v, Tensor[] F, float mass, float csw) -> Tensor");
    m.def("plaquette_action(Tensor U, float g) -> float");
    m.def("domain_wall_dirac_call(Tensor U, Tensor v, float mass, float m5) -> Tensor");

    m.def("dw_avx_templ(Tensor U_tensor, Tensor v_tensor, Tensor hops_tensor, float mass) -> Tensor");
    m.def("dwc_avx_templ(Tensor U_tensor, Tensor v_tensor, Tensor fs_tensors, Tensor hops_tensor, float mass, float csw) -> Tensor");
    m.def("dwc_avx_templ_grid(Tensor U_tensor, Tensor v_tensor, Tensor fs_tensors, Tensor hops_tensor, float mass) -> Tensor");

    m.def("dw_avx_templ_backw(Tensor U_tensor, Tensor grad_tensor, Tensor hops_tensor, float mass) -> Tensor");
    m.def("dwc_avx_templ_grid_backw(Tensor U_tensor, Tensor grad_tensor, Tensor fs_tensors, Tensor hops_tensor, float mass) -> Tensor");
}

// Registers backend implementations
TORCH_LIBRARY_IMPL(qcd_ml_accel_dirac, CPU, m) {
    m.impl("shift_gaugemul", &shift_gaugemul_cpu);
    m.impl("dirac_wilson_call", &dw_call_cpu);
    m.impl("dirac_wilson_clover_call", &dwc_call_cpu);
    m.impl("plaquette_action", &plaq_action_cpu);
    m.impl("domain_wall_dirac_call", &domain_wall_call_cpu);

    m.impl("dw_avx_templ", &dw_avx_templ);
    m.impl("dwc_avx_templ", &dwc_avx_templ);
    m.impl("dwc_avx_templ_grid", &dwc_avx_templ_grid);

    m.impl("dw_avx_templ_backw", &dw_avx_templ_backw);
    m.impl("dwc_avx_templ_grid_backw", &dwc_avx_templ_grid_backw);
}

}
