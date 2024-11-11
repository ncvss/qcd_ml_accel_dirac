#include <torch/extension.h>
#include <vector>


namespace qcd_ml_accel_dirac{

/**
 * @brief apply the dirac wilson operator to a vector field
 * 
 * @param U the gauge field configuration, (4,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param v the vector field
 * @param mass the mass parameter
 * @return at::Tensor the vector field after the operator action
 */
at::Tensor dw_call_cpu (const at::Tensor& U, const at::Tensor& v, double mass);


/**
 * @brief apply the dirac wilson clover operator to a vector field
 * 
 * @param U the gauge field configuration, (4,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param v the vector field, (Lx,Ly,Lz,Lt,4,3)-tensor
 * @param F precomputed field strength matrices from the clover terms
 * @param mass the mass parameter
 * @param csw the clover term prefactor
 * @return at::Tensor the vector field after the operator action
 */
at::Tensor dwc_call_cpu (const at::Tensor& U, const at::Tensor& v, const std::vector<at::Tensor>& F,
                         double mass, double csw);

// test the dirac wilson call using only c++ instead of torch
// There is no input, the fields are chosen randomly in the function
// the output is the time taken
double dw_call_c_test(at::Tensor dummy);

at::Tensor dw_call_c_correct (const at::Tensor& Ufl, const at::Tensor& vfl,
                              std::vector<int64_t> u_size, std::vector<int64_t> v_size, double mass);
}
