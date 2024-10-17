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
at::Tensor dw_call_p_cpu (const at::Tensor& U, const at::Tensor& v, double mass);

// also the dirac wilson, but rearranged
at::Tensor dw_call_rearr_cpu (const at::Tensor& U, const at::Tensor& v, double mass);

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
at::Tensor dwc_call_p_cpu (const at::Tensor& U, const at::Tensor& v, const std::vector<at::Tensor>& F,
                     double mass, double csw);

// dirac wilson clover rearranged
at::Tensor dwc_call_rearr_cpu (const at::Tensor& U, const at::Tensor& v, const std::vector<at::Tensor>& F,
                     double mass, double csw);

/**
 * @brief call the domain wall dirac operator on a vector field
 * 
 * @param U the gauge field configuration, (4,Ls,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param v the vector field, (Ls,Lx,Ly,Lz,Lt,4,3)-tensor
 * @param mass the quark mass parameter
 * @param m5 the domain wall mass parameter
 * @return at::Tensor the vector field after operator action
 */
at::Tensor domainwall_call_cpu (const at::Tensor& U, const at::Tensor& v, double mass, double m5);

}
