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

/**
 * @brief apply the shamir domain wall dirac operator to a field
 * 
 * @param U the gauge field configuration, (4,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param v the domain wall fermion field, (Ls,Lx,Ly,Lz,Lt,4,3)-tensor
 * @param mass the bare mass parameter
 * @param m5 the bulk mass parameter
 * @return at::Tensor 
 */
at::Tensor domain_wall_call_cpu (const at::Tensor& U, const at::Tensor& v, double mass, double m5);

/**
 * @brief apply the dirac wilson operator to a vector field, using AVX instructions
 * 
 * @param Ut the gauge field configuration, (Lx,Ly,Lz,Lt,4,3,3)-tensor
 * @param vt the vector field, (Lx,Ly,Lz,Lt,3,4)-tensor
 * @param hops lookup table for the indices for shifts by +1 and -1 in all directions
 * @param mass the mass parameter
 * @return at::Tensor 
 */
at::Tensor dw_call_lookup_256d_cpu (const at::Tensor& Ut, const at::Tensor& vt, const at::Tensor& hops, double mass);

}
