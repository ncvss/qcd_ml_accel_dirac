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
 * @brief Apply the Dirac Wilson operator on the even and odd sites separately.
 *        The even tensors contain in entry (x,y,z,t') the sites (x,y,z,t), where the
 *        t axis is shrunk in half and t is chosen to be only on even sites.
 *        The odd tensors contain in entry (x,y,z,t') the odd site that is t+1 from the even
 *        site with the same (x,y,z,t').
 * 
 * @param Ue gauge fields on even sites
 * @param Uo gauge fields on odd sites
 * @param ve vector on even sites
 * @param vo vector on odd sites
 * @param mass 
 * @return at::Tensor 
 */
at::Tensor dw_call_eo (const at::Tensor& Ue, const at::Tensor& Uo,
                       const at::Tensor& ve, const at::Tensor& vo, double mass);


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

}
