#include <torch/extension.h>
#include <vector>


namespace qcd_ml_accel_dirac{

/**
 * @brief Apply the dirac wilson operator to a vector field.
 * 
 * @param U the gauge field configuration, (4,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param v the vector field
 * @param mass the mass parameter
 * @return at::Tensor, the vector field after the operator action
 */
at::Tensor dw_call_cpu (const at::Tensor& U, const at::Tensor& v, double mass);


/**
 * @brief Apply the Dirac Wilson operator on the even and odd sites separately.
 *        The even and odd tensors contain in entry (x,y,z,t') the sites (x,y,z,t), where the
 *        t axis is shrunk in half and t is chosen to be only on even/odd sites.
 *        Thus, if V is the total number of lattice sites, the even/odd tensors only have V/2 sites.
 * 
 * @param Ue gauge fields on even sites, (4,V/2,3,3)-tensor
 * @param Uo gauge fields on odd sites, (4,V/2,3,3)-tensor
 * @param ve vector on even sites, (V/2,4,3)-tensor
 * @param vo vector on odd sites, (V/2,4,3)-tensor
 * @param mass the mass parameter
 * @param eodim the lattice dimensions for the even/odd arrays (so t is halved)
 * @return at::Tensor with shape (2,V/2,4,3)
 */
at::Tensor dw_call_eo (const at::Tensor& Ue, const at::Tensor& Uo,
                       const at::Tensor& ve, const at::Tensor& vo,
                       double mass, std::vector<int64_t> eodim);

/**
 * @brief Apply the Dirac Wilson operator with a different algorithm.
 *        The grid points are accessed with one flattened index of length vol=Lx*Ly*Lt*Lt,
 *        this allows for both flattened and multi-dimensional input grids.
 *        The shifts are taken from a lookup table.
 *        Also, the axis ordering is different.
 *        
 * @param U gauge fields, (vol,4,3,3)-tensor (xyzt,mu,g,gi)
 * @param v vector fields, (vol,3,4)-tensor (xyzt,g,s)
 * @param hops lookup table that contains a list [-x,+x,-y,+y,-z,+z,-t,+t] at each grid site
 *             whose entries are the flattened indices of the site that is reached after
 *             taking that step from the current site
 * @param mass mass parameter
 * @return at::Tensor 
 */
at::Tensor dw_call_lookup_cpu (const at::Tensor& U, const at::Tensor& v, const at::Tensor& hops, double mass);


/**
 * @brief Apply the dirac wilson clover operator to a vector field.
 * 
 * @param U the gauge field configuration, (4,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param v the vector field, (Lx,Ly,Lz,Lt,4,3)-tensor
 * @param F precomputed field strength matrices from the clover terms
 * @param mass the mass parameter
 * @param csw the clover term prefactor
 * @return at::Tensor, the vector field after the operator action
 */
at::Tensor dwc_call_cpu (const at::Tensor& U, const at::Tensor& v, const std::vector<at::Tensor>& F,
                         double mass, double csw);

/**
 * @brief Apply the shamir domain wall dirac operator to a vector field in 5 dimensions.
 * 
 * @param U the gauge field configuration, (4,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param v the domain wall fermion field, (Ls,Lx,Ly,Lz,Lt,4,3)-tensor
 * @param mass the bare mass parameter
 * @param m5 the bulk mass parameter
 * @return at::Tensor, the vector field after the operator action
 */
at::Tensor domain_wall_call_cpu (const at::Tensor& U, const at::Tensor& v, double mass, double m5);

}
