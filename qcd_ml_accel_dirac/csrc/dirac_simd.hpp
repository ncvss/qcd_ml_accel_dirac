#include <torch/extension.h>


namespace qcd_ml_accel_dirac{

/**
 * @brief apply the dirac wilson operator to a vector field, using AVX instructions
 * 
 * @param U_tensor the gauge field configuration, (4,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param v_tensor the vector field, (Lx,Ly,Lz,Lt,4,3)-tensor
 * @param hops_tensor lookup table for the indices for shifts by +1 and -1 in all directions
 * @param mass the mass parameter
 * @return at::Tensor 
 */
at::Tensor dw_call_256d_om_template (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                     const at::Tensor& hops_tensor, double mass);


/**
 * @brief apply the dirac wilson clover operator to a vector field, using AVX instructions
 * 
 * @param U_tensor the gauge field configuration, (4,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param v_tensor the vector field, (Lx,Ly,Lz,Lt,4,3)-tensor
 * @param fs_tensors (Lx,Ly,Lz,Lt,6,3,3)-tensor that contains the precomputed
 *                   field strength matrices
 * @param hops_tensor lookup table for the indices for shifts by +1 and -1 in all directions
 * @param mass the mass parameter
 * @param csw the Sheikholeslami-Wohlert coefficient
 * @return at::Tensor 
 */
at::Tensor dwc_call_256d_om_template (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                      const at::Tensor& fs_tensors, const at::Tensor& hops_tensor,
                                      double mass, double csw);

}
