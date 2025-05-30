#include <torch/extension.h>


namespace qcd_ml_accel_dirac{

/**
 * @brief apply the wilson dirac operator to a vector field, using AVX instructions
 * 
 * @param U_tensor the gauge field configuration, (4,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param v_tensor the vector field, (Lx,Ly,Lz,Lt,4,3)-tensor
 * @param hops_tensor lookup table for the indices for shifts by -1 and +1 in all directions
 * @param bound_tensor lookup table for the phase after a shift by -1 and +1 in alll directions
 * @param mass the mass parameter
 * @return at::Tensor 
 */
at::Tensor dw_avx_templ (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                         const at::Tensor& hops_tensor, const at::Tensor& bound_tensor,
                         double mass);


/**
 * @brief apply the wilson clover dirac operator to a vector field, using AVX instructions
 * 
 * @param U_tensor the gauge field configuration, (4,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param v_tensor the vector field, (Lx,Ly,Lz,Lt,4,3)-tensor
 * @param fs_tensors (Lx,Ly,Lz,Lt,6,3,3)-tensor that contains the precomputed
 *                   field strength matrices
 * @param hops_tensor lookup table for the indices for shifts by -1 and +1 in all directions
 * @param mass the mass parameter
 * @param csw the Sheikholeslami-Wohlert coefficient
 * @return at::Tensor 
 */
at::Tensor dwc_avx_templ (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                          const at::Tensor& fs_tensors, const at::Tensor& hops_tensor,
                          double mass, double csw);

/**
 * @brief apply the wilson clover dirac operator to a vector field, using AVX instructions
 *        and the computation taken from grid
 * 
 * @param U_tensor the gauge field configuration, (4,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param v_tensor the vector field, (Lx,Ly,Lz,Lt,4,3)-tensor
 * @param fs_tensors (Lx,Ly,Lz,Lt,21,2)-tensor that contains the upper triangle of the
 *                   precomputed tensor product of field strength and sigma, the lower
 *                   triangle is deduced from hermiticity
 * @param hops_tensor lookup table for the indices for shifts by -1 and +1 in all directions
 * @param bound_tensor lookup table for the phase after a shift by -1 and +1 in alll directions
 * @param mass the mass parameter
 * @return at::Tensor 
 */
at::Tensor dwc_avx_templ_grid (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                               const at::Tensor& fs_tensors, const at::Tensor& hops_tensor,
                               const at::Tensor& bound_tensor, double mass);


/**
 * @brief apply the backward pass of the wilson dirac operator to the gradients,
 *        using AVX instructions
 * 
 * @param U_tensor the saved gauge field configuration, (4,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param grad_tensor the gradients from Pytorch
 * @param hops_tensor lookup table for the indices for shifts by -1 and +1 in all directions
 * @param bound_tensor lookup table for the phase after a shift by -1 and +1 in alll directions
 * @param mass the mass parameter
 * @return at::Tensor 
 */
at::Tensor dw_avx_templ_backw (const at::Tensor& U_tensor, const at::Tensor& grad_tensor,
                               const at::Tensor& hops_tensor, const at::Tensor& bound_tensor,
                               double mass);

/**
 * @brief apply the backward pass of the wilson clover dirac operator to the gradients,
 *        using AVX instructions
 * 
 * @param U_tensor the saved gauge field configuration, (4,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param grad_tensor the gradients from Pytorch
 * @param fs_tensors the saved field strength x sigma tensor product,
 *                   (Lx,Ly,Lz,Lt,21,2)-tensor that contains the upper triangle
 * @param hops_tensor lookup table for the indices for shifts by -1 and +1 in all directions
 * @param bound_tensor lookup table for the phase after a shift by -1 and +1 in alll directions
 * @param mass the mass parameter
 * @return at::Tensor 
 */
at::Tensor dwc_avx_templ_grid_backw (const at::Tensor& U_tensor, const at::Tensor& grad_tensor,
                                     const at::Tensor& fs_tensors, const at::Tensor& hops_tensor,
                                     const at::Tensor& bound_tensor, double mass);

}
