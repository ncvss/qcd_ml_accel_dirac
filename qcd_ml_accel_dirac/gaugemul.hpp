#include <torch/extension.h>
#include <vector>


namespace qcd_ml_accel_dirac{

/**
 * @brief multiply a gauge field with a gauge or vector field, with position shifts
 * 
 * @param U2 a gauge field, (Lx,Ly,Lz,Lt,3,3)-tensor
 * @param Uv a gauge field or vector field, (Lx,Ly,Lz,Lt,n,3)-tensor. n=3 corresponds to gauge, otherwise vector field.
 * @param u2shifts the positional shifts of U2 in each lattice axis
 * @param uvshifts the positional shifts of U2 in each lattice axis
 * @return at::Tensor the product
 */
at::Tensor shift_gaugemul_p_cpu (const at::Tensor& U2, const at::Tensor& Uv,
                                std::vector<int64_t> u2shifts, std::vector<int64_t> uvshifts);

}
