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
 * @brief Apply the dirac wilson operator,
 *        but replace the next neighbour contributions by the local contribution.
 *        Used for finding performance bottlenecks.
 * 
 * @param U 
 * @param v 
 * @param mass 
 * @return at::Tensor 
 */
at::Tensor dw_call_nohop (const at::Tensor& U, const at::Tensor& v, double mass);

/**
 * @brief Apply the dirac wilson operator,
 *        but replace the next neighbour contributions by the local contribution.
 *        Also, U has a different layout here.
 *        Used for finding performance bottlenecks.
 * 
 * @param U 
 * @param v 
 * @param mass 
 * @return at::Tensor 
 */
at::Tensor dw_call_nohop_Usw (const at::Tensor& U, const at::Tensor& v, double mass);

/**
 * @brief Apply the dirac wilson operator,
 *        but replace the next neighbour contributions by the local contribution
 *        and use the same gauge field for all directions.
 *        Used for finding performance bottlenecks.
 * 
 * @param U 
 * @param v 
 * @param mass 
 * @return at::Tensor 
 */
at::Tensor dw_call_nohop_1U (const at::Tensor& U, const at::Tensor& v, double mass);


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


}
