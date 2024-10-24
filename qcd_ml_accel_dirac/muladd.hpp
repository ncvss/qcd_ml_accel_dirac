#include <torch/extension.h>

namespace qcd_ml_accel_dirac {

/**
 * @brief function to test the maximum possible memory throughput, not parallelized
 * 
 * @param a Tensor
 * @param b Tensor (same shape)
 * @param c Tensor (same shape)
 * @return at::Tensor, contains a*b+c component wise
 */
at::Tensor muladd_bench_nopar(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c);

/**
 * @brief function to test the maximum possible memory throughput, parallelised
 * 
 * @param a Tensor
 * @param b Tensor (same shape)
 * @param c Tensor (same shape)
 * @return at::Tensor, contains a*b+c component wise
 */
at::Tensor muladd_bench_par(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c);

}