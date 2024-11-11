#include <torch/extension.h>

namespace qcd_ml_accel_dirac {

/**
 * @brief function that computes a*b+c component wise,
 *        to test the maximum possible memory throughput, not parallelized
 * 
 * @param a Tensor
 * @param b Tensor (same shape)
 * @param c Tensor (same shape)
 * @return at::Tensor
 */
at::Tensor muladd_bench_nopar(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c);

/**
 * @brief function that computes a*b+c component wise,
 *        to test the maximum possible memory throughput, parallelised
 * 
 * @param a Tensor
 * @param b Tensor (same shape)
 * @param c Tensor (same shape)
 * @return at::Tensor
 */
at::Tensor muladd_bench_par(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c);

/**
 * @brief function that computes a*b+c component wise, not parallelised, measures the time it takes itself
 * 
 * @param a Tensor
 * @param b Tensor (same shape)
 * @param c Tensor (same shape)
 * @return at::Tensor, first component is time taken
 */
at::Tensor muladd_bench_nopar_time(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c);

/**
 * @brief function that computes a*b+c component wise, parallelised, measures the time it takes itself
 * 
 * @param a Tensor
 * @param b Tensor (same shape)
 * @param c Tensor (same shape)
 * @return at::Tensor, first component is time taken
 */
at::Tensor muladd_bench_par_time(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c);

/**
 * @brief computes a[i]*b[i] for input of type double
 * 
 * @param a 
 * @param b 
 * @return at::Tensor 
 */
at::Tensor mul_real_bench_nopar(const at::Tensor& a, const at::Tensor& b);

/**
 * @brief computes a[i]*b[i] for input of type c10::complex<double>
 * 
 * @param a 
 * @param b 
 * @return at::Tensor 
 */
at::Tensor mul_compl_bench_nopar(const at::Tensor& a, const at::Tensor& b);

}