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
 * @brief Computes a[i]*b[i] for input of type double.
 * 
 * @param a 
 * @param b 
 * @return at::Tensor 
 */
at::Tensor mul_real_bench_nopar(const at::Tensor& a, const at::Tensor& b);

/**
 * @brief Computes a[i]*b[i] for input of type c10::complex<double>.
 * 
 * @param a 
 * @param b 
 * @return at::Tensor 
 */
at::Tensor mul_compl_bench_nopar(const at::Tensor& a, const at::Tensor& b);


// note: the following functions take gauge field tensors of different sizes,
// not necessarily of the usual size 4x[lattice]x3x3,
// as the partial computations need a varying number of gauge links

/**
 * @brief Multiply a gauge matrix onto a vector at each lattice site.
 * 
 * @param U 
 * @param v 
 * @return at::Tensor 
 */
at::Tensor gauge_transform (const at::Tensor& U, const at::Tensor& v);

/**
 * @brief Multiply a gauge matrix onto a vector at each lattice site, parallel.
 * 
 * @param U 
 * @param v 
 * @return at::Tensor 
 */
at::Tensor gauge_transform_par (const at::Tensor& U, const at::Tensor& v);

/**
 * @brief Multiply the gamma_1 matrix onto a vector, then multiply a gauge matrix onto vector + gamma * vector.
 * 
 * @param U 
 * @param v 
 * @return at::Tensor 
 */
at::Tensor gauge_transform_gamma (const at::Tensor& U, const at::Tensor& v);

/**
 * @brief Shift a vector by +1 in y, multiply the gamma_1 matrix onto it,
 *        then multiply a gauge matrix onto vector + gamma * vector.
 * 
 * @param U 
 * @param v 
 * @return at::Tensor 
 */
at::Tensor gauge_transform_gamma_shift (const at::Tensor& U, const at::Tensor& v);

/**
 * @brief Shift a vector by +1 in y, multiply the gamma_1 matrix onto it,
 *        then multiply a gauge matrix onto vector + gamma_1 * vector.
 *        Do the same for a shift by -1 in y for both vector and gauge matrix, and add it.
 *        This is the gauge hop prescription without complex conjugation.
 *        It is roughly equivalent to the computation of:
 * 
 *        (H_+1 + H_-1) (v + gamma_1 v)
 * 
 * @param U 
 * @param v 
 * @return at::Tensor 
 */
at::Tensor gauge_transform_gamma_2shift (const at::Tensor& U, const at::Tensor& v);

/**
 * @brief Shift a vector by +1 in y, multiply the gamma_1 matrix onto it,
 *        then multiply a gauge matrix onto vector + gamma_1 * vector.
 *        In a separate loop, do the same for a shift by -1 in y for both vector and gauge matrix, and add it.
 * 
 * @param U 
 * @param v 
 * @return at::Tensor 
 */
at::Tensor gauge_transform_gamma_2shift_split (const at::Tensor& U, const at::Tensor& v);

/**
 * @brief Shift a vector by +1 in y, multiply the gamma_1 matrix onto it,
 *        then multiply a gauge matrix onto vector + gamma_1 * vector.
 *        Do the same for a shift by -1 in y for both vector and gauge matrix, and add it.
 *        Here, y is the innermost loop.
 * 
 * @param U 
 * @param v 
 * @return at::Tensor 
 */
at::Tensor gauge_transform_gamma_2shift_ysw (const at::Tensor& U, const at::Tensor& v);

/**
 * @brief Shift a vector by +1 in t, multiply the gamma_3 matrix onto it,
 *        then multiply a gauge matrix onto vector + gamma_3 * vector.
 *        Do the same for a shift by -1 in t for both vector and gauge matrix, and add it.
 *        This is the gauge hop prescription without complex conjugation.
 *        It is roughly equivalent to the computation of:
 * 
 *        (H_+3 + H_-3) (v + gamma_3 v)
 * 
 * @param U 
 * @param v 
 * @return at::Tensor 
 */
at::Tensor gauge_transform_gamma_2tshift (const at::Tensor& U, const at::Tensor& v);

/**
 * @brief Do the above shift, gamma multiplication and gauge multiplication for both y and t.
 *        This is roughly equivalent to the computation of:
 * 
 *        (H_+1 + H_-1) (v + gamma_1 v) + (H_+3 + H_-3) (v + gamma_3 v)
 * 
 * @param U 
 * @param v 
 * @return at::Tensor 
 */
at::Tensor gauge_transform_gamma_2ytshift (const at::Tensor& U, const at::Tensor& v);

/**
 * @brief Do roughly the following computation:
 *        
 *        (H_+1 + H_-3) v
 * 
 * @param U 
 * @param v 
 * @return at::Tensor 
 */
at::Tensor gauge_transform_simple_ytshift (const at::Tensor& U, const at::Tensor& v);

/**
 * @brief Matrix multiply U onto gamma * v + y-shifted v
 * 
 * @param U 
 * @param v 
 * @return at::Tensor 
 */
at::Tensor gauge_transform_gamma_2point (const at::Tensor& U, const at::Tensor& v);

}