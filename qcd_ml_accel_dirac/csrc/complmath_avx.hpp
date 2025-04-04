#pragma once

#include <immintrin.h>

namespace qcd_ml_accel_dirac{

/**
 * @brief multiply a scalar from a memory pointer onto 2 complex numbers
 *        in a 256 bit register
 * 
 * @param bp pointer to a complex number b, stored as 2 doubles (Re b, Im b)
 * @param a register that contains 2 complex numbers a1,a2, stored as 4 doubles
 *          (Re a1, Im a1, Re a2, Im a2)
 * @return __m256d 
 */
inline __m256d compl_scalarmem_vectorreg_mul (const double * bp, __m256d a){
    __m256d a_swap_reg = _mm256_permute_pd(a, 5); // a1i | a1r | a2i | a2r
    __m256d re_b_reg = _mm256_broadcast_sd(bp); // br | br | br | br
    __m256d im_b_reg = _mm256_broadcast_sd(bp+1); // bi | bi | bi | bi
    __m256d im_b_mul_reg = _mm256_mul_pd(a_swap_reg, im_b_reg); // a1i*bi | a1r*bi | a2i*bi | a2r*bi
    __m256d res_reg = _mm256_fmaddsub_pd(a, re_b_reg, im_b_mul_reg);
               // a1r*br-a1i*bi | a1i*br+a1r*bi | a2r*br-a2i*bi | a2i*br+a2r*bi
    return res_reg;
}

/**
 * @brief multiply the complex conjugate of a scalar from a memory pointer onto
 *        2 complex numbers in a 256 bit register
 * 
 * @param bp pointer to a complex number b, stored as 2 doubles (Re b, Im b)
 * @param a register that contains 2 complex numbers a1,a2, stored as 4 doubles
 *          (Re a1, Im a1, Re a2, Im a2)
 * @return __m256d 
 */
inline __m256d compl_scalarmem_conj_vectorreg_mul (const double * bp, __m256d a){
    __m256d a_swap_reg = _mm256_permute_pd(a, 5); // a1i | a1r | a2i | a2r
    __m256d re_b_reg = _mm256_broadcast_sd(bp); // br | br | br | br
    __m256d im_b_reg = _mm256_broadcast_sd(bp+1); // bi | bi | bi | bi
    __m256d im_b_mul_reg = _mm256_mul_pd(a_swap_reg, im_b_reg); // a1i*bi | a1r*bi | a2i*bi | a2r*bi
    __m256d res_reg = _mm256_fmsubadd_pd(a, re_b_reg, im_b_mul_reg);
               // a1r*br+a1i*bi | a1i*br-a1r*bi | a2r*br+a2i*bi | a2i*br-a2r*bi
    return res_reg;
}

/**
 * @brief pointwise multiply 2 256 bit registers that each contain 2 complex numbers
 *        (a1*b1, a2*b2)
 * 
 * @param a register that contains 2 complex numbers a1,a2, stored as 4 doubles
 *          (Re a1, Im a1, Re a2, Im a2)
 * @param b register that contains 2 complex numbers b1,b2, stored as 4 doubles
 *          (Re b1, Im b1, Re b2, Im b2)
 * @return __m256d 
 */
inline __m256d compl_vectorreg_pointwise_mul (__m256d a, __m256d b){
    __m256d a_swap_reg = _mm256_permute_pd(a, 5); // a1i | a1r | a2i | a2r

    // the following 2 operations can be done with
    // _mm256_permute_pd, _mm256_movedup_pd or _mm256_shuffle_pd
    // Grid uses movedup and shuffle
    // shuffle has the lowest CPI and equal latency, thus it should be used?
    __m256d re_b_reg = _mm256_shuffle_pd(b, b, 0); // b1r | b1r | b2r | b2r
    __m256d im_b_reg = _mm256_shuffle_pd(b, b, 15); // b1i | b1i | b2i | b2i
    
    __m256d im_b_mul_reg = _mm256_mul_pd(a_swap_reg, im_b_reg); // a1i*b1i | a1r*b1i | a2i*b2i | a2r*b2i
    __m256d res_reg = _mm256_fmaddsub_pd(a, re_b_reg, im_b_mul_reg);
               // a1r*b1r-a1i*b1i | a1i*b1r+a1r*b1i | a2r*b2r-a2i*b2i | a2i*b2r+a2r*b2i
    return res_reg;
}

/**
 * @brief pointwise multiply 2 256 bit registers that each contain 2 complex numbers,
 *        with the second getting conjugated (a1*(b1*), a2*(b2*))
 * 
 * @param a register that contains 2 complex numbers a1,a2, stored as 4 doubles
 *          (Re a1, Im a1, Re a2, Im a2)
 * @param b register that contains 2 complex numbers b1,b2, stored as 4 doubles
 *          (Re b1, Im b1, Re b2, Im b2)
 * @return __m256d 
 */
inline __m256d compl_vectorreg_conj_pointwise_mul (__m256d a, __m256d b){
    __m256d a_swap_reg = _mm256_permute_pd(a, 5); // a1i | a1r | a2i | a2r

    __m256d re_b_reg = _mm256_shuffle_pd(b, b, 0); // b1r | b1r | b2r | b2r
    __m256d im_b_reg = _mm256_shuffle_pd(b, b, 15); // b1i | b1i | b2i | b2i
    
    __m256d im_b_mul_reg = _mm256_mul_pd(a_swap_reg, im_b_reg); // a1i*b1i | a1r*b1i | a2i*b2i | a2r*b2i
    __m256d res_reg = _mm256_fmsubadd_pd(a, re_b_reg, im_b_mul_reg);
               // a1r*b1r+a1i*b1i | a1i*b1r-a1r*b1i | a2r*b2r+a2i*b2i | a2i*b2r-a2r*b2i
    return res_reg;
}

/**
 * @brief pointwise multiply 2 256 bit registers that each contain 2 complex numbers,
 *        with the first number being purely real
 * 
 * @param a register that contains 2 real numbers (a1, ., a2, .) (imaginary parts
 *          are ignored)
 * @param b register that contains 2 complex numbers b1,b2, stored as 4 doubles
 *          (Re b1, Im b1, Re b2, Im b2)
 * @return __m256d 
 */
inline __m256d realxcompl_vectorreg_mul (__m256d a, __m256d b){
    __m256d re_a_reg = _mm256_shuffle_pd(a, a, 0); // a1 | a1 | a2 | a2
    return _mm256_mul_pd(re_a_reg, b);
}

/**
 * @brief pointwise multiply 2 256 bit registers that each contain 2 complex numbers,
 *        with the first number being purely real
 * 
 * @param a register that contains 2 purely imaginary numbers a1, a2 in the format
 *          (a1,a1,a2,a2)
 * @param b register that contains 2 complex numbers b1,b2, stored as 4 doubles
 *          (Re b1, Im b1, Re b2, Im b2)
 * @return __m256d 
 */
inline __m256d imagxcompl_vectorreg_mul (__m256d a, __m256d b){
    __m256d b_swap_reg = _mm256_shuffle_pd(b, b, 5); // b1i | b1r | b2i | b2r
    __m256d b_mul_reg = _mm256_mul_pd(a, b_swap_reg); // a1i*b1i | a1i*b1r | a2i*b2i | a2i*b2r
    return _mm256_addsub_pd(_mm256_setzero_pd(), b_mul_reg); // -a1i*b1i | a1i*b1r | -a2i*b2i | a2i*b2r
}


inline __m256d compl_reg_times_i (__m256d a){
    __m256d a_swap_reg = _mm256_shuffle_pd(a, a, 5); // a1i | a1r | a2i | a2r
    return _mm256_addsub_pd(_mm256_setzero_pd(), a_swap_reg); // -a1i | a1r | -a2i | a2r
}

inline __m256d compl_reg_times_minus_i (__m256d a){
    __m256d minus_re = _mm256_addsub_pd(_mm256_setzero_pd(), a); // -a1r | a1i | -a2r | a2i
    return _mm256_shuffle_pd(minus_re, minus_re, 5); // a1i | -a1r | a2i | -a2r
}

}
