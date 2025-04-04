#include <immintrin.h>

#include "complmath_avx.hpp"

namespace qcd_ml_accel_dirac{

// gamma index for 2 packed spin components
// s=0 is for the spin components 0 and 1, s=2 for the spin components 2 and 3
static const int gamx_pd [3] = {2, 0, 0};

// prefactor for the multiplication with gamma_mu
// it is always either real or imaginary
// 2 adjacent doubles in this array are both either the real or imaginary part
// the handling whether it is real or imaginary is done by gamma_mul
static const double gamf_temp [24] =
    { 1, 1,   1, 1,  -1,-1,  -1,-1, // imag
     -1,-1,   1, 1,   1, 1,  -1,-1, // real
      1, 1,  -1,-1,  -1,-1,   1, 1, // imag
      //1, 1,   1, 1,   1, 1,   1, 1 // real
      };
// gamf = [[ i, i,-i,-i],
//         [-1, 1, 1,-1],
//         [ i,-i,-i, i],
//         [ 1, 1, 1, 1] ]

// multiplication with gamf as a template function
template <int M, int S> inline __m256d gamma_mul (__m256d a){
    if constexpr (M == 3){
        return a;
    } else {
        __m256d g_reg = _mm256_loadu_pd(gamf_temp+gixd(M,S));
        if constexpr (M == 0 || M == 2){
            return imagxcompl_vectorreg_mul(g_reg, a);
        } else {
            return _mm256_mul_pd(g_reg,a);
        }
    }
}


// for 2 packed spin components, the sigma index is just the spin
// the two numbers have to be swapped for munu=1,2,3,4

// prefactor for the multiplication with sigma_munu
// it is always either real or imaginary
// 2 adjacent doubles in this array are both either the real or imaginary part
// the handling whether it is real or imaginary is done by sigma_mul
static const double sigf_temp [48] = {
      1, 1,  -1,-1,   1, 1,  -1,-1,  // imag
      1, 1,  -1,-1,   1, 1,  -1,-1,  // real
      1, 1,   1, 1,   1, 1,   1, 1,  // imag
     -1,-1,  -1,-1,   1, 1,   1, 1,  // imag
      1, 1,  -1,-1,  -1,-1,   1, 1,  // real
     -1,-1,   1, 1,   1, 1,  -1,-1   // imag
     };
// sigf = [[ i,-i, i,-i],
//         [ 1,-1, 1,-1],
//         [ i, i, i, i],
//         [-i,-i, i, i],
//         [ 1,-1,-1, 1],
//         [-i, i, i,-i] ]

// multiplication with sigf as a template function
template <int MN, int S> inline __m256d sigma_mul (__m256d a){
    __m256d s_reg = _mm256_loadu_pd(sigf_temp+sixd(MN,S));
    if constexpr (MN == 0 || MN == 2 || MN == 3 || MN == 5){
        return imagxcompl_vectorreg_mul(s_reg, a);
    } else {
        return _mm256_mul_pd(s_reg,a);
    }
}

}
