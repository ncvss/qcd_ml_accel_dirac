#include <immintrin.h>

namespace qcd_ml_accel_dirac {

/**
 * @brief Load 2 spin components s and s+1 from a vector field with the memory
 *        layout v[t,s,g] into a 256 bit register.
 *        The register will look like | Re s | Im s | Re s+1 | Im s+1 |
 * 
 * @param addr pointer to the real part of the first spin component (s)
 * @return __m256d 
 */
inline __m256d load_split_spin (const double * addr){
    // high part of the register should be s+1, so the address is increased by 6
    return _mm256_loadu2_m128d(addr+6,addr);
}

/**
 * @brief Load 2 spin components s and s+1 from a vector field with the memory
 *        layout v[t,s,g] into a 256 bit register, with the 2 components swapped.
 *        The register will look like | Re s+1 | Im s+1 | Re s | Im s |
 * 
 * @param addr pointer to the real part of the first spin component (s) in the vector field
 * @return __m256d 
 */
inline __m256d load_split_spin_sw (const double * addr){
    // low part of the register should be s+1, so the address is increased by 6
    return _mm256_loadu2_m128d(addr,addr+6);
}

/**
 * @brief Store the numbers from a 256 bit register as 2 spin components in a vector field
 *        with memory layout v[t,s,g].
 * 
 * @param addr pointer to the real part of the first spin component (s) in the vector field
 * @param a register that contains th numbers | Re s | Im s | Re s+1 | Im s+1 |
 */
inline void store_split_spin (double * addr, __m256d a){
    _mm256_storeu2_m128d(addr+6,addr,a);
}

}
