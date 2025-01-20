#ifdef CPU_IS_AVX_CAPABLE

#include <torch/extension.h>
#include <omp.h>
#include <immintrin.h>
//#include <emmintrin.h>

#include "complmath256d.hpp"

namespace qcd_ml_accel_dirac{

// gamma index for 2 packed spin components
// s=0 is for the spin components 0 and 1, s=2 for th spin components 2 and 3
static const int gamx_pd [3] = {2, 0, 0};

// gamfd[mu][i] is the prefactor of spin component i of gammamu @ v
// complex numbers are stored as 2 doubles
static const double gamfd [32] =
    { 0, 1,   0, 1,   0,-1,   0,-1,
     -1, 0,   1, 0,   1, 0,  -1, 0,
      0, 1,   0,-1,   0,-1,   0, 1,
      1, 0,   1, 0,   1, 0,   1, 0 };
// gamf = [[ i, i,-i,-i],
//         [-1, 1, 1,-1],
//         [ i,-i,-i, i],
//         [ 1, 1, 1, 1] ]

inline __attribute__((always_inline)) int gixd (int mu, int s);

static const double gamf_temp [24] =
    { 1, 1,   1, 1,  -1,-1,  -1,-1, // imag
     -1,-1,   1, 1,   1, 1,  -1,-1, // real
      1, 1,  -1,-1,  -1,-1,   1, 1, // imag
      //1, 1,   1, 1,   1, 1,   1, 1 // real
      };

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

// sigf[munu][i] is the prefactor of spin component i of sigmamunu @ v
// complex numbers are stored as 2 doubles
static const double sigfd [48] =
    { 0, 1,   0,-1,   0, 1,   0,-1,
      1, 0,  -1, 0,   1, 0,  -1, 0,
      0, 1,   0, 1,   0, 1,   0, 1,
      0,-1,   0,-1,   0, 1,   0, 1,
      1, 0,  -1, 0,  -1, 0,   1, 0,
      0,-1,   0, 1,   0, 1,   0,-1 };
// sigf = [[ i,-i, i,-i],
//         [ 1,-1, 1,-1],
//         [ i, i, i, i],
//         [-i,-i, i, i],
//         [ 1,-1,-1, 1],
//         [-i, i, i,-i] ]

static const double sigf_temp [48] = {
      1, 1,  -1,-1,   1, 1,  -1,-1,  // imag
      1, 1,  -1,-1,   1, 1,  -1,-1,  // real
      1, 1,   1, 1,   1, 1,   1, 1,  // imag
     -1,-1,  -1,-1,   1, 1,   1, 1,  // imag
      1, 1,  -1,-1,  -1,-1,   1, 1,  // real
     -1,-1,   1, 1,   1, 1,  -1,-1   // imag
     };

inline __attribute__((always_inline)) int sixd (int munu, int s);

// multiplication with sigf as a template function
template <int MN, int S> inline __m256d sigma_mul (__m256d a){
    __m256d s_reg = _mm256_loadu_pd(sigf_temp+sixd(MN,S));
    if constexpr (MN == 0 || MN == 2 || MN == 3 || MN == 5){
        return imagxcompl_vectorreg_mul(s_reg, a);
    } else {
        return _mm256_mul_pd(s_reg,a);
    }
}

// address for complex gauge field pointer that is stored as 2 doubles
inline __attribute__((always_inline)) int uixd (int t, int mu, int g, int gi){
    return t*72 + mu*18 + g*6 + gi*2;
}
// address for complex vector field pointer that is stored as 2 doubles
inline __attribute__((always_inline)) int vixd (int t, int g, int s){
    return t*24 + g*8 + s*2;
}
// address for hop pointer
inline __attribute__((always_inline)) int hixd (int t, int h, int d){
    return t*8 + h*2 + d;
}
// address for gamfd
inline __attribute__((always_inline)) int gixd (int mu, int s){
    return mu*8 + s*2;
}
// address for field strength tensors (index order F[t,munu,g,gi])
inline __attribute__((always_inline)) int fixd (int t, int munu, int g, int gi){
    return t*108 + munu*18 + g*6 + gi*2;
}
// address for sigfd
inline __attribute__((always_inline)) int sixd (int munu, int s){
    return munu*8 + s*2;
}

// addresses for old layout
inline int uixo (int t, int mu, int g, int gi, int vol){
    return mu*vol*18 + t*18 + g*6 + gi*2;
}
inline int vixo (int t, int g, int s){
    return t*24 + s*6 + g*2;
}

// load function for v with the old layout
inline __m256d load_split_spin (const double * addr){
    // high part of the register should be s+1, so the address is increased by 6
    return _mm256_loadu2_m128d(addr+6,addr);
}
// load function for v with the old layout, that swaps the values
inline __m256d load_split_spin_sw (const double * addr){
    // low part of the register should be s+1, so the address is increased by 6
    return _mm256_loadu2_m128d(addr,addr+6);
}
// store in v with the old layout
inline void store_split_spin (double * addr, __m256d a){
    _mm256_storeu2_m128d(addr+6,addr,a);
}

// // load the spin components s and s+1 of gamma_mu @ v into a register
// // gamma_mu swaps around the spin components of v in the following way:
// // - for mu=0 or mu=1, the order is 3,2,1,0
// // - for mu=2 or mu=3, the order is 2,3,0,1
// // this means: for mu=0,1 our register will be (v3,v2), for mu=2,3 it will be (v2,v3)
// // gamf_reg is a register with the correct prefactor
// template <int mu>
// inline __m256d load_vec_times_gamma_mu (double + v, int gi, int s, __m256d gamf_reg){
//     if constexpr (mu == 0 || mu == 1) {
//         // 
//         __m256d v_x_gamma = _mm256_loadu_pd(v,gi,gamx_pd[s]);
//         v_x_gamma = _mm256_permute4x64_pd(v_x_gamma,78);
//         v_x_gamma = compl_vectorreg_pointwise_mul(gamf_reg, v_x_gamma);
//         return v_x_gamma;
//     } else {
//         __m256d v_x_gamma = _mm256_loadu_pd(v,gi,gamx_pd[s]);
//         v_x_gamma = compl_vectorreg_pointwise_mul(gamf_reg, v_x_gamma);
//         return v_x_gamma;
//     }
// }



at::Tensor dw_call_lookup_256d_cpu (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                    const at::Tensor& hops_tensor, double mass){

    TORCH_CHECK(v_tensor.dim() == 6);
    TORCH_CHECK(U_tensor.size(0) == v_tensor.size(0));
    TORCH_CHECK(U_tensor.size(1) == v_tensor.size(1));
    TORCH_CHECK(U_tensor.size(2) == v_tensor.size(2));
    TORCH_CHECK(U_tensor.size(3) == v_tensor.size(3));
    TORCH_CHECK(v_tensor.size(4) == 3);
    TORCH_CHECK(v_tensor.size(5) == 4);

    TORCH_CHECK(U_tensor.dtype() == at::kComplexDouble);
    TORCH_CHECK(v_tensor.dtype() == at::kComplexDouble);

    TORCH_CHECK(U_tensor.is_contiguous());
    TORCH_CHECK(v_tensor.is_contiguous());

    int vol = U_tensor.size(0) * U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3);
    

    at::Tensor result_tensor = torch::empty(v_tensor.sizes(), v_tensor.options());

    // we create a pointer to the complex tensor, then typecast it to double*
    // this allows us to access the complex numbers as doubles in riri format
    const double* U = (double*)U_tensor.const_data_ptr<c10::complex<double>>();
    const double* v = (double*)v_tensor.const_data_ptr<c10::complex<double>>();
    const int* hops = hops_tensor.const_data_ptr<int>();
    double* result = (double*)result_tensor.mutable_data_ptr<c10::complex<double>>();

    // register for the mass prefactor
    __m256d massf_reg = _mm256_set1_pd(4.0 + mass);

    // register for the -1/2 prefactor
    __m256d m0p5_reg = _mm256_set1_pd(-0.5);


#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        // mu loop partially unrolled
        // for (int mu = 0; mu < 4; mu++){
        int mu = 0;
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s+=2){

                // mass term is the first term in the result
                __m256d incr = _mm256_loadu_pd(v+vixd(t,g,s));
                incr = _mm256_mul_pd(incr,massf_reg);

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register
                    v_Hmum_gam = _mm256_permute4x64_pd(v_Hmum_gam,78);
                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixd(hops[hixd(t,mu,0)],mu,gi,g),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register
                    v_Hmup_gam = _mm256_permute4x64_pd(v_Hmup_gam,78);
                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixd(t,mu,g,gi),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                }
                // store incr in result
                _mm256_storeu_pd(result+vixd(t,g,s),incr);
            }
        }
        mu = 1;
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s+=2){

                // load result of previous computations to add to
                __m256d incr = _mm256_loadu_pd(result+vixd(t,g,s));

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register
                    v_Hmum_gam = _mm256_permute4x64_pd(v_Hmum_gam,78);
                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixd(hops[hixd(t,mu,0)],mu,gi,g),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register
                    v_Hmup_gam = _mm256_permute4x64_pd(v_Hmup_gam,78);
                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixd(t,mu,g,gi),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                }
                // store incr in result
                _mm256_storeu_pd(result+vixd(t,g,s),incr);
            }
        }
        for (mu = 2; mu < 4; mu++){
            for (int g = 0; g < 3; g++){
                for (int s = 0; s < 4; s+=2){

                    // load result of previous computations to add to
                    __m256d incr = _mm256_loadu_pd(result+vixd(t,g,s));

                    // gamma_mu prefactor for the spins s and s+1
                    __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                    for (int gi = 0; gi < 3; gi++){
                        // v hop in negative mu * gammma
                        __m256d v_Hmum_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                        // multiply the gamma prefactor for s and s+1
                        v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                        // v hop in negative mu
                        __m256d v_Hmum = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,s));

                        // add those together
                        v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                        // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                        v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixd(hops[hixd(t,mu,0)],mu,gi,g),v_Hmum);


                        // v hop in positive mu * gamma
                        __m256d v_Hmup_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                        v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                        // v hop in positive mu
                        __m256d v_Hmup = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,s));

                        // subtract those 2
                        v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                        // multiply U at this point onto v sum
                        v_Hmup = compl_scalarmem_vectorreg_mul(U+uixd(t,mu,g,gi),v_Hmup);


                        // add both U*v terms
                        v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                        // *(-0.5) and add to incr
                        incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                    }
                    // store incr in result
                    _mm256_storeu_pd(result+vixd(t,g,s),incr);
                }
            }
        }
        //} end mu loop

    }


    return result_tensor;
}


// at::Tensor dw_call_lookup_256d_old_layout (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
//                                            const at::Tensor& hops_tensor, double mass){
    
//     // here the indices are the old layout: U[mu,x,y,z,t,g,gi] and v[x,y,z,t,s,g]
//     // which means the entire code does not work anymore!
//     // And there is no way to make it work

//     // there is of course the simple version to just manually load both spin
//     // contributions into one register

at::Tensor dw_call_lookup_256d_old_layout (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                           const at::Tensor& hops_tensor, double mass){

    TORCH_CHECK(v_tensor.dim() == 6);
    TORCH_CHECK(U_tensor.size(1) == v_tensor.size(0));
    TORCH_CHECK(U_tensor.size(2) == v_tensor.size(1));
    TORCH_CHECK(U_tensor.size(3) == v_tensor.size(2));
    TORCH_CHECK(U_tensor.size(4) == v_tensor.size(3));
    TORCH_CHECK(v_tensor.size(5) == 3);
    TORCH_CHECK(v_tensor.size(4) == 4);

    TORCH_CHECK(U_tensor.dtype() == at::kComplexDouble);
    TORCH_CHECK(v_tensor.dtype() == at::kComplexDouble);

    TORCH_CHECK(U_tensor.is_contiguous());
    TORCH_CHECK(v_tensor.is_contiguous());

    int vol = U_tensor.size(4) * U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3);
    

    at::Tensor result_tensor = torch::empty(v_tensor.sizes(), v_tensor.options());

    // we create a pointer to the complex tensor, then typecast it to double*
    // this allows us to access the complex numbers as doubles in riri format
    const double* U = (double*)U_tensor.const_data_ptr<c10::complex<double>>();
    const double* v = (double*)v_tensor.const_data_ptr<c10::complex<double>>();
    const int* hops = hops_tensor.const_data_ptr<int>();
    double* result = (double*)result_tensor.mutable_data_ptr<c10::complex<double>>();

    // register for the mass prefactor
    __m256d massf_reg = _mm256_set1_pd(4.0 + mass);

    // register for the -1/2 prefactor
    __m256d m0p5_reg = _mm256_set1_pd(-0.5);


#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        // mu loop partially unrolled
        // for (int mu = 0; mu < 4; mu++){
        int mu = 0;
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s+=2){

                // mass term is the first term in the result
                __m256d incr = load_split_spin(v+vixo(t,g,s));
                incr = _mm256_mul_pd(incr,massf_reg);

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                }
                // store incr in result
                store_split_spin(result+vixo(t,g,s),incr);
            }
        }
        mu = 1;
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s+=2){

                // load result of previous computations to add to
                __m256d incr = load_split_spin(result+vixo(t,g,s));

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                }
                // store incr in result
                store_split_spin(result+vixo(t,g,s),incr);
            }
        }
        for (mu = 2; mu < 4; mu++){
            for (int g = 0; g < 3; g++){
                for (int s = 0; s < 4; s+=2){

                    // load result of previous computations to add to
                    __m256d incr = load_split_spin(result+vixo(t,g,s));

                    // gamma_mu prefactor for the spins s and s+1
                    __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                    for (int gi = 0; gi < 3; gi++){
                        // v hop in negative mu * gammma
                        __m256d v_Hmum_gam = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                        // multiply the gamma prefactor for s and s+1
                        v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                        // v hop in negative mu
                        __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                        // add those together
                        v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                        // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                        v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                        // v hop in positive mu * gamma
                        __m256d v_Hmup_gam = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                        v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                        // v hop in positive mu
                        __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                        // subtract those 2
                        v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                        // multiply U at this point onto v sum
                        v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                        // add both U*v terms
                        v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                        // *(-0.5) and add to incr
                        incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                    }
                    // store incr in result
                    store_split_spin(result+vixo(t,g,s),incr);
                }
            }
        }
        //} end mu loop

    }


    return result_tensor;
}


// template for the body of the t,mu,g,s loop in dw_call_256d_om_template
// mu and s are template parameters so that the loop body can differ between iterations
// without having to check at runtime, instead generating the different code at compile time
// also, now gamma works as a template function too
// g and t are parameters, as their loop is always the same
template <int mu, int g, int s>
void dw_256d_om_mu_s_loop (const double * U, const double * v,
                           const int * hops, __m256d massf_reg,
                           double * result, int t, int vol){

    // register for the -1/2 prefactor
    __m256d m0p5_reg = _mm256_set1_pd(-0.5);
    __m256d incr;

    if constexpr (mu == 0){
        // mass term is the first term in the result
        incr = load_split_spin(v+vixo(t,g,s));
        incr = _mm256_mul_pd(incr,massf_reg);
    } else {
        // load result of previous computations to add to
        incr = load_split_spin(result+vixo(t,g,s));
    }
    
    // gamma_mu prefactor for the spins s and s+1
    //__m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

    for (int gi = 0; gi < 3; gi++){

        // v hop in negative mu * gammma
        __m256d v_Hmum_gam;
        if constexpr (mu == 0 || mu == 1){
            v_Hmum_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
            // because mu=0,1: swap the 2 numbers in the register
        } else {
            v_Hmum_gam = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
        }

        // multiply the gamma prefactor for s and s+1
        v_Hmum_gam = gamma_mul<mu,s>(v_Hmum_gam);
        //v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

        // v hop in negative mu
        __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

        // add those together
        v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

        // take Umu hop in negative mu, adjoint it, and multiply onto v sum
        v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);

        // v hop in positive mu * gamma
        __m256d v_Hmup_gam;
        if constexpr (mu == 0 || mu == 1){
            v_Hmup_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
            // because mu=0,1: swap the 2 numbers in the register
        } else {
            v_Hmup_gam = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
        }

        v_Hmup_gam = gamma_mul<mu,s>(v_Hmup_gam);
        //v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

        // v hop in positive mu
        __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

        // subtract those 2
        v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

        // multiply U at this point onto v sum
        v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


        // add both U*v terms
        v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

        // *(-0.5) and add to incr
        incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

    }
    // store incr in result
    store_split_spin(result+vixo(t,g,s),incr);
    
}


at::Tensor dw_call_256d_om_template (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                     const at::Tensor& hops_tensor, double mass){

    TORCH_CHECK(v_tensor.dim() == 6);
    TORCH_CHECK(U_tensor.size(1) == v_tensor.size(0));
    TORCH_CHECK(U_tensor.size(2) == v_tensor.size(1));
    TORCH_CHECK(U_tensor.size(3) == v_tensor.size(2));
    TORCH_CHECK(U_tensor.size(4) == v_tensor.size(3));
    TORCH_CHECK(v_tensor.size(4) == 4);
    TORCH_CHECK(v_tensor.size(5) == 3);

    TORCH_CHECK(U_tensor.dtype() == at::kComplexDouble);
    TORCH_CHECK(v_tensor.dtype() == at::kComplexDouble);

    TORCH_CHECK(U_tensor.is_contiguous());
    TORCH_CHECK(v_tensor.is_contiguous());

    int vol = U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3) * U_tensor.size(4);
    

    at::Tensor result_tensor = torch::empty(v_tensor.sizes(), v_tensor.options());

    // we create a pointer to the complex tensor, then typecast it to double*
    // this allows us to access the complex numbers as doubles in riri format
    const double* U = (double*)U_tensor.const_data_ptr<c10::complex<double>>();
    const double* v = (double*)v_tensor.const_data_ptr<c10::complex<double>>();
    const int* hops = hops_tensor.const_data_ptr<int>();
    double* result = (double*)result_tensor.mutable_data_ptr<c10::complex<double>>();

    // register for the mass prefactor
    __m256d massf_reg = _mm256_set1_pd(4.0 + mass);


#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        // loop over mu=0,1,2,3 g=0,1,2 and s=0,2 manually with template

        dw_256d_om_mu_s_loop<0,0,0>(U,v,hops,massf_reg,result,t,vol);
        dw_256d_om_mu_s_loop<0,0,2>(U,v,hops,massf_reg,result,t,vol);

        dw_256d_om_mu_s_loop<0,1,0>(U,v,hops,massf_reg,result,t,vol);
        dw_256d_om_mu_s_loop<0,1,2>(U,v,hops,massf_reg,result,t,vol);

        dw_256d_om_mu_s_loop<0,2,0>(U,v,hops,massf_reg,result,t,vol);
        dw_256d_om_mu_s_loop<0,2,2>(U,v,hops,massf_reg,result,t,vol);


        dw_256d_om_mu_s_loop<1,0,0>(U,v,hops,massf_reg,result,t,vol);
        dw_256d_om_mu_s_loop<1,0,2>(U,v,hops,massf_reg,result,t,vol);

        dw_256d_om_mu_s_loop<1,1,0>(U,v,hops,massf_reg,result,t,vol);
        dw_256d_om_mu_s_loop<1,1,2>(U,v,hops,massf_reg,result,t,vol);

        dw_256d_om_mu_s_loop<1,2,0>(U,v,hops,massf_reg,result,t,vol);
        dw_256d_om_mu_s_loop<1,2,2>(U,v,hops,massf_reg,result,t,vol);


        dw_256d_om_mu_s_loop<2,0,0>(U,v,hops,massf_reg,result,t,vol);
        dw_256d_om_mu_s_loop<2,0,2>(U,v,hops,massf_reg,result,t,vol);

        dw_256d_om_mu_s_loop<2,1,0>(U,v,hops,massf_reg,result,t,vol);
        dw_256d_om_mu_s_loop<2,1,2>(U,v,hops,massf_reg,result,t,vol);

        dw_256d_om_mu_s_loop<2,2,0>(U,v,hops,massf_reg,result,t,vol);
        dw_256d_om_mu_s_loop<2,2,2>(U,v,hops,massf_reg,result,t,vol);


        dw_256d_om_mu_s_loop<3,0,0>(U,v,hops,massf_reg,result,t,vol);
        dw_256d_om_mu_s_loop<3,0,2>(U,v,hops,massf_reg,result,t,vol);

        dw_256d_om_mu_s_loop<3,1,0>(U,v,hops,massf_reg,result,t,vol);
        dw_256d_om_mu_s_loop<3,1,2>(U,v,hops,massf_reg,result,t,vol);

        dw_256d_om_mu_s_loop<3,2,0>(U,v,hops,massf_reg,result,t,vol);
        dw_256d_om_mu_s_loop<3,2,2>(U,v,hops,massf_reg,result,t,vol);

    }

    return result_tensor;
}



at::Tensor dwc_call_lookup_256d_cpu (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                     const at::Tensor& fs_tensors,
                                     const at::Tensor& hops_tensor, double mass, double csw){

    TORCH_CHECK(v_tensor.dim() == 6);
    TORCH_CHECK(U_tensor.size(0) == v_tensor.size(0));
    TORCH_CHECK(U_tensor.size(1) == v_tensor.size(1));
    TORCH_CHECK(U_tensor.size(2) == v_tensor.size(2));
    TORCH_CHECK(U_tensor.size(3) == v_tensor.size(3));
    TORCH_CHECK(v_tensor.size(4) == 3);
    TORCH_CHECK(v_tensor.size(5) == 4);

    TORCH_CHECK(U_tensor.dtype() == at::kComplexDouble);
    TORCH_CHECK(v_tensor.dtype() == at::kComplexDouble);

    TORCH_CHECK(U_tensor.is_contiguous());
    TORCH_CHECK(v_tensor.is_contiguous());
    TORCH_CHECK(fs_tensors.is_contiguous());

    int vol = U_tensor.size(0) * U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3);
    

    at::Tensor result_tensor = torch::empty(v_tensor.sizes(), v_tensor.options());

    // we create a pointer to the complex tensor, then typecast it to double*
    // this allows us to access the complex numbers as doubles in riri format
    const double* U = (double*)U_tensor.const_data_ptr<c10::complex<double>>();
    const double* v = (double*)v_tensor.const_data_ptr<c10::complex<double>>();
    const double* F = (double*)fs_tensors.const_data_ptr<c10::complex<double>>();
    const int* hops = hops_tensor.const_data_ptr<int>();
    double* result = (double*)result_tensor.mutable_data_ptr<c10::complex<double>>();

    // register for the mass prefactor
    __m256d massf_reg = _mm256_set1_pd(4.0 + mass);

    // register for the -1/2 prefactor
    __m256d m0p5_reg = _mm256_set1_pd(-0.5);

    // register for the field strength term prefactor -1/2*csw
    __m256d csw_reg = _mm256_set1_pd(-0.5*csw);


#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        // mu loop partially unrolled
        // for (int mu = 0; mu < 4; mu++){
        int mu = 0;
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s+=2){

                // mass term is the first term in the result
                __m256d incr = _mm256_loadu_pd(v+vixd(t,g,s));
                incr = _mm256_mul_pd(incr,massf_reg);

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register
                    v_Hmum_gam = _mm256_permute4x64_pd(v_Hmum_gam,78);
                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixd(hops[hixd(t,mu,0)],mu,gi,g),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register
                    v_Hmup_gam = _mm256_permute4x64_pd(v_Hmup_gam,78);
                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixd(t,mu,g,gi),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);


                    // now the field strength tensor for the clover improvement
                    
                    // v at point t, without any swap in s (for munu=0,5)
                    __m256d v_F_05 = _mm256_loadu_pd(v+vixd(t,gi,s));
                    // v at point t, with the 2 numbers in the register swapped (for munu=1,2,3,4)
                    __m256d v_F_1234 = _mm256_permute4x64_pd(v_F_05,78);

                    // sigma prefactor for munu=0
                    __m256d sigf = _mm256_loadu_pd(sigfd+sixd(0,s));
                    // improvement for munu=0
                    __m256d v_F_0 = compl_vectorreg_pointwise_mul(sigf,v_F_05);
                    v_F_0 = compl_scalarmem_vectorreg_mul(F+fixd(t,0,g,gi),v_F_0);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_0,csw_reg,incr);

                    // sigma prefactor for munu=1
                    sigf = _mm256_loadu_pd(sigfd+sixd(1,s));
                    // improvement for munu=1
                    __m256d v_F_1 = compl_vectorreg_pointwise_mul(sigf,v_F_1234);
                    v_F_1 = compl_scalarmem_vectorreg_mul(F+fixd(t,1,g,gi),v_F_1);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_1,csw_reg,incr);

                    // sigma prefactor for munu=2
                    sigf = _mm256_loadu_pd(sigfd+sixd(2,s));
                    // improvement for munu=2
                    __m256d v_F_2 = compl_vectorreg_pointwise_mul(sigf,v_F_1234);
                    v_F_2 = compl_scalarmem_vectorreg_mul(F+fixd(t,2,g,gi),v_F_2);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_2,csw_reg,incr);

                    // sigma prefactor for munu=3
                    sigf = _mm256_loadu_pd(sigfd+sixd(3,s));
                    // improvement for munu=3
                    __m256d v_F_3 = compl_vectorreg_pointwise_mul(sigf,v_F_1234);
                    v_F_3 = compl_scalarmem_vectorreg_mul(F+fixd(t,3,g,gi),v_F_3);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_3,csw_reg,incr);

                    // sigma prefactor for munu=4
                    sigf = _mm256_loadu_pd(sigfd+sixd(4,s));
                    // improvement for munu=4
                    __m256d v_F_4 = compl_vectorreg_pointwise_mul(sigf,v_F_1234);
                    v_F_4 = compl_scalarmem_vectorreg_mul(F+fixd(t,4,g,gi),v_F_4);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_4,csw_reg,incr);

                    // sigma prefactor for munu=5
                    sigf = _mm256_loadu_pd(sigfd+sixd(5,s));
                    // improvement for munu=5
                    __m256d v_F_5 = compl_vectorreg_pointwise_mul(sigf,v_F_05);
                    v_F_5 = compl_scalarmem_vectorreg_mul(F+fixd(t,5,g,gi),v_F_5);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_5,csw_reg,incr);

                }
                // store incr in result
                _mm256_storeu_pd(result+vixd(t,g,s),incr);
            }
        }
        mu = 1;
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s+=2){

                // load result of previous computations to add to
                __m256d incr = _mm256_loadu_pd(result+vixd(t,g,s));

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register
                    v_Hmum_gam = _mm256_permute4x64_pd(v_Hmum_gam,78);
                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixd(hops[hixd(t,mu,0)],mu,gi,g),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register
                    v_Hmup_gam = _mm256_permute4x64_pd(v_Hmup_gam,78);
                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixd(t,mu,g,gi),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                }
                // store incr in result
                _mm256_storeu_pd(result+vixd(t,g,s),incr);
            }
        }
        for (mu = 2; mu < 4; mu++){
            for (int g = 0; g < 3; g++){
                for (int s = 0; s < 4; s+=2){

                    // load result of previous computations to add to
                    __m256d incr = _mm256_loadu_pd(result+vixd(t,g,s));

                    // gamma_mu prefactor for the spins s and s+1
                    __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                    for (int gi = 0; gi < 3; gi++){
                        // v hop in negative mu * gammma
                        __m256d v_Hmum_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                        // multiply the gamma prefactor for s and s+1
                        v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                        // v hop in negative mu
                        __m256d v_Hmum = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,s));

                        // add those together
                        v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                        // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                        v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixd(hops[hixd(t,mu,0)],mu,gi,g),v_Hmum);


                        // v hop in positive mu * gamma
                        __m256d v_Hmup_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                        v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                        // v hop in positive mu
                        __m256d v_Hmup = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,s));

                        // subtract those 2
                        v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                        // multiply U at this point onto v sum
                        v_Hmup = compl_scalarmem_vectorreg_mul(U+uixd(t,mu,g,gi),v_Hmup);


                        // add both U*v terms
                        v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                        // *(-0.5) and add to incr
                        incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                    }
                    // store incr in result
                    _mm256_storeu_pd(result+vixd(t,g,s),incr);
                }
            }
        }
        //} end mu loop

    }

    return result_tensor;
}


at::Tensor dwc_call_lookup_256d_old_layout (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                     const at::Tensor& fs_tensors,
                                     const at::Tensor& hops_tensor, double mass, double csw){

    TORCH_CHECK(v_tensor.dim() == 6);
    TORCH_CHECK(U_tensor.size(1) == v_tensor.size(0));
    TORCH_CHECK(U_tensor.size(2) == v_tensor.size(1));
    TORCH_CHECK(U_tensor.size(3) == v_tensor.size(2));
    TORCH_CHECK(U_tensor.size(4) == v_tensor.size(3));
    TORCH_CHECK(v_tensor.size(4) == 4);
    TORCH_CHECK(v_tensor.size(5) == 3);

    TORCH_CHECK(U_tensor.dtype() == at::kComplexDouble);
    TORCH_CHECK(v_tensor.dtype() == at::kComplexDouble);

    TORCH_CHECK(U_tensor.is_contiguous());
    TORCH_CHECK(v_tensor.is_contiguous());
    TORCH_CHECK(fs_tensors.is_contiguous());

    int vol = U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3) * U_tensor.size(4);
    

    at::Tensor result_tensor = torch::empty(v_tensor.sizes(), v_tensor.options());

    // we create a pointer to the complex tensor, then typecast it to double*
    // this allows us to access the complex numbers as doubles in riri format
    const double* U = (double*)U_tensor.const_data_ptr<c10::complex<double>>();
    const double* v = (double*)v_tensor.const_data_ptr<c10::complex<double>>();
    const double* F = (double*)fs_tensors.const_data_ptr<c10::complex<double>>();
    const int* hops = hops_tensor.const_data_ptr<int>();
    double* result = (double*)result_tensor.mutable_data_ptr<c10::complex<double>>();

    // register for the mass prefactor
    __m256d massf_reg = _mm256_set1_pd(4.0 + mass);

    // register for the -1/2 prefactor
    __m256d m0p5_reg = _mm256_set1_pd(-0.5);

    // register for the field strength term prefactor -1/2*csw
    __m256d csw_reg = _mm256_set1_pd(-0.5*csw);


#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        // mu loop partially unrolled
        // for (int mu = 0; mu < 4; mu++){
        int mu = 0;
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s+=2){

                // mass term is the first term in the result
                __m256d incr = load_split_spin(v+vixo(t,g,s));
                incr = _mm256_mul_pd(incr,massf_reg);

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                // sigma_munu prefactors for all munu for s and s+1
                __m256d sigf0 = _mm256_loadu_pd(sigfd+sixd(0,s));
                __m256d sigf1 = _mm256_loadu_pd(sigfd+sixd(1,s));
                __m256d sigf2 = _mm256_loadu_pd(sigfd+sixd(2,s));
                __m256d sigf3 = _mm256_loadu_pd(sigfd+sixd(3,s));
                __m256d sigf4 = _mm256_loadu_pd(sigfd+sixd(4,s));
                __m256d sigf5 = _mm256_loadu_pd(sigfd+sixd(5,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);


                    // now the field strength tensor for the clover improvement
                    
                    // v at point t, without any swap in s (for munu=0,5)
                    __m256d v_F_05 = load_split_spin(v+vixo(t,gi,s));
                    // v at point t, with the 2 numbers in the register swapped (for munu=1,2,3,4)
                    __m256d v_F_1234 = _mm256_permute4x64_pd(v_F_05,78);

                    // improvement for munu=0
                    __m256d v_F_0 = compl_vectorreg_pointwise_mul(sigf0,v_F_05);
                    v_F_0 = compl_scalarmem_vectorreg_mul(F+fixd(t,0,g,gi),v_F_0);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_0,csw_reg,incr);

                    // improvement for munu=1
                    __m256d v_F_1 = compl_vectorreg_pointwise_mul(sigf1,v_F_1234);
                    v_F_1 = compl_scalarmem_vectorreg_mul(F+fixd(t,1,g,gi),v_F_1);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_1,csw_reg,incr);

                    // improvement for munu=2
                    __m256d v_F_2 = compl_vectorreg_pointwise_mul(sigf2,v_F_1234);
                    v_F_2 = compl_scalarmem_vectorreg_mul(F+fixd(t,2,g,gi),v_F_2);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_2,csw_reg,incr);

                    // improvement for munu=3
                    __m256d v_F_3 = compl_vectorreg_pointwise_mul(sigf3,v_F_1234);
                    v_F_3 = compl_scalarmem_vectorreg_mul(F+fixd(t,3,g,gi),v_F_3);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_3,csw_reg,incr);

                    // improvement for munu=4
                    __m256d v_F_4 = compl_vectorreg_pointwise_mul(sigf4,v_F_1234);
                    v_F_4 = compl_scalarmem_vectorreg_mul(F+fixd(t,4,g,gi),v_F_4);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_4,csw_reg,incr);

                    // improvement for munu=5
                    __m256d v_F_5 = compl_vectorreg_pointwise_mul(sigf5,v_F_05);
                    v_F_5 = compl_scalarmem_vectorreg_mul(F+fixd(t,5,g,gi),v_F_5);
                    // multiply -1/2*csw and add to incr
                    incr = _mm256_fmadd_pd(v_F_5,csw_reg,incr);

                }
                // store incr in result
                store_split_spin(result+vixo(t,g,s),incr);
            }
        }
        mu = 1;
        for (int g = 0; g < 3; g++){
            for (int s = 0; s < 4; s+=2){

                // load result of previous computations to add to
                __m256d incr = load_split_spin(result+vixo(t,g,s));

                // gamma_mu prefactor for the spins s and s+1
                __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                for (int gi = 0; gi < 3; gi++){
                    // v hop in negative mu * gammma
                    __m256d v_Hmum_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    // multiply the gamma prefactor for s and s+1
                    v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                    // v hop in negative mu
                    __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                    // add those together
                    v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                    // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                    v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                    // v hop in positive mu * gamma
                    __m256d v_Hmup_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                    // because mu=0,1: swap the 2 numbers in the register

                    v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                    // v hop in positive mu
                    __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                    // subtract those 2
                    v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                    // multiply U at this point onto v sum
                    v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                    // add both U*v terms
                    v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                    // *(-0.5) and add to incr
                    incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                }
                // store incr in result
                store_split_spin(result+vixo(t,g,s),incr);
            }
        }
        for (mu = 2; mu < 4; mu++){
            for (int g = 0; g < 3; g++){
                for (int s = 0; s < 4; s+=2){

                    // load result of previous computations to add to
                    __m256d incr = load_split_spin(result+vixo(t,g,s));

                    // gamma_mu prefactor for the spins s and s+1
                    __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

                    for (int gi = 0; gi < 3; gi++){
                        // v hop in negative mu * gammma
                        __m256d v_Hmum_gam = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
                        // multiply the gamma prefactor for s and s+1
                        v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

                        // v hop in negative mu
                        __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

                        // add those together
                        v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

                        // take Umu hop in negative mu, adjoint it, and multiply onto v sum
                        v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);


                        // v hop in positive mu * gamma
                        __m256d v_Hmup_gam = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
                        v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

                        // v hop in positive mu
                        __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

                        // subtract those 2
                        v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

                        // multiply U at this point onto v sum
                        v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


                        // add both U*v terms
                        v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

                        // *(-0.5) and add to incr
                        incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

                    }
                    // store incr in result
                    store_split_spin(result+vixo(t,g,s),incr);
                }
            }
        }
        //} end mu loop

    }

    return result_tensor;
}


// template for the body of the t,mu,g,s loop in dwc_call_256d_om_template
// mu and s are template parameters so that the loop body can differ between iterations
// without having to check at runtime, instead generating the different code at compile time
// also, now gamma and sigma work as template functions too
// g and t are parameters, as their loop is always the same
template <int mu, int g, int s>
void dwc_256d_om_mu_s_loop (const double * U, const double * v, const double * F,
                                   const int * hops, __m256d massf_reg, __m256d csw_reg,
                                   double * result, int t, int vol){

    // register for the -1/2 prefactor
    __m256d m0p5_reg = _mm256_set1_pd(-0.5);
    __m256d incr;

    if constexpr (mu == 0){
        // mass term is the first term in the result
        incr = load_split_spin(v+vixo(t,g,s));
        incr = _mm256_mul_pd(incr,massf_reg);
    } else {
        // load result of previous computations to add to
        incr = load_split_spin(result+vixo(t,g,s));
    }
    
    // gamma_mu prefactor for the spins s and s+1
    //__m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

    for (int gi = 0; gi < 3; gi++){

        // v hop in negative mu * gammma
        __m256d v_Hmum_gam;
        if constexpr (mu == 0 || mu == 1){
            v_Hmum_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
            // because mu=0,1: swap the 2 numbers in the register
        } else {
            v_Hmum_gam = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
        }

        // multiply the gamma prefactor for s and s+1
        v_Hmum_gam = gamma_mul<mu,s>(v_Hmum_gam);
        //v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

        // v hop in negative mu
        __m256d v_Hmum = load_split_spin(v+vixo(hops[hixd(t,mu,0)],gi,s));

        // add those together
        v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

        // take Umu hop in negative mu, adjoint it, and multiply onto v sum
        v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixo(hops[hixd(t,mu,0)],mu,gi,g,vol),v_Hmum);

        // v hop in positive mu * gamma
        __m256d v_Hmup_gam;
        if constexpr (mu == 0 || mu == 1){
            v_Hmup_gam = load_split_spin_sw(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
            // because mu=0,1: swap the 2 numbers in the register
        } else {
            v_Hmup_gam = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
        }

        v_Hmup_gam = gamma_mul<mu,s>(v_Hmup_gam);
        //v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

        // v hop in positive mu
        __m256d v_Hmup = load_split_spin(v+vixo(hops[hixd(t,mu,1)],gi,s));

        // subtract those 2
        v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

        // multiply U at this point onto v sum
        v_Hmup = compl_scalarmem_vectorreg_mul(U+uixo(t,mu,g,gi,vol),v_Hmup);


        // add both U*v terms
        v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

        // *(-0.5) and add to incr
        incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);


        if constexpr (mu == 0){
            // now the field strength tensor for the clover improvement
            
            // v at point t, without any swap in s (for munu=0,5)
            __m256d v_F_05 = load_split_spin(v+vixo(t,gi,s));
            // v at point t, with the 2 numbers in the register swapped (for munu=1,2,3,4)
            __m256d v_F_1234 = _mm256_permute4x64_pd(v_F_05,78);

            // improvement for munu=0
            __m256d v_F_0 = sigma_mul<0,s>(v_F_05);
            v_F_0 = compl_scalarmem_vectorreg_mul(F+fixd(t,0,g,gi),v_F_0);
            // multiply -1/2*csw and add to incr
            incr = _mm256_fmadd_pd(v_F_0,csw_reg,incr);

            // improvement for munu=1
            __m256d v_F_1 = sigma_mul<1,s>(v_F_1234);
            v_F_1 = compl_scalarmem_vectorreg_mul(F+fixd(t,1,g,gi),v_F_1);
            // multiply -1/2*csw and add to incr
            incr = _mm256_fmadd_pd(v_F_1,csw_reg,incr);

            // improvement for munu=2
            __m256d v_F_2 = sigma_mul<2,s>(v_F_1234);
            v_F_2 = compl_scalarmem_vectorreg_mul(F+fixd(t,2,g,gi),v_F_2);
            // multiply -1/2*csw and add to incr
            incr = _mm256_fmadd_pd(v_F_2,csw_reg,incr);

            // improvement for munu=3
            __m256d v_F_3 = sigma_mul<3,s>(v_F_1234);
            v_F_3 = compl_scalarmem_vectorreg_mul(F+fixd(t,3,g,gi),v_F_3);
            // multiply -1/2*csw and add to incr
            incr = _mm256_fmadd_pd(v_F_3,csw_reg,incr);

            // improvement for munu=4
            __m256d v_F_4 = sigma_mul<4,s>(v_F_1234);
            v_F_4 = compl_scalarmem_vectorreg_mul(F+fixd(t,4,g,gi),v_F_4);
            // multiply -1/2*csw and add to incr
            incr = _mm256_fmadd_pd(v_F_4,csw_reg,incr);

            // improvement for munu=5
            __m256d v_F_5 = sigma_mul<5,s>(v_F_05);
            v_F_5 = compl_scalarmem_vectorreg_mul(F+fixd(t,5,g,gi),v_F_5);
            // multiply -1/2*csw and add to incr
            incr = _mm256_fmadd_pd(v_F_5,csw_reg,incr);
        }

    }
    // store incr in result
    store_split_spin(result+vixo(t,g,s),incr);
    
}


at::Tensor dwc_call_256d_om_template (const at::Tensor& U_tensor, const at::Tensor& v_tensor,
                                     const at::Tensor& fs_tensors,
                                     const at::Tensor& hops_tensor, double mass, double csw){

    TORCH_CHECK(v_tensor.dim() == 6);
    TORCH_CHECK(U_tensor.size(1) == v_tensor.size(0));
    TORCH_CHECK(U_tensor.size(2) == v_tensor.size(1));
    TORCH_CHECK(U_tensor.size(3) == v_tensor.size(2));
    TORCH_CHECK(U_tensor.size(4) == v_tensor.size(3));
    TORCH_CHECK(v_tensor.size(4) == 4);
    TORCH_CHECK(v_tensor.size(5) == 3);

    TORCH_CHECK(U_tensor.dtype() == at::kComplexDouble);
    TORCH_CHECK(v_tensor.dtype() == at::kComplexDouble);

    TORCH_CHECK(U_tensor.is_contiguous());
    TORCH_CHECK(v_tensor.is_contiguous());
    TORCH_CHECK(fs_tensors.is_contiguous());

    int vol = U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3) * U_tensor.size(4);
    

    at::Tensor result_tensor = torch::empty(v_tensor.sizes(), v_tensor.options());

    // we create a pointer to the complex tensor, then typecast it to double*
    // this allows us to access the complex numbers as doubles in riri format
    const double* U = (double*)U_tensor.const_data_ptr<c10::complex<double>>();
    const double* v = (double*)v_tensor.const_data_ptr<c10::complex<double>>();
    const double* F = (double*)fs_tensors.const_data_ptr<c10::complex<double>>();
    const int* hops = hops_tensor.const_data_ptr<int>();
    double* result = (double*)result_tensor.mutable_data_ptr<c10::complex<double>>();

    // register for the mass prefactor
    __m256d massf_reg = _mm256_set1_pd(4.0 + mass);


    // register for the field strength term prefactor -1/2*csw
    __m256d csw_reg = _mm256_set1_pd(-0.5*csw);


#pragma omp parallel for
    for (int t = 0; t < vol; t++){

        // loop over mu=0,1,2,3 g=0,1,2 and s=0,2 manually with template


        dwc_256d_om_mu_s_loop<0,0,0>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);
        dwc_256d_om_mu_s_loop<0,0,2>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);

        dwc_256d_om_mu_s_loop<0,1,0>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);
        dwc_256d_om_mu_s_loop<0,1,2>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);

        dwc_256d_om_mu_s_loop<0,2,0>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);
        dwc_256d_om_mu_s_loop<0,2,2>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);


        dwc_256d_om_mu_s_loop<1,0,0>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);
        dwc_256d_om_mu_s_loop<1,0,2>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);

        dwc_256d_om_mu_s_loop<1,1,0>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);
        dwc_256d_om_mu_s_loop<1,1,2>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);

        dwc_256d_om_mu_s_loop<1,2,0>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);
        dwc_256d_om_mu_s_loop<1,2,2>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);


        dwc_256d_om_mu_s_loop<2,0,0>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);
        dwc_256d_om_mu_s_loop<2,0,2>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);

        dwc_256d_om_mu_s_loop<2,1,0>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);
        dwc_256d_om_mu_s_loop<2,1,2>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);

        dwc_256d_om_mu_s_loop<2,2,0>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);
        dwc_256d_om_mu_s_loop<2,2,2>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);


        dwc_256d_om_mu_s_loop<3,0,0>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);
        dwc_256d_om_mu_s_loop<3,0,2>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);

        dwc_256d_om_mu_s_loop<3,1,0>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);
        dwc_256d_om_mu_s_loop<3,1,2>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);

        dwc_256d_om_mu_s_loop<3,2,0>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);
        dwc_256d_om_mu_s_loop<3,2,2>(U,v,F,hops,massf_reg,csw_reg,result,t,vol);
        

    }

    return result_tensor;
}

}

#endif
