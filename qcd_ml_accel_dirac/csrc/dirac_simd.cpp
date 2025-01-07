#include <torch/extension.h>
#include <omp.h>
#include <immintrin.h>

#include "complmath256d.hpp"

namespace qcd_ml_accel_dirac{

// gamma index for 2 packed spin components
// s=0 is for the spin components 0 and 1, s=2 for th spin components 2 and 3
static const int gamx_pd [3] = {2, 0, 0};

// gamfd[mu][i] is the prefactor of spin component i of gammamu @ v
// complex numbers are stored as 2 doubles
static const double gamfd [32] =
    { 0,1,  0, 1,  0,-1,  0,-1,
     -1,0,  1, 0,  1, 0, -1, 0,
      0,1,  0,-1,  0,-1,  0, 1,
      1,0,  1, 0,  1, 0,  1, 0 };

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
    
//     // now the indices are the old layout: U[mu,x,y,z,t,g,gi] and v[x,y,z,t,s,g]
//     // which means the entire code does not work anymore!
//     // And there is no way to make it work
//     TORCH_CHECK(v_tensor.dim() == 6);
//     TORCH_CHECK(U_tensor.size(1) == v_tensor.size(0));
//     TORCH_CHECK(U_tensor.size(2) == v_tensor.size(1));
//     TORCH_CHECK(U_tensor.size(3) == v_tensor.size(2));
//     TORCH_CHECK(U_tensor.size(4) == v_tensor.size(3));
//     TORCH_CHECK(v_tensor.size(5) == 3);
//     TORCH_CHECK(v_tensor.size(4) == 4);

//     TORCH_CHECK(U_tensor.dtype() == at::kComplexDouble);
//     TORCH_CHECK(v_tensor.dtype() == at::kComplexDouble);

//     TORCH_CHECK(U_tensor.is_contiguous());
//     TORCH_CHECK(v_tensor.is_contiguous());

//     int vol = U_tensor.size(1) * U_tensor.size(2) * U_tensor.size(3) * U_tensor.size(4);
    

//     at::Tensor result_tensor = torch::empty(v_tensor.sizes(), v_tensor.options());

//     // we create a pointer to the complex tensor, then typecast it to double*
//     // this allows us to access the complex numbers as doubles in riri format
//     const double* U = (double*)U_tensor.const_data_ptr<c10::complex<double>>();
//     const double* v = (double*)v_tensor.const_data_ptr<c10::complex<double>>();
//     const int* hops = hops_tensor.const_data_ptr<int>();
//     double* result = (double*)result_tensor.mutable_data_ptr<c10::complex<double>>();

//     // register for the mass prefactor
//     __m256d massf_reg = _mm256_set1_pd(4.0 + mass);

//     // register for the -1/2 prefactor
//     __m256d m0p5_reg = _mm256_set1_pd(-0.5);


// #pragma omp parallel for
//     for (int t = 0; t < vol; t++){

//         // mu loop partially unrolled
//         // for (int mu = 0; mu < 4; mu++){
//         int mu = 0;
//         for (int g = 0; g < 3; g++){
//             for (int s = 0; s < 4; s+=2){

//                 // mass term is the first term in the result
//                 __m256d incr = _mm256_loadu_pd(v+vixd(t,g,s));
//                 incr = _mm256_mul_pd(incr,massf_reg);

//                 // gamma_mu prefactor for the spins s and s+1
//                 __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

//                 for (int gi = 0; gi < 3; gi++){
//                     // v hop in negative mu * gammma
//                     __m256d v_Hmum_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
//                     // because mu=0,1: swap the 2 numbers in the register
//                     v_Hmum_gam = _mm256_permute4x64_pd(v_Hmum_gam,78);
//                     // multiply the gamma prefactor for s and s+1
//                     v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

//                     // v hop in negative mu
//                     __m256d v_Hmum = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,s));

//                     // add those together
//                     v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

//                     // take Umu hop in negative mu, adjoint it, and multiply onto v sum
//                     v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixd(hops[hixd(t,mu,0)],mu,gi,g),v_Hmum);


//                     // v hop in positive mu * gamma
//                     __m256d v_Hmup_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
//                     // because mu=0,1: swap the 2 numbers in the register
//                     v_Hmup_gam = _mm256_permute4x64_pd(v_Hmup_gam,78);
//                     v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

//                     // v hop in positive mu
//                     __m256d v_Hmup = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,s));

//                     // subtract those 2
//                     v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

//                     // multiply U at this point onto v sum
//                     v_Hmup = compl_scalarmem_vectorreg_mul(U+uixd(t,mu,g,gi),v_Hmup);


//                     // add both U*v terms
//                     v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

//                     // *(-0.5) and add to incr
//                     incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

//                 }
//                 // store incr in result
//                 _mm256_storeu_pd(result+vixd(t,g,s),incr);
//             }
//         }
//         mu = 1;
//         for (int g = 0; g < 3; g++){
//             for (int s = 0; s < 4; s+=2){

//                 // load result of previous computations to add to
//                 __m256d incr = _mm256_loadu_pd(result+vixd(t,g,s));

//                 // gamma_mu prefactor for the spins s and s+1
//                 __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

//                 for (int gi = 0; gi < 3; gi++){
//                     // v hop in negative mu * gammma
//                     __m256d v_Hmum_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
//                     // because mu=0,1: swap the 2 numbers in the register
//                     v_Hmum_gam = _mm256_permute4x64_pd(v_Hmum_gam,78);
//                     // multiply the gamma prefactor for s and s+1
//                     v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

//                     // v hop in negative mu
//                     __m256d v_Hmum = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,s));

//                     // add those together
//                     v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

//                     // take Umu hop in negative mu, adjoint it, and multiply onto v sum
//                     v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixd(hops[hixd(t,mu,0)],mu,gi,g),v_Hmum);


//                     // v hop in positive mu * gamma
//                     __m256d v_Hmup_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
//                     // because mu=0,1: swap the 2 numbers in the register
//                     v_Hmup_gam = _mm256_permute4x64_pd(v_Hmup_gam,78);
//                     v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

//                     // v hop in positive mu
//                     __m256d v_Hmup = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,s));

//                     // subtract those 2
//                     v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

//                     // multiply U at this point onto v sum
//                     v_Hmup = compl_scalarmem_vectorreg_mul(U+uixd(t,mu,g,gi),v_Hmup);


//                     // add both U*v terms
//                     v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

//                     // *(-0.5) and add to incr
//                     incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

//                 }
//                 // store incr in result
//                 _mm256_storeu_pd(result+vixd(t,g,s),incr);
//             }
//         }
//         for (mu = 2; mu < 4; mu++){
//             for (int g = 0; g < 3; g++){
//                 for (int s = 0; s < 4; s+=2){

//                     // load result of previous computations to add to
//                     __m256d incr = _mm256_loadu_pd(result+vixd(t,g,s));

//                     // gamma_mu prefactor for the spins s and s+1
//                     __m256d gamf2s = _mm256_loadu_pd(gamfd+gixd(mu,s));

//                     for (int gi = 0; gi < 3; gi++){
//                         // v hop in negative mu * gammma
//                         __m256d v_Hmum_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,gamx_pd[s]));
//                         // multiply the gamma prefactor for s and s+1
//                         v_Hmum_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmum_gam);

//                         // v hop in negative mu
//                         __m256d v_Hmum = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,0)],gi,s));

//                         // add those together
//                         v_Hmum = _mm256_add_pd(v_Hmum, v_Hmum_gam);

//                         // take Umu hop in negative mu, adjoint it, and multiply onto v sum
//                         v_Hmum = compl_scalarmem_conj_vectorreg_mul(U+uixd(hops[hixd(t,mu,0)],mu,gi,g),v_Hmum);


//                         // v hop in positive mu * gamma
//                         __m256d v_Hmup_gam = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,gamx_pd[s]));
//                         v_Hmup_gam = compl_vectorreg_pointwise_mul(gamf2s,v_Hmup_gam);

//                         // v hop in positive mu
//                         __m256d v_Hmup = _mm256_loadu_pd(v+vixd(hops[hixd(t,mu,1)],gi,s));

//                         // subtract those 2
//                         v_Hmup = _mm256_sub_pd(v_Hmup, v_Hmup_gam);

//                         // multiply U at this point onto v sum
//                         v_Hmup = compl_scalarmem_vectorreg_mul(U+uixd(t,mu,g,gi),v_Hmup);


//                         // add both U*v terms
//                         v_Hmum = _mm256_add_pd(v_Hmum,v_Hmup);

//                         // *(-0.5) and add to incr
//                         incr = _mm256_fmadd_pd(v_Hmum,m0p5_reg,incr);

//                     }
//                     // store incr in result
//                     _mm256_storeu_pd(result+vixd(t,g,s),incr);
//                 }
//             }
//         }
//         //} end mu loop

//     }


//     return result_tensor;
// }

}
