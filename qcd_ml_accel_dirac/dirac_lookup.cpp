#include <torch/extension.h>
#include <vector>

#ifndef _OPENMP
#define _OPENMP
#endif
#include <ATen/ParallelOpenMP.h>
#include <omp.h>

#include "gamma.hpp"

#include <iostream>

namespace qcd_ml_accel_dirac{

inline __attribute__((always_inline)) int64_t uix (int64_t t, int64_t mu, int64_t g, int64_t gi){
    return t*36 + mu*9 + g*3 + gi;
}

inline __attribute__((always_inline)) int64_t vix (int64_t t, int64_t g, int64_t s){
    return t*12 + g*4 + s;
}

inline __attribute__((always_inline)) int64_t hix (int64_t t, int64_t h, int64_t d){
    return t*8 + h*2 + d;
}

at::Tensor dw_call_lookup_cpu (const at::Tensor& U, const at::Tensor& v, const at::Tensor& hops, double mass){
    TORCH_CHECK(U.is_contiguous());
    TORCH_CHECK(v.is_contiguous());
    TORCH_CHECK(hops.is_contiguous());

    // in this function, we use only the flattened space-time index!
    // The indices for the input arrays are U[t,mu,g,gi] and v[t,g,s]

    int64_t vol = hops.size(0);
    // std::cout << "volume: " << vol << std::endl;

    at::Tensor result = torch::empty(v.sizes(), v.options());

    const c10::complex<double>* U_ptr = U.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* v_ptr = v.const_data_ptr<c10::complex<double>>();
    const int32_t* h_ptr = hops.const_data_ptr<int32_t>();
    c10::complex<double>* res_ptr = result.mutable_data_ptr<c10::complex<double>>();
    //return result;

//#pragma omp parallel for
    at::parallel_for(0, vol, 1, [&](int64_t start, int64_t end){
    for (int64_t t = start; t < end; t++){
        // if (t==1){
        //     std::cout << "t worked" << std::endl;
        // }
        // if (t==16*8){
        //     std::cout << "z worked" << std::endl;
        // }
        // if (t==16*8*8){
        //     std::cout << "y worked" << std::endl;
        //     //break;
        // }

        for (int64_t g = 0; g < 3; g++){
            for (int64_t s = 0; s < 4; s++){
                res_ptr[vix(t,g,s)] = (4.0 + mass) * v_ptr[vix(t,g,s)];
            }
        }
        for (int64_t mu = 0; mu < 4; mu++){
            for (int64_t g = 0; g < 3; g++){
                for (int64_t gi = 0; gi < 3; gi++){
                    for (int64_t s = 0; s < 4; s++){
                        res_ptr[vix(t,g,s)] += (
                            std::conj(U_ptr[uix(h_ptr[hix(t,mu,0)],mu,gi,g)])
                            * (
                                -v_ptr[vix(h_ptr[hix(t,mu,0)],gi,s)]
                                -gamf[mu][s] * v_ptr[vix(h_ptr[hix(t,mu,0)],gi,gamx[mu][s])]
                            )
                            + U_ptr[uix(t,mu,g,gi)]
                            * (
                                -v_ptr[vix(h_ptr[hix(t,mu,1)],gi,s)]
                                +gamf[mu][s] * v_ptr[vix(h_ptr[hix(t,mu,1)],gi,gamx[mu][s])]
                            )
                        ) * 0.5;
                    }
                }
            }
        }
    }
    });

    return result;
}


}
