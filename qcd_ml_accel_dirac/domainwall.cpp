#include <torch/extension.h>
#include <vector>

#ifndef _OPENMP
#define _OPENMP
#endif
#include <ATen/ParallelOpenMP.h>
#include <omp.h>

#include "indexfunc.hpp"
#include "gamma.hpp"

// file for the domain wall fermion dirac operator

namespace qcd_ml_accel_dirac{

at::Tensor domainwall_call_cpu (const at::Tensor& U, const at::Tensor& v, double mass, double m5){

    // check for correct size of gauge and vector field
    TORCH_CHECK(U.dim() == 8);
    TORCH_CHECK(v.dim() == 7);
    TORCH_CHECK(U.size(0) == 4);
    TORCH_CHECK(U.size(1) == v.size(0));
    TORCH_CHECK(U.size(2) == v.size(1));
    TORCH_CHECK(U.size(3) == v.size(2));
    TORCH_CHECK(U.size(4) == v.size(3));
    TORCH_CHECK(U.size(5) == v.size(4));
    TORCH_CHECK(U.size(6) == 3);
    TORCH_CHECK(U.size(7) == 3);
    TORCH_CHECK(v.size(5) == 4);
    TORCH_CHECK(v.size(6) == 3);

    TORCH_CHECK(U.dtype() == at::kComplexDouble);
    TORCH_CHECK(v.dtype() == at::kComplexDouble);

    TORCH_INTERNAL_ASSERT(U.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(v.device().type() == at::DeviceType::CPU);

    at::Tensor U_contig = U.contiguous();
    at::Tensor v_contig = v.contiguous();

    // size of space-time, spin and gauge axes
    int64_t v_size [7];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v_contig.size(sj);
    }
    int64_t u_size [8];
    for (int64_t sj = 0; sj < 7; sj++){
        u_size[sj] = U_contig.size(sj);
    }

    // strides of the memory blocks
    int64_t vstride [7];
    vstride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride[sj] = vstride[sj+1] * v_size[sj+1];
    }
    int64_t ustride [8];
    ustride[6] = 1;
    for (int64_t sj = 5; sj >= 0; sj--){
        ustride[sj] = ustride[sj+1] * u_size[sj+1];
    }

    at::Tensor result = torch::empty(v_size, v.options());

    const c10::complex<double>* U_ptr = U_contig.data_ptr<c10::complex<double>>();
    const c10::complex<double>* v_ptr = v_contig.data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.data_ptr<c10::complex<double>>();


    // iterate over the whole field
    // r is the domain wall fifth dimension index
    // x,y,z,t are the space-time indices of U, v and result
    // s is the spin index of v and result
    // g is the gauge index of result and the first gauge index of U
    // gi is the gauge index of v and the second gauge index of U, which is summed over

    // // we split the values r=0 and r=v_size[0]-1 from the loop, as they are treated differently
    // int64_t r = 0;

    // shift in r from the projection term of the domain wall operator
    int64_t prs [4] = {v_size[0]-1, v_size[0]-1, 1, 1};
    // prefactors for the projection terms P_+ and P_-
    // the first index shows if r==0, r>0 and r<v_size[0]-1, or r==v_size[0]-1
    // the second index is the spin
    // this is the mass only if s=0,1 and r=0, or s=2,3 and r=v_size[0]-1
    double prf [3][4] = {{mass,mass,-1,-1},{-1,-1,-1,-1},{-1,-1,mass,mass}};

    // parallelisation
    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t r = start; r < end; r++){
        int64_t rc = 1;
        if (r == 0){
            rc = 0;
        }
        if (r == v_size[0]-1){
            rc = 2;
        }
        for (int64_t x = 0; x < v_size[1]; x++){
            for (int64_t y = 0; y < v_size[2]; y++){
                for (int64_t z = 0; z < v_size[3]; z++){
                    for (int64_t t = 0; t < v_size[4]; t++){
                        for (int64_t s = 0; s < 4; s++){
                            for (int64_t g = 0; g < 3; g++){
                                // mass term
                                res_ptr[ptridx7(r,x,y,z,t,s,g,vstride)] = (5.0 - m5) * v_ptr[ptridx7(r,x,y,z,t,s,g,vstride)];

                                // domain wall term
                                res_ptr[ptridx7(r,x,y,z,t,s,g,vstride)]
                                += prf[rc][s]*v_ptr[ptridx7((r+prs[s]) %v_size[0],x,y,z,t,s,g,vstride)];

                                // hop terms written out for mu = 0, 1, 2, 3
                                // sum over gi corresponds to matrix product U_mu @ v
                                for (int64_t gi = 0; gi < 3; gi++){
                                    res_ptr[ptridx7(r,x,y,z,t,s,g,vstride)]
                                    +=( // mu = 0 term
                                        std::conj(U_ptr[ptridx8(0,r,(x-1+u_size[1])%u_size[1],y,z,t,gi,g,ustride)])
                                        * (
                                            -v_ptr[ptridx7(r,(x-1+v_size[0])%v_size[0],y,z,t,s,gi,vstride)]
                                            -gamf[0][s] * v_ptr[ptridx7(r,(x-1+v_size[0])%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                        )
                                        + U_ptr[ptridx8(0,r,x,y,z,t,g,gi,ustride)]
                                        * (
                                            -v_ptr[ptridx7(r,(x+1)%v_size[0],y,z,t,s,gi,vstride)]
                                            +gamf[0][s] * v_ptr[ptridx7(r,(x+1)%v_size[0],y,z,t,gamx[0][s],gi,vstride)]
                                        )

                                        // mu = 1 term
                                        +std::conj(U_ptr[ptridx8(1,r,x,(y-1+u_size[2])%u_size[2],z,t,gi,g,ustride)])
                                        * (
                                            -v_ptr[ptridx7(r,x,(y-1+v_size[1])%v_size[1],z,t,s,gi,vstride)]
                                            -gamf[1][s] * v_ptr[ptridx7(r,x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                        )
                                        + U_ptr[ptridx8(1,r,x,y,z,t,g,gi,ustride)]
                                        * (
                                            -v_ptr[ptridx7(r,x,(y+1)%v_size[1],z,t,s,gi,vstride)]
                                            +gamf[1][s] * v_ptr[ptridx7(r,x,(y+1)%v_size[1],z,t,gamx[1][s],gi,vstride)]
                                        )

                                        // mu = 2 term
                                        +std::conj(U_ptr[ptridx8(2,r,x,y,(z-1+u_size[3])%u_size[3],t,gi,g,ustride)])
                                        * (
                                            -v_ptr[ptridx7(r,x,y,(z-1+v_size[2])%v_size[2],t,s,gi,vstride)]
                                            -gamf[2][s] * v_ptr[ptridx7(r,x,y,(z-1+v_size[2])%v_size[2],t,gamx[2][s],gi,vstride)]
                                        )
                                        + U_ptr[ptridx8(2,r,x,y,z,t,g,gi,ustride)]
                                        * (
                                            -v_ptr[ptridx7(r,x,y,(z+1)%v_size[2],t,s,gi,vstride)]
                                            +gamf[2][s] * v_ptr[ptridx7(r,x,y,(z+1)%v_size[2],t,gamx[2][s],gi,vstride)]
                                        )

                                        // mu = 3 term
                                        +std::conj(U_ptr[ptridx8(3,r,x,y,z,(t-1+u_size[4])%u_size[4],gi,g,ustride)])
                                        * (
                                            -v_ptr[ptridx7(r,x,y,z,(t-1+v_size[3])%v_size[3],s,gi,vstride)]
                                            -gamf[3][s] * v_ptr[ptridx7(r,x,y,z,(t-1+v_size[3])%v_size[3],gamx[3][s],gi,vstride)]
                                        )
                                        + U_ptr[ptridx8(3,r,x,y,z,t,g,gi,ustride)]
                                        * (
                                            -v_ptr[ptridx7(r,x,y,z,(t+1)%v_size[3],s,gi,vstride)]
                                            +gamf[3][s] * v_ptr[ptridx7(r,x,y,z,(t+1)%v_size[3],gamx[3][s],gi,vstride)]
                                        )
                                    ) *0.5;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    });

    return result;
}


}