#include <torch/extension.h>
#include <vector>

#ifndef _OPENMP
#define _OPENMP
#endif
#include <ATen/ParallelOpenMP.h>
#include <omp.h>

// file for the plaquette action

namespace qcd_ml_accel_dirac{

double plaq_action_cpu (const at::Tensor& U, double g){
    
    // check that U has the correct shape for a gauge tensor
    TORCH_CHECK(U.dim() == 7);
    TORCH_CHECK(U.size(5) == 3);
    TORCH_CHECK(U.size(6) == 3);

    // check that U is complex double
    TORCH_CHECK(U.dtype() == at::kComplexDouble);

    TORCH_INTERNAL_ASSERT(U.device().type() == at::DeviceType::CPU);

    at::Tensor U_contig = U.contiguous();

    // sizes of the different axes
    int64_t u_size [7];
    for (int64_t sj = 0; sj < 7; sj++){
        u_size[sj] = U_contig.size(sj);
    }
    // strides of the memory blocks
    int64_t u_stride [7];
    u_stride[6] = 1;
    for (int64_t sj = 5; sj >= 0; sj--){
        u_stride [sj] = u_stride[sj+1] * u_size[sj+1];
    }

    // sizes and strides for a gauge tensor for a specific direction
    int64_t um_size [6];
    for (int64_t si = 0; si < 6; si++){
        um_size[si] = u_size[si+1];
    }
    int64_t um_stride [6];
    um_stride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        um_stride [sj] = um_stride[sj+1] * um_size[sj+1];
    }

    const c10::complex<double>* U_ptr = U_contig.data_ptr<c10::complex<double>>();

    // refer to the python version for a more intuitive computation

    // lookup tables for shifts in direction mu (basically an identity matrix)
    std::vector<std::vector<int64_t>> shvec = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};

    // lookup tables for the possible tuples of mu and nu
    int64_t nus [6] = {1,2,2,3,3,3};
    int64_t mus [6] = {0,0,1,0,1,2};

    double result = 0.0;
    
    // parallel loop over the possible tuples (mu,nu)
    // not good to parallelize, as a matrix multiplication happens inside
    // as per the docs, parallelization should not contain tensor operations
    //at::parallel_reduce(0, 6, 1, 0, [&](int64_t start, int64_t end){
    for (int64_t im = 0; im < 6; im++){
        //for (int64_t nu = 0; nu < 4; nu++){
        //for (int64_t mu = 0; mu < nu; mu++){
            // only compute the matrix product of the first 3 matrices
            // the fourth one is unneeded, as we only take the trace
            // also, we reverse the order to only call the adjoint once
            at::Tensor Umn_prel = shift_gaugemul_p_cpu(U[nus[im]],
                                                       at::adjoint(shift_gaugemul_p_cpu(U[nus[im]], U[mus[im]], {0,0,0,0}, shvec[nus[im]])),
                                                       shvec[mus[im]], {0,0,0,0});
            at::Tensor Umnp_contig = Umn_prel.contiguous();
            const c10::complex<double>* Umnp_ptr = Umnp_contig.data_ptr<c10::complex<double>>();

            at::parallel_reduce(0, u_size[1], 1, 0, [&](int64_t start, int64_t end){
            for (int64_t x = start; x < end; x++){
                for (int64_t y = 0; y < u_size[2]; y++){
                    for (int64_t z = 0; z < u_size[3]; z++){
                        for (int64_t t = 0; t < u_size[4]; t++){
                            // sum over 3 contributions to the trace
                            for (int64_t g = 0; g < 3; g++){
                                // the contribution is the gth row of U_mu times the gth column of Umn_prel
                                result += std::real( 1.0
                                                    -U_ptr[ptridx7(mus[im],x,y,z,t,g,0,u_stride)] * Umnp_ptr[ptridx6(x,y,z,t,0,g,um_stride)]
                                                    -U_ptr[ptridx7(mus[im],x,y,z,t,g,1,u_stride)] * Umnp_ptr[ptridx6(x,y,z,t,1,g,um_stride)]
                                                    -U_ptr[ptridx7(mus[im],x,y,z,t,g,2,u_stride)] * Umnp_ptr[ptridx6(x,y,z,t,2,g,um_stride)]
                                                    );
                            }
                        }
                    }
                }
            }
            }, [&](double r1, double r2){return r1+r2;});
        //}
        //}
    }
    //}, [](double r1, double r2){return r1+r2;});
    

    return result *2/(g*g);
}


}

