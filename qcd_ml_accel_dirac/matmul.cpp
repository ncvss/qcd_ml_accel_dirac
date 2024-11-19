#include <torch/extension.h>

#ifndef _OPENMP
#define _OPENMP
#endif
#include <ATen/ParallelOpenMP.h>
#include <omp.h>

#include "indexfunc.hpp"
#include "gamma.hpp"

namespace qcd_ml_accel_dirac{

// note: the following functions take gauge field tensors of different sizes,
// not necessarily of the usual size 4x[lattice]x3x3,
// as the partial computations need a varying number of gauge links

at::Tensor gauge_transform (const at::Tensor& U, const at::Tensor& v){

    at::Tensor v_contig = v.contiguous();
    at::Tensor U_contig = U.contiguous();

    int64_t v_size [6];
    int64_t u_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v_contig.size(sj);
        u_size[sj] = U_contig.size(sj);
    }

    int64_t vstride [6];
    int64_t ustride [6];
    vstride[5] = 1;
    ustride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride [sj] = vstride[sj+1] * v_size[sj+1];
        ustride [sj] = ustride[sj+1] * u_size[sj+1];
    }

    at::Tensor result = torch::zeros(v_size, v.options());

    const c10::complex<double>* v_ptr = v_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* U_ptr = U_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.mutable_data_ptr<c10::complex<double>>();

    for (int64_t x = 0; x < v_size[0]; x++){
        for (int64_t y = 0; y < v_size[1]; y++){
            for (int64_t z = 0; z < v_size[2]; z++){
                for (int64_t t = 0; t < v_size[3]; t++){
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t i = 0; i < 3; i++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)]
                                +=  U_ptr[ptridx6(x,y,z,t,g,i,ustride)] * v_ptr[ptridx6(x,y,z,t,s,i,vstride)];
                            }
                        }
                    }
                }
            }
        }
    }

    return result;

}

at::Tensor gauge_transform_par (const at::Tensor& U, const at::Tensor& v){

    at::Tensor v_contig = v.contiguous();
    at::Tensor U_contig = U.contiguous();

    int64_t v_size [6];
    int64_t u_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v_contig.size(sj);
        u_size[sj] = U_contig.size(sj);
    }

    int64_t vstride [6];
    int64_t ustride [6];
    vstride[5] = 1;
    ustride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride [sj] = vstride[sj+1] * v_size[sj+1];
        ustride [sj] = ustride[sj+1] * u_size[sj+1];
    }

    at::Tensor result = torch::zeros(v_size, v.options());

    const c10::complex<double>* v_ptr = v_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* U_ptr = U_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.mutable_data_ptr<c10::complex<double>>();

    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
        for (int64_t y = 0; y < v_size[1]; y++){
            for (int64_t z = 0; z < v_size[2]; z++){
                for (int64_t t = 0; t < v_size[3]; t++){
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t i = 0; i < 3; i++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)]
                                +=  U_ptr[ptridx6(x,y,z,t,g,i,ustride)] * v_ptr[ptridx6(x,y,z,t,s,i,vstride)];
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

at::Tensor gauge_transform_gamma (const at::Tensor& U, const at::Tensor& v){

    at::Tensor v_contig = v.contiguous();
    at::Tensor U_contig = U.contiguous();

    int64_t v_size [6];
    int64_t u_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v_contig.size(sj);
        u_size[sj] = U_contig.size(sj);
    }

    int64_t vstride [6];
    int64_t ustride [6];
    vstride[5] = 1;
    ustride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride [sj] = vstride[sj+1] * v_size[sj+1];
        ustride [sj] = ustride[sj+1] * u_size[sj+1];
    }

    at::Tensor result = torch::zeros(v_size, v.options());

    const c10::complex<double>* v_ptr = v_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* U_ptr = U_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.mutable_data_ptr<c10::complex<double>>();

    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
        for (int64_t y = 0; y < v_size[1]; y++){
            for (int64_t z = 0; z < v_size[2]; z++){
                for (int64_t t = 0; t < v_size[3]; t++){
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t i = 0; i < 3; i++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    U_ptr[ptridx6(x,y,z,t,g,i,ustride)] * (
                                        v_ptr[ptridx6(x,y,z,t,s,i,vstride)]
                                        + gamf[1][s] * v_ptr[ptridx6(x,y,z,t,gamx[1][s],i,vstride)]
                                    )
                                );
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

at::Tensor gauge_transform_gamma_shift (const at::Tensor& U, const at::Tensor& v){

    at::Tensor v_contig = v.contiguous();
    at::Tensor U_contig = U.contiguous();

    int64_t v_size [6];
    int64_t u_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v_contig.size(sj);
        u_size[sj] = U_contig.size(sj);
    }

    int64_t vstride [6];
    int64_t ustride [6];
    vstride[5] = 1;
    ustride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride [sj] = vstride[sj+1] * v_size[sj+1];
        ustride [sj] = ustride[sj+1] * u_size[sj+1];
    }

    at::Tensor result = torch::zeros(v_size, v.options());

    const c10::complex<double>* v_ptr = v_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* U_ptr = U_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.mutable_data_ptr<c10::complex<double>>();

    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
        for (int64_t y = 0; y < v_size[1]; y++){
            for (int64_t z = 0; z < v_size[2]; z++){
                for (int64_t t = 0; t < v_size[3]; t++){
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t i = 0; i < 3; i++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    U_ptr[ptridx6(x,y,z,t,g,i,ustride)] * (
                                        v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,s,i,vstride)]
                                        + gamf[1][s] * v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],i,vstride)]
                                    )
                                    // + U_ptr[ptridx6(x,y,z,t,g,i,ustride)] * (
                                    //     v_ptr[ptridx6(x,y,z,t,s,i,vstride)]
                                    //     + gamf[1][s] * v_ptr[ptridx6(x,y,z,t,gamx[1][s],i,vstride)]
                                    // )
                                );
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

at::Tensor gauge_transform_gamma_2shift (const at::Tensor& U, const at::Tensor& v){

    at::Tensor v_contig = v.contiguous();
    at::Tensor U_contig = U.contiguous();

    int64_t v_size [6];
    int64_t u_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v_contig.size(sj);
        u_size[sj] = U_contig.size(sj);
    }

    int64_t vstride [6];
    int64_t ustride [6];
    vstride[5] = 1;
    ustride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride [sj] = vstride[sj+1] * v_size[sj+1];
        ustride [sj] = ustride[sj+1] * u_size[sj+1];
    }

    at::Tensor result = torch::zeros(v_size, v.options());

    const c10::complex<double>* v_ptr = v_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* U_ptr = U_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.mutable_data_ptr<c10::complex<double>>();

    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
        for (int64_t y = 0; y < v_size[1]; y++){
            for (int64_t z = 0; z < v_size[2]; z++){
                for (int64_t t = 0; t < v_size[3]; t++){
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t i = 0; i < 3; i++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    U_ptr[ptridx6(x,y,z,t,g,i,ustride)] * (
                                        v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,s,i,vstride)]
                                        + gamf[1][s] * v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],i,vstride)]
                                    )
                                    + U_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,g,i,ustride)] * (
                                        v_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,i,vstride)]
                                        + gamf[1][s] * v_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],i,vstride)]
                                    )
                                );
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

at::Tensor gauge_transform_gamma_2shift_split (const at::Tensor& U, const at::Tensor& v){

    at::Tensor v_contig = v.contiguous();
    at::Tensor U_contig = U.contiguous();

    int64_t v_size [6];
    int64_t u_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v_contig.size(sj);
        u_size[sj] = U_contig.size(sj);
    }

    int64_t vstride [6];
    int64_t ustride [6];
    vstride[5] = 1;
    ustride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride [sj] = vstride[sj+1] * v_size[sj+1];
        ustride [sj] = ustride[sj+1] * u_size[sj+1];
    }

    at::Tensor result = torch::zeros(v_size, v.options());

    const c10::complex<double>* v_ptr = v_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* U_ptr = U_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.mutable_data_ptr<c10::complex<double>>();

    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
        for (int64_t y = 0; y < v_size[1]; y++){
            for (int64_t z = 0; z < v_size[2]; z++){
                for (int64_t t = 0; t < v_size[3]; t++){
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t i = 0; i < 3; i++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    U_ptr[ptridx6(x,y,z,t,g,i,ustride)] * (
                                        v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,s,i,vstride)]
                                        + gamf[1][s] * v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],i,vstride)]
                                    )
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    });

    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
        for (int64_t y = 0; y < v_size[1]; y++){
            for (int64_t z = 0; z < v_size[2]; z++){
                for (int64_t t = 0; t < v_size[3]; t++){
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t i = 0; i < 3; i++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    U_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,g,i,ustride)] * (
                                        v_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,i,vstride)]
                                        + gamf[1][s] * v_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],i,vstride)]
                                    )
                                );
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


at::Tensor gauge_transform_gamma_2shift_ysw (const at::Tensor& U, const at::Tensor& v){

    at::Tensor v_contig = v.contiguous();
    at::Tensor U_contig = U.contiguous();

    int64_t v_size [6];
    int64_t u_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v_contig.size(sj);
        u_size[sj] = U_contig.size(sj);
    }

    int64_t vstride [6];
    int64_t ustride [6];
    vstride[5] = 1;
    ustride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride [sj] = vstride[sj+1] * v_size[sj+1];
        ustride [sj] = ustride[sj+1] * u_size[sj+1];
    }

    at::Tensor result = torch::zeros(v_size, v.options());

    const c10::complex<double>* v_ptr = v_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* U_ptr = U_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.mutable_data_ptr<c10::complex<double>>();

    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
        for (int64_t z = 0; z < v_size[2]; z++){
            for (int64_t t = 0; t < v_size[3]; t++){
                for (int64_t y = 0; y < v_size[1]; y++){
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t i = 0; i < 3; i++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    U_ptr[ptridx6(x,y,z,t,g,i,ustride)] * (
                                        v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,s,i,vstride)]
                                        + gamf[1][s] * v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],i,vstride)]
                                    )
                                    + U_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,g,i,ustride)] * (
                                        v_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,i,vstride)]
                                        + gamf[1][s] * v_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],i,vstride)]
                                    )
                                );
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


at::Tensor gauge_transform_gamma_2tshift (const at::Tensor& U, const at::Tensor& v){

    at::Tensor v_contig = v.contiguous();
    at::Tensor U_contig = U.contiguous();

    int64_t v_size [6];
    int64_t u_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v_contig.size(sj);
        u_size[sj] = U_contig.size(sj);
    }

    int64_t vstride [6];
    int64_t ustride [6];
    vstride[5] = 1;
    ustride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride [sj] = vstride[sj+1] * v_size[sj+1];
        ustride [sj] = ustride[sj+1] * u_size[sj+1];
    }

    at::Tensor result = torch::zeros(v_size, v.options());

    const c10::complex<double>* v_ptr = v_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* U_ptr = U_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.mutable_data_ptr<c10::complex<double>>();

    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
        for (int64_t y = 0; y < v_size[1]; y++){
            for (int64_t z = 0; z < v_size[2]; z++){
                for (int64_t t = 0; t < v_size[3]; t++){
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t i = 0; i < 3; i++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    U_ptr[ptridx6(x,y,z,t,g,i,ustride)] * (
                                        v_ptr[ptridx6(x,y,z,(t+1)%v_size[3],s,i,vstride)]
                                        + gamf[3][s] * v_ptr[ptridx6(x,y,z,(t+1)%v_size[3],gamx[1][s],i,vstride)]
                                    )
                                    + U_ptr[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],g,i,ustride)] * (
                                        v_ptr[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],s,i,vstride)]
                                        + gamf[3][s] * v_ptr[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],gamx[3][s],i,vstride)]
                                    )
                                );
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


at::Tensor gauge_transform_gamma_2ytshift (const at::Tensor& U, const at::Tensor& v){

    at::Tensor v_contig = v.contiguous();
    at::Tensor U_contig = U.contiguous();

    int64_t v_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v_contig.size(sj);
    }
    int64_t u_size [7];
    for (int64_t sj = 0; sj < 7; sj++){
        u_size[sj] = U_contig.size(sj);
    }
    int64_t vstride [6];
    vstride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride[sj] = vstride[sj+1] * v_size[sj+1];
    }
    int64_t ustride [7];
    ustride[6] = 1;
    for (int64_t sj = 5; sj >= 0; sj--){
        ustride[sj] = ustride[sj+1] * u_size[sj+1];
    }

    at::Tensor result = torch::zeros(v_size, v.options());

    const c10::complex<double>* v_ptr = v_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* U_ptr = U_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.mutable_data_ptr<c10::complex<double>>();

    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
        for (int64_t y = 0; y < v_size[1]; y++){
            for (int64_t z = 0; z < v_size[2]; z++){
                for (int64_t t = 0; t < v_size[3]; t++){
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t i = 0; i < 3; i++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    U_ptr[ptridx7(0,x,y,z,t,g,i,ustride)] * (
                                        v_ptr[ptridx6(x,y,z,(t+1)%v_size[3],s,i,vstride)]
                                        + gamf[3][s] * v_ptr[ptridx6(x,y,z,(t+1)%v_size[3],gamx[1][s],i,vstride)]
                                    )
                                    + U_ptr[ptridx7(0,x,y,z,(t-1+v_size[3])%v_size[3],g,i,ustride)] * (
                                        v_ptr[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],s,i,vstride)]
                                        + gamf[3][s] * v_ptr[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],gamx[3][s],i,vstride)]
                                    )
                                );
                            }
                        }
                    }
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t i = 0; i < 3; i++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    U_ptr[ptridx7(1,x,y,z,t,g,i,ustride)] * (
                                        v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,s,i,vstride)]
                                        + gamf[1][s] * v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,gamx[1][s],i,vstride)]
                                    )
                                    + U_ptr[ptridx7(1,x,(y-1+v_size[1])%v_size[1],z,t,g,i,ustride)] * (
                                        v_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,s,i,vstride)]
                                        + gamf[1][s] * v_ptr[ptridx6(x,(y-1+v_size[1])%v_size[1],z,t,gamx[1][s],i,vstride)]
                                    )
                                );
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

at::Tensor gauge_transform_simple_ytshift (const at::Tensor& U, const at::Tensor& v){

    at::Tensor v_contig = v.contiguous();
    at::Tensor U_contig = U.contiguous();

    int64_t v_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v_contig.size(sj);
    }
    int64_t u_size [7];
    for (int64_t sj = 0; sj < 7; sj++){
        u_size[sj] = U_contig.size(sj);
    }
    int64_t vstride [6];
    vstride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride[sj] = vstride[sj+1] * v_size[sj+1];
    }
    int64_t ustride [7];
    ustride[6] = 1;
    for (int64_t sj = 5; sj >= 0; sj--){
        ustride[sj] = ustride[sj+1] * u_size[sj+1];
    }

    at::Tensor result = torch::zeros(v_size, v.options());

    const c10::complex<double>* v_ptr = v_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* U_ptr = U_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.mutable_data_ptr<c10::complex<double>>();

    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
        for (int64_t y = 0; y < v_size[1]; y++){
            for (int64_t z = 0; z < v_size[2]; z++){
                for (int64_t t = 0; t < v_size[3]; t++){
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t i = 0; i < 3; i++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    U_ptr[ptridx7(1,x,y,z,t,g,i,ustride)] * (
                                        v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,s,i,vstride)]
                                    )
                                    + U_ptr[ptridx7(0,x,y,z,(t-1+v_size[3])%v_size[3],g,i,ustride)] * (
                                        v_ptr[ptridx6(x,y,z,(t-1+v_size[3])%v_size[3],s,i,vstride)]
                                    )
                                );
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

at::Tensor gauge_transform_gamma_2point (const at::Tensor& U, const at::Tensor& v){

    at::Tensor v_contig = v.contiguous();
    at::Tensor U_contig = U.contiguous();

    int64_t v_size [6];
    int64_t u_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        v_size[sj] = v_contig.size(sj);
        u_size[sj] = U_contig.size(sj);
    }

    int64_t vstride [6];
    int64_t ustride [6];
    vstride[5] = 1;
    ustride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride [sj] = vstride[sj+1] * v_size[sj+1];
        ustride [sj] = ustride[sj+1] * u_size[sj+1];
    }

    at::Tensor result = torch::zeros(v_size, v.options());

    const c10::complex<double>* v_ptr = v_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* U_ptr = U_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.mutable_data_ptr<c10::complex<double>>();

    at::parallel_for(0, v_size[0], 1, [&](int64_t start, int64_t end){
    for (int64_t x = start; x < end; x++){
        for (int64_t y = 0; y < v_size[1]; y++){
            for (int64_t z = 0; z < v_size[2]; z++){
                for (int64_t t = 0; t < v_size[3]; t++){
                    for (int64_t g = 0; g < 3; g++){
                        for (int64_t i = 0; i < 3; i++){
                            for (int64_t s = 0; s < 4; s++){
                                res_ptr[ptridx6(x,y,z,t,s,g,vstride)] += (
                                    U_ptr[ptridx6(x,y,z,t,g,i,ustride)] * (
                                        v_ptr[ptridx6(x,(y+1)%v_size[1],z,t,s,i,vstride)]
                                        + gamf[1][s] * v_ptr[ptridx6(x,y,z,t,gamx[1][s],i,vstride)]
                                    )
                                );
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
