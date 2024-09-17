#include <torch/extension.h>

#include <vector>

#include "wilson.hpp"


// Multiply a gauge field U2 with a gauge or vector field Uv.
// Their positions on the grid in each dimension are shifted by
// the corresponding component of u2shifts and uvshifts respectively.

at::Tensor shift_gaugemul_p_cpu (const at::Tensor& U2, const at::Tensor& Uv,
                                std::vector<int64_t> u2shifts, std::vector<int64_t> uvshifts){

    // check that both tensors have the correct shape for a gauge or vector field tensor
    TORCH_CHECK(Uv.dim() == 6);
    TORCH_CHECK(U2.dim() == 6);
    TORCH_CHECK(Uv.size(0) == U2.size(0));
    TORCH_CHECK(Uv.size(1) == U2.size(1));
    TORCH_CHECK(Uv.size(2) == U2.size(2));
    TORCH_CHECK(Uv.size(3) == U2.size(3));
    TORCH_CHECK(U2.size(4) == 3);
    TORCH_CHECK(U2.size(5) == 3);
    TORCH_CHECK(Uv.size(5) == 3);

    // check that the shift vector length is the number of space-time dimensions
    TORCH_CHECK(u2shifts.size() == 4);
    TORCH_CHECK(uvshifts.size() == 4);

    // check that both tensors are complex double
    TORCH_CHECK(U2.dtype() == at::kComplexDouble);
    TORCH_CHECK(Uv.dtype() == at::kComplexDouble);

    TORCH_INTERNAL_ASSERT(U2.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(Uv.device().type() == at::DeviceType::CPU);

    at::Tensor Uv_contig = Uv.contiguous();
    at::Tensor U2_contig = U2.contiguous();

    // sizes of the different axes
    int64_t uvsize [6];
    int64_t u2size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        uvsize[sj] = Uv_contig.size(sj);
        u2size[sj] = U2_contig.size(sj);
    }

    // strides of the memory blocks
    int64_t uvstride [6];
    int64_t u2stride [6];
    uvstride[5] = 1;
    u2stride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        uvstride [sj] = uvstride[sj+1] * uvsize[sj+1];
        u2stride [sj] = u2stride[sj+1] * u2size[sj+1];
    }

    // if shifts are negative, they have to be changed to the positive equivalent
    // this is because the modulo has undefined behavior for negative shifts
    for (int64_t sd = 0; sd < 4; sd++){
        if (uvshifts[sd] < 0){
            uvshifts[sd] += uvsize[sd];
        }
        if (u2shifts[sd] < 0){
            u2shifts[sd] += u2size[sd];
        }
    }

    at::Tensor result = torch::zeros(uvsize, Uv.options());

    const c10::complex<double>* Uv_ptr = Uv_contig.data_ptr<c10::complex<double>>();
    const c10::complex<double>* U2_ptr = U2_contig.data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.data_ptr<c10::complex<double>>();


    // different multiplication if Uv is a gauge or vector field
    // the gauge field is identified by its axis 4 having length 3
    // if Uv is a gauge field, this is a SU3 matrix product, and result is a gauge field
    // if Uv is a vector, this is a gauge transform, which acts on the last dimension of Uv, and result is a vector
    if (uvsize[4] == 3){
        // simple matrix multiplication, where the elements are accessed with shifts
        for (int64_t x = 0; x < uvsize[0]; x++){
            for (int64_t y = 0; y < uvsize[1]; y++){
                for (int64_t z = 0; z < uvsize[2]; z++){
                    for (int64_t t = 0; t < uvsize[3]; t++){
                        for (int64_t g = 0; g < 3; g++){
                            for (int64_t h = 0; h < 3; h++){
                                for (int64_t i = 0; i < 3; i++){
                                    res_ptr[  x *uvstride[0]
                                            + y *uvstride[1]
                                            + z *uvstride[2]
                                            + t *uvstride[3]
                                            + g *uvstride[4]
                                            + h *uvstride[5]]+= U2_ptr[  ((x + u2shifts[0])%u2size[0]) *u2stride[0]
                                                                       + ((y + u2shifts[1])%u2size[1]) *u2stride[1]
                                                                       + ((z + u2shifts[2])%u2size[2]) *u2stride[2]
                                                                       + ((t + u2shifts[3])%u2size[3]) *u2stride[3]
                                                                       +   g                           *u2stride[4]
                                                                       +   i                           *u2stride[5] ]
                                                                *Uv_ptr[  ((x + uvshifts[0])%uvsize[0]) *uvstride[0]
                                                                        + ((y + uvshifts[1])%uvsize[1]) *uvstride[1]
                                                                        + ((z + uvshifts[2])%uvsize[2]) *uvstride[2]
                                                                        + ((t + uvshifts[3])%uvsize[3]) *uvstride[3]
                                                                        +   i                           *uvstride[4]
                                                                        +   h                           *uvstride[5] ] ;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        // matrix multiplication w.r.t. the last indices of both tensors
        for (int64_t x = 0; x < uvsize[0]; x++){
            for (int64_t y = 0; y < uvsize[1]; y++){
                for (int64_t z = 0; z < uvsize[2]; z++){
                    for (int64_t t = 0; t < uvsize[3]; t++){
                        for (int64_t g = 0; g < 3; g++){
                            for (int64_t h = 0; h < uvsize[4]; h++){
                                for (int64_t i = 0; i < 3; i++){
                                    res_ptr[  x *uvstride[0]
                                            + y *uvstride[1]
                                            + z *uvstride[2]
                                            + t *uvstride[3]
                                            + h *uvstride[4]
                                            + g *uvstride[5]]+= U2_ptr[  ((x + u2shifts[0])%u2size[0]) *u2stride[0]
                                                                       + ((y + u2shifts[1])%u2size[1]) *u2stride[1]
                                                                       + ((z + u2shifts[2])%u2size[2]) *u2stride[2]
                                                                       + ((t + u2shifts[3])%u2size[3]) *u2stride[3]
                                                                       +   g                           *u2stride[4]
                                                                       +   i                           *u2stride[5] ]
                                                                *Uv_ptr[  ((x + uvshifts[0])%uvsize[0]) *uvstride[0]
                                                                        + ((y + uvshifts[1])%uvsize[1]) *uvstride[1]
                                                                        + ((z + uvshifts[2])%uvsize[2]) *uvstride[2]
                                                                        + ((t + uvshifts[3])%uvsize[3]) *uvstride[3]
                                                                        +   h                           *uvstride[4]
                                                                        +   i                           *uvstride[5] ] ;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

// other functions are in wilson.hpp


// Registers _C as a Python extension module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(cppoperations, m) {
    m.def("shift_gaugemul_p(Tensor U2, Tensor Uv, int[] u2shifts, int[] uvshifts) -> Tensor");
    m.def("dirac_wilson_call(Tensor U, Tensor v, float mass) -> Tensor");
    m.def("dirac_wilson_clover_call(Tensor U, Tensor v, Tensor[] F, float mass, float csw) -> Tensor");
}

// Registers backend implementations
TORCH_LIBRARY_IMPL(cppoperations, CPU, m) {
    m.impl("shift_gaugemul_p", &shift_gaugemul_p_cpu);
    m.impl("dirac_wilson_call", &dw_call_p);
    m.impl("dirac_wilson_clover_call", &dwc_call_p);
}

