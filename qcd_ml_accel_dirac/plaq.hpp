#include <torch/extension.h>
#include <vector>

// file for the plaquette action

namespace qcd_ml_accel_dirac{

c10::complex<double> plaq_action_cpu (const at::Tensor& U, double g){
    
    // check that U has the correct shape for a gauge tensor
    TORCH_CHECK(U.dim() == 6);
    TORCH_CHECK(U.size(4) == 3);
    TORCH_CHECK(U.size(5) == 3);

    // check that U is complex double
    TORCH_CHECK(U.dtype() == at::kComplexDouble);

    TORCH_INTERNAL_ASSERT(U.device().type() == at::DeviceType::CPU);

    at::Tensor U_contig = U.contiguous();

    // sizes of the different axes
    int64_t u_size [6];
    for (int64_t sj = 0; sj < 6; sj++){
        u_size[sj] = U_contig.size(sj);
    }

    // strides of the memory blocks
    int64_t u_stride [6];
    u_stride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        u_stride [sj] = u_stride[sj+1] * u_size[sj+1];
    }

    const c10::complex<double>* U_ptr = U_contig.data_ptr<c10::complex<double>>();

    // easier to first do this in python

}


}

