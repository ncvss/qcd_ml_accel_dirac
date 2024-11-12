#include <torch/extension.h>

#ifndef _OPENMP
#define _OPENMP
#endif
#include <ATen/ParallelOpenMP.h>
#include <omp.h>

#include <chrono>

namespace qcd_ml_accel_dirac {

at::Tensor muladd_bench_nopar(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c){
    // TORCH_CHECK(a.sizes() == b.sizes());
    // TORCH_CHECK(a.sizes() == c.sizes());
    // TORCH_CHECK(a.dtype() == at::kComplexDouble);
    // TORCH_CHECK(b.dtype() == at::kComplexDouble);
    // TORCH_CHECK(c.dtype() == at::kComplexDouble);
    // TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    // TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor c_contig = c.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());

    const c10::complex<double>* a_ptr = a_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* b_ptr = b_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* c_ptr = c_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* result_ptr = result.mutable_data_ptr<c10::complex<double>>();

    int64_t len = result.numel();

    for (int64_t i = 0; i < len; i++) {
        result_ptr[i] = a_ptr[i] * b_ptr[i] + c_ptr[i];
    }
    return result;
}


at::Tensor muladd_bench_par(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c){
    // TORCH_CHECK(a.sizes() == b.sizes());
    // TORCH_CHECK(a.sizes() == c.sizes());
    // TORCH_CHECK(a.dtype() == at::kComplexDouble);
    // TORCH_CHECK(b.dtype() == at::kComplexDouble);
    // TORCH_CHECK(c.dtype() == at::kComplexDouble);
    // TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    // TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor c_contig = c.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());

    const c10::complex<double>* a_ptr = a_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* b_ptr = b_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* c_ptr = c_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* result_ptr = result.mutable_data_ptr<c10::complex<double>>();

    int64_t len = result.numel();

    at::parallel_for(0, len, len/32, [&](int64_t start, int64_t end){
    for (int64_t i = start; i < end; i++) {
        result_ptr[i] = a_ptr[i] * b_ptr[i] + c_ptr[i];
    }
    });

    return result;
}


at::Tensor muladd_bench_nopar_time(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c){
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.sizes() == c.sizes());
    TORCH_CHECK(a.dtype() == at::kComplexDouble);
    TORCH_CHECK(b.dtype() == at::kComplexDouble);
    TORCH_CHECK(c.dtype() == at::kComplexDouble);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor c_contig = c.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());

    auto start = std::chrono::high_resolution_clock::now();

    const c10::complex<double>* a_ptr = a_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* b_ptr = b_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* c_ptr = c_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* result_ptr = result.mutable_data_ptr<c10::complex<double>>();

    int64_t len = result.numel();

    for (int64_t i = 0; i < len; i++) {
        result_ptr[i] = a_ptr[i] * b_ptr[i] + c_ptr[i];
    }

    auto stop = std::chrono::high_resolution_clock::now();

    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
    result_ptr[0] = c10::complex<double>(dur.count());

    return result;
}


at::Tensor muladd_bench_par_time(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c){
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.sizes() == c.sizes());
    TORCH_CHECK(a.dtype() == at::kComplexDouble);
    TORCH_CHECK(b.dtype() == at::kComplexDouble);
    TORCH_CHECK(c.dtype() == at::kComplexDouble);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor c_contig = c.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());

    auto start = std::chrono::high_resolution_clock::now();

    const c10::complex<double>* a_ptr = a_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* b_ptr = b_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* c_ptr = c_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* result_ptr = result.mutable_data_ptr<c10::complex<double>>();

    int64_t len = result.numel();

    at::parallel_for(0, len, len/32, [&](int64_t start, int64_t end){
    for (int64_t i = start; i < end; i++) {
        result_ptr[i] = a_ptr[i] * b_ptr[i] + c_ptr[i];
    }
    });

    auto stop = std::chrono::high_resolution_clock::now();

    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start);
    result_ptr[0] = c10::complex<double>(dur.count());

    return result;
}


at::Tensor mul_real_bench_nopar(const at::Tensor& a, const at::Tensor& b){
    // TORCH_CHECK(a.sizes() == b.sizes());
    // TORCH_CHECK(a.dtype() == at::kDouble);
    // TORCH_CHECK(b.dtype() == at::kDouble);
    // TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    // TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());

    const double* a_ptr = a_contig.const_data_ptr<double>();
    const double* b_ptr = b_contig.const_data_ptr<double>();
    double* result_ptr = result.mutable_data_ptr<double>();

    int64_t len = result.numel();

    for (int64_t i = 0; i < len; i++) {
        result_ptr[i] = a_ptr[i] * b_ptr[i];
    }
    return result;
}

at::Tensor mul_compl_bench_nopar(const at::Tensor& a, const at::Tensor& b){
    // TORCH_CHECK(a.sizes() == b.sizes());
    // TORCH_CHECK(a.dtype() == at::kComplexDouble);
    // TORCH_CHECK(b.dtype() == at::kComplexDouble);
    // TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    // TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());

    const c10::complex<double>* a_ptr = a_contig.const_data_ptr<c10::complex<double>>();
    const c10::complex<double>* b_ptr = b_contig.const_data_ptr<c10::complex<double>>();
    c10::complex<double>* result_ptr = result.mutable_data_ptr<c10::complex<double>>();

    int64_t len = result.numel();

    for (int64_t i = 0; i < len; i++) {
        result_ptr[i] = a_ptr[i] * b_ptr[i];
    }
    return result;
}


}
