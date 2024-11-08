// file for inline functions to access indices

namespace qcd_ml_accel_dirac{

// function to calculate the pointer address from coordinates for complex tensors with 6 dimensions
inline int ptridx6 (int a, int b, int c, int d, int e, int f, int com, int* stridearr){
    return a*stridearr[0] + b*stridearr[1] + c*stridearr[2]
           + d*stridearr[3] + e*stridearr[4] + f*stridearr[5] + com;
}

// function to calculate the pointer address from coordinates for tensors with 7 dimensions
inline int ptridx7 (int a, int b, int c, int d, int e, int f, int g, int com, int* stridearr){
    return a*stridearr[0] + b*stridearr[1] + c*stridearr[2]
           + d*stridearr[3] + e*stridearr[4] + f*stridearr[5]
           + g*stridearr[6] + com;
}

// function to calculate the pointer address from coordinates for tensors with 8 dimensions
inline int ptridx8 (int a, int b, int c, int d, int e, int f, int g, int h, int com, int* stridearr){
    return a*stridearr[0] + b*stridearr[1] + c*stridearr[2]
           + d*stridearr[3] + e*stridearr[4] + f*stridearr[5]
           + g*stridearr[6] + h*stridearr[7] + com;
}

}

// funny sse test code
// float square(float  num[4], float  num2[4]) {
//     float a[4]; 
    
//         a[0] = num[0] * num2[0];
//         a[1] = num[1] * num2[1];
//         a[2] = num[2] * num2[2];
//         a[3] = num[3] * num2[3];
    
//     return a[0] + a[1] + a[2] + a[3];
// }
// #include <xmmintrin.h>
// #include <immintrin.h>

// float square2(float  num[4], float  num2[4]) {
//     float a[4]; 

//     __m128 n = _mm_load_ps(num);
//     __m128 n2 = _mm_load_ps(num2);
    
//     __m128 a_ = _mm_mul_ps(n, n2);
//     _mm_store_ps(a, a_);
    
//     return a[0] + a[1] + a[2] + a[3];
// }

// #include <smmintrin.h>
// #include <emmintrin.h>

// int ptridx4 (int a, int b, int c, int d, int* stridearr)
// {
//     __m128i data = _mm_loadu_si128((__m128i*)stridearr);
//     __m128i data2 = _mm_set_epi32(d,c,b,a);
//     __m128i pro = _mm_mullo_epi32(data, data2);
//     int ret[4];
//     _mm_store_si128((__m128i*)ret, pro);
//     return ret[0] + ret[1] + ret[2] + ret[3];
// }

// int ptridx42 (int a, int b, int c, int d, int* stridearr)
// {
//     __m128i data = _mm_loadu_si128((__m128i*)stridearr);
//     __m128i data2 = _mm_set_epi32(d,c,b,a);
//     __m128i pro = _mm_mullo_epi32(data, data2);
//     __m128i temp = _mm_hadd_epi32(pro, pro);
//     temp = _mm_hadd_epi32(temp, temp);
//     return _mm_cvtsi128_si32(temp);
// }
