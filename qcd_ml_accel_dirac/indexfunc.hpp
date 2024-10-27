#include <cstdint>

// file for inline functions to access indices

namespace qcd_ml_accel_dirac{

// function to calculate the pointer address from coordinates for tensors with 6 dimensions
inline int64_t ptridx6 (int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f,
                        int64_t* stridearr){
    return a*stridearr[0] + b*stridearr[1] + c*stridearr[2]
           + d*stridearr[3] + e*stridearr[4] + f*stridearr[5];
}

// function to calculate the pointer address from coordinates for tensors with 7 dimensions
inline int64_t ptridx7 (int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f,
                        int64_t g, int64_t* stridearr){
    return a*stridearr[0] + b*stridearr[1] + c*stridearr[2]
           + d*stridearr[3] + e*stridearr[4] + f*stridearr[5]
           + g*stridearr[6];
}

// function to calculate the pointer address from coordinates for tensors with 8 dimensions
inline int64_t ptridx8 (int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f,
                        int64_t g, int64_t h, int64_t* stridearr){
    return a*stridearr[0] + b*stridearr[1] + c*stridearr[2]
           + d*stridearr[3] + e*stridearr[4] + f*stridearr[5]
           + g*stridearr[6] + h*stridearr[7];
}

// function to calculate the pointer address from indices for tensors with 6 dimensions,
// where the spatial dimensions (0,1,2,3) are each split into blocks of size bls
inline int64_t ptrblidx6 (int64_t bla, int64_t blb, int64_t blc, int64_t bld, int64_t bls,
                          int64_t a, int64_t b, int64_t c, int64_t d,
                          int64_t e, int64_t f, int64_t* stridearr){
    return (bla*bls + a)*stridearr[0] + (blb*bls + b)*stridearr[1]
           + (blc*bls + c)*stridearr[2] + (bld*bls + d)*stridearr[3]
           + e*stridearr[4] + f*stridearr[5];
}

// function to calculate the pointer address from indices for tensors with 7 dimensions,
// where the spatial dimensions (1,2,3,4) are each split into blocks of size bls
inline int64_t ptrblidx7 (int64_t z,
                          int64_t bla, int64_t blb, int64_t blc, int64_t bld, int64_t bls,
                          int64_t a, int64_t b, int64_t c, int64_t d,
                          int64_t e, int64_t f, int64_t* stridearr){
    return (bla*bls + a)*stridearr[1] + (blb*bls + b)*stridearr[2]
           + (blc*bls + c)*stridearr[3] + (bld*bls + d)*stridearr[4]
           + e*stridearr[5] + f*stridearr[6] + z*stridearr[0];
}

}
