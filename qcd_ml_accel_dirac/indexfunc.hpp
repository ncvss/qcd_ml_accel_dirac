
// file for inline functions to access indices

namespace qcd_ml_accel_dirac{

// function to calculate the pointer address from coordinates
// for tensors with 6 dimensions
inline int64_t ptridx6 (int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f,
                        int64_t* stridearr){
    return a*stridearr[0] + b*stridearr[1] + c*stridearr[2]
           + d*stridearr[3] + e*stridearr[4] + f*stridearr[5];
}
// for tensors with 7 dimensions
inline int64_t ptridx7 (int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f,
                        int64_t g, int64_t* stridearr){
    return a*stridearr[0] + b*stridearr[1] + c*stridearr[2]
           + d*stridearr[3] + e*stridearr[4] + f*stridearr[5]
           + g*stridearr[6];
}
// for tensors with 8 dimensions
inline int64_t ptridx8 (int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f,
                        int64_t g, int64_t h, int64_t* stridearr){
    return a*stridearr[0] + b*stridearr[1] + c*stridearr[2]
           + d*stridearr[3] + e*stridearr[4] + f*stridearr[5]
           + g*stridearr[6] + h*stridearr[7];
}

}
