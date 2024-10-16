#include <torch/extension.h>


namespace qcd_ml_accel_dirac{

// lookup tables for the result of the matrix product vmu = gammamu @ v
// gamx[mu][i] is the spin component of v that is proportional to spin component i of vmu
// gamf[mu][i] is the prefactor of spin component i of vmu
// in total, the spin component i of vmu is:
// vmu_{i} = gamf[mu][i] * v_{gamx[mu][i]}

// gamx[mu][i] is the spin component of v that is proportional to spin component i of gammamu @ v
static const int64_t gamx [4][4] =
    {{3, 2, 1, 0},
     {3, 2, 1, 0},
     {2, 3, 0, 1},
     {2, 3, 0, 1} };

// gamf[mu][i] is the prefactor of spin component i of gammamu @ v
// gamf = [[ i, i,-i,-i],
//         [-1, 1, 1,-1],
//         [ i,-i,-i, i],
//         [ 1, 1, 1, 1] ]
static const c10::complex<double> gamf [4][4] =
    {{c10::complex<double>(0, 1), c10::complex<double>(0, 1), c10::complex<double>(0,-1), c10::complex<double>(0,-1)},
     { -1,   1,   1,  -1},
     {c10::complex<double>(0, 1), c10::complex<double>(0,-1), c10::complex<double>(0,-1), c10::complex<double>(0, 1)},
     {  1,   1,   1,   1} };


// lookup tables for vmunu = sigmamunu @ v
// munu is a single index {0,1,2,3,4,5}, corresponding to {(1,0),(2,0),(2,1),(3,0),(3,1),(3,2)}
// the ith spin component of vmunu is:
// vmunu_{i} = sigf[munu][i] * v_{sigx[munu][i]}

// sigx[munu][i] is the spin component of v that is proportional to spin component i of sigmamunu @ v
static const int64_t sigx [6][4] =
    {{0,1,2,3},
     {1,0,3,2},
     {1,0,3,2},
     {1,0,3,2},
     {1,0,3,2},
     {0,1,2,3} };

// sigf[munu][i] is the prefactor of spin component i of sigmamunu @ v
// sigf = [[ i,-i, i,-i],
//         [ 1,-1, 1,-1],
//         [ i, i, i, i],
//         [-i,-i, i, i],
//         [ 1,-1,-1, 1],
//         [-i, i, i,-i] ]
static const c10::complex<double> sigf [6][4] =
    {{c10::complex<double>(0, 1), c10::complex<double>(0,-1), c10::complex<double>(0, 1), c10::complex<double>(0,-1)},
     {   1,  -1,   1,  -1},
     {c10::complex<double>(0, 1), c10::complex<double>(0, 1), c10::complex<double>(0, 1), c10::complex<double>(0, 1)},
     {c10::complex<double>(0,-1), c10::complex<double>(0,-1), c10::complex<double>(0, 1), c10::complex<double>(0, 1)},
     {   1,  -1,  -1,   1},
     {c10::complex<double>(0,-1), c10::complex<double>(0, 1), c10::complex<double>(0, 1), c10::complex<double>(0,-1)} };

}
