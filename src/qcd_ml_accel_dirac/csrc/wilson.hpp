#include <torch/extension.h>

#include <vector>

// file for the c++ pointer version of dirac wilson and dirac wilson clover



// define imaginary unit (for brevity of following definitions)
static const c10::complex<double> iun (0,1);

// lookup tables for the result of the matrix product vmu = gammamu @ v
// gamx[mu][i] is the spin component of v that is proportional to spin component i of vmu
// gamf[mu][i] is the prefactor of that spin component of v
// in total, the spin component i of vmu is:
// vmu_{i} = gamf[mu][i] * v_{gamx[mu][i]}
static const int64_t gamx [4][4] = {{3, 2, 1, 0},
                                    {3, 2, 1, 0},
                                    {2, 3, 0, 1},
                                    {2, 3, 0, 1} };
static const c10::complex<double> gamf [4][4] = {{iun, iun,-iun,-iun},
                                                 { -1,   1,   1,  -1},
                                                 {iun,-iun,-iun, iun},
                                                 {  1,   1,   1,   1} };

// lookup tables for vmunu = sigmamunu @ v
// munu is a single index {0,1,2,3,4,5}, corresponding to {(1,0),(2,0),(2,1),(3,0),(3,1),(3,2)}
// the ith spin component of vmunu is:
// vmunu_{i} = sigf[munu][i] * v_{sigx[munu][i]}
static const int64_t sigx [6][4] = {{0,1,2,3},
                                    {1,0,3,2},
                                    {1,0,3,2},
                                    {1,0,3,2},
                                    {1,0,3,2},
                                    {0,1,2,3}};
static const c10::complex<double> sigf [6][4] = {{ iun,-iun, iun,-iun},
                                                 {   1,  -1,   1,  -1},
                                                 { iun, iun, iun, iun},
                                                 {-iun,-iun, iun, iun},
                                                 {   1,  -1,  -1,   1},
                                                 {-iun, iun, iun,-iun}};



at::Tensor dw_call_p (const at::Tensor& U, const at::Tensor& v, double mass){

    // check for correct size of gauge and vector field
    TORCH_CHECK(U.dim() == 7);
    TORCH_CHECK(v.dim() == 6);
    TORCH_CHECK(U.size(0) == 4);
    TORCH_CHECK(U.size(1) == v.size(0));
    TORCH_CHECK(U.size(2) == v.size(1));
    TORCH_CHECK(U.size(3) == v.size(2));
    TORCH_CHECK(U.size(4) == v.size(3));
    TORCH_CHECK(U.size(5) == 3);
    TORCH_CHECK(U.size(6) == 3);
    TORCH_CHECK(v.size(4) == 4);
    TORCH_CHECK(v.size(5) == 3);

    TORCH_CHECK(U.dtype() == at::kComplexDouble);
    TORCH_CHECK(v.dtype() == at::kComplexDouble);

    TORCH_INTERNAL_ASSERT(U.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(v.device().type() == at::DeviceType::CPU);

    at::Tensor U_contig = U.contiguous();
    at::Tensor v_contig = v.contiguous();

    // size of space-time, spin and gauge axes
    int64_t vsize [6];
    for (int64_t sj = 0; sj < 6; sj++){
        vsize[sj] = v_contig.size(sj);
    }
    int64_t u_size [7];
    for (int64_t sj = 0; sj < 7; sj++){
        u_size[sj] = U_contig.size(sj);
    }

    // strides of the memory blocks
    int64_t vstride [6];
    vstride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride[sj] = vstride[sj+1] * vsize[sj+1];
    }
    int64_t ustride [7];
    ustride[6] = 1;
    for (int64_t sj = 5; sj >= 0; sj--){
        ustride[sj] = ustride[sj+1] * u_size[sj+1];
    }

    at::Tensor result = torch::empty(vsize, v.options());

    const c10::complex<double>* U_ptr = U_contig.data_ptr<c10::complex<double>>();
    const c10::complex<double>* v_ptr = v_contig.data_ptr<c10::complex<double>>();
    c10::complex<double>* res_ptr = result.data_ptr<c10::complex<double>>();


    // iterate over the whole field
    // x,y,z,t are the space-time indices of U, v and result
    // s is the spin index of v and result
    // g is the gauge index of result and the first gauge index of U
    // gi is the gauge index of v and the second gauge index of U, which is summed over
    for (int64_t x = 0; x < vsize[0]; x++){
        for (int64_t y = 0; y < vsize[1]; y++){
            for (int64_t z = 0; z < vsize[2]; z++){
                for (int64_t t = 0; t < vsize[3]; t++){
                    for (int64_t s = 0; s < vsize[4]; s++){
                        for (int64_t g = 0; g < vsize[5]; g++){
                            // mass term
                            res_ptr[  x *vstride[0]
                                    + y *vstride[1]
                                    + z *vstride[2]
                                    + t *vstride[3]
                                    + s *vstride[4]
                                    + g *vstride[5]] = (4.0 + mass) * v_ptr[  x *vstride[0]
                                                                            + y *vstride[1]
                                                                            + z *vstride[2]
                                                                            + t *vstride[3]
                                                                            + s *vstride[4]
                                                                            + g *vstride[5]];
                            // hop terms written out for mu = 0, 1, 2, 3
                            // sum over gi corresponds to matrix product U_mu @ v
                            for (int64_t gi = 0; gi < vsize[5]; gi++){
                                res_ptr[  x *vstride[0]
                                        + y *vstride[1]
                                        + z *vstride[2]
                                        + t *vstride[3]
                                        + s *vstride[4]
                                        + g *vstride[5]]
                                +=( // mu = 0 term
                                    std::conj(
                                        U_ptr[    0 *ustride[0]
                                              + ((x-1+u_size[1])%u_size[1]) *ustride[1]
                                              +   y *ustride[2]
                                              +   z *ustride[3]
                                              +   t *ustride[4]
                                              +   gi*ustride[5]
                                              +   g *ustride[6] ]
                                    ) * (
                                        -v_ptr[  ((x-1+vsize[0])%vsize[0]) *vstride[0]
                                               +   y *vstride[1]
                                               +   z *vstride[2]
                                               +   t *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5] ]
                                        -gamf[0][s]* v_ptr[  ((x-1+vsize[0])%vsize[0]) *vstride[0]
                                                          +   y *vstride[1]
                                                          +   z *vstride[2]
                                                          +   t *vstride[3]
                                                          + gamx[0][s] *vstride[4]
                                                          +   gi*vstride[5] ]
                                    )
                                    + U_ptr[  0 *ustride[0]
                                            + x *ustride[1]
                                            + y *ustride[2]
                                            + z *ustride[3]
                                            + t *ustride[4]
                                            + g *ustride[5]
                                            + gi*ustride[6] ]
                                    * (
                                        -v_ptr[  ((x+1)%vsize[0]) *vstride[0]
                                               +   y *vstride[1]
                                               +   z *vstride[2]
                                               +   t *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5]]
                                        +gamf[0][s] * v_ptr[  ((x+1)%vsize[0]) *vstride[0]
                                                           +   y *vstride[1]
                                                           +   z *vstride[2]
                                                           +   t *vstride[3]
                                                           + gamx[0][s] *vstride[4]
                                                           +   gi*vstride[5]]
                                    )

                                    // mu = 1 term
                                    +std::conj(
                                        U_ptr[    1 *ustride[0]
                                              +   x *ustride[1]
                                              + ((y-1+u_size[2])%u_size[2]) *ustride[2]
                                              +   z *ustride[3]
                                              +   t *ustride[4]
                                              +   gi*ustride[5]
                                              +   g *ustride[6] ]
                                    ) * (
                                        -v_ptr[    x *vstride[0]
                                               + ((y-1+vsize[1])%vsize[1]) *vstride[1]
                                               +   z *vstride[2]
                                               +   t *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5] ]
                                        -gamf[1][s]* v_ptr[    x *vstride[0]
                                                          + ((y-1+vsize[1])%vsize[1]) *vstride[1]
                                                          +   z *vstride[2]
                                                          +   t *vstride[3]
                                                          + gamx[1][s] *vstride[4]
                                                          +   gi*vstride[5] ]
                                    )
                                    + U_ptr[  1 *ustride[0]
                                            + x *ustride[1]
                                            + y *ustride[2]
                                            + z *ustride[3]
                                            + t *ustride[4]
                                            + g *ustride[5]
                                            + gi*ustride[6] ]
                                    * (
                                        -v_ptr[    x *vstride[0]
                                               + ((y+1)%vsize[1]) *vstride[1]
                                               +   z *vstride[2]
                                               +   t *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5]]
                                        +gamf[1][s] * v_ptr[    x *vstride[0]
                                                           + ((y+1)%vsize[1]) *vstride[1]
                                                           +   z *vstride[2]
                                                           +   t *vstride[3]
                                                           + gamx[1][s] *vstride[4]
                                                           +   gi*vstride[5]]
                                    )

                                    // mu = 2 term
                                    +std::conj(
                                        U_ptr[    2 *ustride[0]
                                              +   x *ustride[1]
                                              +   y *ustride[2]
                                              + ((z-1+u_size[3])%u_size[3]) *ustride[3]
                                              +   t *ustride[4]
                                              +   gi*ustride[5]
                                              +   g *ustride[6] ]
                                    ) * (
                                        -v_ptr[    x *vstride[0]
                                               +   y *vstride[1]
                                               + ((z-1+vsize[2])%vsize[2]) *vstride[2]
                                               +   t *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5] ]
                                        -gamf[2][s]* v_ptr[    x *vstride[0]
                                                          +   y *vstride[1]
                                                          + ((z-1+vsize[2])%vsize[2]) *vstride[2]
                                                          +   t *vstride[3]
                                                          + gamx[2][s] *vstride[4]
                                                          +   gi*vstride[5] ]
                                    )
                                    + U_ptr[  2 *ustride[0]
                                            + x *ustride[1]
                                            + y *ustride[2]
                                            + z *ustride[3]
                                            + t *ustride[4]
                                            + g *ustride[5]
                                            + gi*ustride[6] ]
                                    * (
                                        -v_ptr[    x *vstride[0]
                                               +   y *vstride[1]
                                               + ((z+1)%vsize[2]) *vstride[2]
                                               +   t *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5]]
                                        +gamf[2][s] * v_ptr[    x *vstride[0]
                                                           +   y *vstride[1]
                                                           + ((z+1)%vsize[2]) *vstride[2]
                                                           +   t *vstride[3]
                                                           + gamx[2][s] *vstride[4]
                                                           +   gi*vstride[5]]
                                    )

                                    // mu = 3 term
                                    +std::conj(
                                        U_ptr[    3 *ustride[0]
                                              +   x *ustride[1]
                                              +   y *ustride[2]
                                              +   z *ustride[3]
                                              + ((t-1+u_size[4])%u_size[4]) *ustride[4]
                                              +   gi*ustride[5]
                                              +   g *ustride[6] ]
                                    ) * (
                                        -v_ptr[    x *vstride[0]
                                               +   y *vstride[1]
                                               +   z *vstride[2]
                                               + ((t-1+vsize[3])%vsize[3]) *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5] ]
                                        -gamf[3][s]* v_ptr[    x *vstride[0]
                                                          +   y *vstride[1]
                                                          +   z *vstride[2]
                                                          + ((t-1+vsize[3])%vsize[3]) *vstride[3]
                                                          + gamx[3][s] *vstride[4]
                                                          +   gi*vstride[5] ]
                                    )
                                    + U_ptr[  3 *ustride[0]
                                            + x *ustride[1]
                                            + y *ustride[2]
                                            + z *ustride[3]
                                            + t *ustride[4]
                                            + g *ustride[5]
                                            + gi*ustride[6] ]
                                    * (
                                        -v_ptr[    x *vstride[0]
                                               +   y *vstride[1]
                                               +   z *vstride[2]
                                               + ((t+1)%vsize[3]) *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5]]
                                        +gamf[3][s] * v_ptr[    x *vstride[0]
                                                           +   y *vstride[1]
                                                           +   z *vstride[2]
                                                           + ((t+1)%vsize[3]) *vstride[3]
                                                           + gamx[3][s] *vstride[4]
                                                           +   gi*vstride[5]]
                                    )
                                ) *0.5;
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}


at::Tensor dwc_call_p (const at::Tensor& U, const at::Tensor& v, const std::vector<at::Tensor>& F,
                     double mass, double csw){

    // check for correct size of gauge and vector field
    TORCH_CHECK(U.dim() == 7);
    TORCH_CHECK(v.dim() == 6);
    TORCH_CHECK(U.size(0) == 4);
    TORCH_CHECK(U.size(1) == v.size(0));
    TORCH_CHECK(U.size(2) == v.size(1));
    TORCH_CHECK(U.size(3) == v.size(2));
    TORCH_CHECK(U.size(4) == v.size(3));
    TORCH_CHECK(U.size(5) == 3);
    TORCH_CHECK(U.size(6) == 3);
    TORCH_CHECK(v.size(4) == 4);
    TORCH_CHECK(v.size(5) == 3);

    // check correct size of the field strength matrices
    TORCH_CHECK(F.size() == 6);
    for (int64_t fi = 0; fi < 6; fi++){
        TORCH_CHECK(F[fi].sizes() == U[0].sizes());
    }

    at::Tensor U_contig = U.contiguous();
    at::Tensor v_contig = v.contiguous();

    at::Tensor F10 = F[0].contiguous();
    at::Tensor F20 = F[1].contiguous();
    at::Tensor F21 = F[2].contiguous();
    at::Tensor F30 = F[3].contiguous();
    at::Tensor F31 = F[4].contiguous();
    at::Tensor F32 = F[5].contiguous();

    // size of space-time, spin and gauge axes
    int64_t vsize [6];
    for (int64_t sj = 0; sj < 6; sj++){
        vsize[sj] = v_contig.size(sj);
    }
    int64_t u_size [7];
    for (int64_t sj = 0; sj < 7; sj++){
        u_size[sj] = U_contig.size(sj);
    }
    int64_t fsize [6];
    for (int64_t sj = 0; sj < 6; sj++){
        fsize[sj] = F10.size(sj);
    }

    // strides of the memory blocks
    int64_t vstride [6];
    vstride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        vstride[sj] = vstride[sj+1] * vsize[sj+1];
    }
    int64_t ustride [7];
    ustride[6] = 1;
    for (int64_t sj = 5; sj >= 0; sj--){
        ustride[sj] = ustride[sj+1] * u_size[sj+1];
    }
    int64_t fstride [6];
    fstride[5] = 1;
    for (int64_t sj = 4; sj >= 0; sj--){
        fstride[sj] = fstride[sj+1] * fsize[sj+1];
    }

    at::Tensor result = torch::empty(vsize, v.options());

    const c10::complex<double>* U_ptr = U_contig.data_ptr<c10::complex<double>>();
    const c10::complex<double>* v_ptr = v_contig.data_ptr<c10::complex<double>>();

    const c10::complex<double>* F10_ptr = F10.data_ptr<c10::complex<double>>();
    const c10::complex<double>* F20_ptr = F20.data_ptr<c10::complex<double>>();
    const c10::complex<double>* F21_ptr = F21.data_ptr<c10::complex<double>>();
    const c10::complex<double>* F30_ptr = F30.data_ptr<c10::complex<double>>();
    const c10::complex<double>* F31_ptr = F31.data_ptr<c10::complex<double>>();
    const c10::complex<double>* F32_ptr = F32.data_ptr<c10::complex<double>>();

    c10::complex<double>* res_ptr = result.data_ptr<c10::complex<double>>();


    // iterate over the whole field
    // x,y,z,t are the space-time indices of U, v and result
    // s is the spin index of v and result
    // g is the gauge index of result and the first gauge index of U and F
    // gi is the gauge index of v and the second gauge index of U and F, which is summed over
    for (int64_t x = 0; x < vsize[0]; x++){
        for (int64_t y = 0; y < vsize[1]; y++){
            for (int64_t z = 0; z < vsize[2]; z++){
                for (int64_t t = 0; t < vsize[3]; t++){
                    for (int64_t s = 0; s < vsize[4]; s++){
                        for (int64_t g = 0; g < vsize[5]; g++){
                            // mass term
                            res_ptr[  x *vstride[0]
                                    + y *vstride[1]
                                    + z *vstride[2]
                                    + t *vstride[3]
                                    + s *vstride[4]
                                    + g *vstride[5]] = (4.0 + mass) * v_ptr[  x *vstride[0]
                                                                            + y *vstride[1]
                                                                            + z *vstride[2]
                                                                            + t *vstride[3]
                                                                            + s *vstride[4]
                                                                            + g *vstride[5]];
                            // hop terms written out for mu = 0, 1, 2, 3
                            // sum over gi corresponds to matrix product U_mu @ v
                            for (int64_t gi = 0; gi < vsize[5]; gi++){
                                res_ptr[  x *vstride[0]
                                        + y *vstride[1]
                                        + z *vstride[2]
                                        + t *vstride[3]
                                        + s *vstride[4]
                                        + g *vstride[5]]
                                += // dirac wilson terms
                                (   // mu = 0 term
                                    std::conj(
                                        U_ptr[    0 *ustride[0]
                                              + ((x-1+u_size[1])%u_size[1]) *ustride[1]
                                              +   y *ustride[2]
                                              +   z *ustride[3]
                                              +   t *ustride[4]
                                              +   gi*ustride[5]
                                              +   g *ustride[6] ]
                                    ) * (
                                        -v_ptr[  ((x-1+vsize[0])%vsize[0]) *vstride[0]
                                               +   y *vstride[1]
                                               +   z *vstride[2]
                                               +   t *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5] ]
                                        -gamf[0][s]* v_ptr[  ((x-1+vsize[0])%vsize[0]) *vstride[0]
                                                          +   y *vstride[1]
                                                          +   z *vstride[2]
                                                          +   t *vstride[3]
                                                          + gamx[0][s] *vstride[4]
                                                          +   gi*vstride[5] ]
                                    )
                                    + U_ptr[  0 *ustride[0]
                                            + x *ustride[1]
                                            + y *ustride[2]
                                            + z *ustride[3]
                                            + t *ustride[4]
                                            + g *ustride[5]
                                            + gi*ustride[6] ]
                                    * (
                                        -v_ptr[  ((x+1)%vsize[0]) *vstride[0]
                                               +   y *vstride[1]
                                               +   z *vstride[2]
                                               +   t *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5]]
                                        +gamf[0][s] * v_ptr[  ((x+1)%vsize[0]) *vstride[0]
                                                           +   y *vstride[1]
                                                           +   z *vstride[2]
                                                           +   t *vstride[3]
                                                           + gamx[0][s] *vstride[4]
                                                           +   gi*vstride[5]]
                                    )

                                    // mu = 1 term
                                    +std::conj(
                                        U_ptr[    1 *ustride[0]
                                              +   x *ustride[1]
                                              + ((y-1+u_size[2])%u_size[2]) *ustride[2]
                                              +   z *ustride[3]
                                              +   t *ustride[4]
                                              +   gi*ustride[5]
                                              +   g *ustride[6] ]
                                    ) * (
                                        -v_ptr[    x *vstride[0]
                                               + ((y-1+vsize[1])%vsize[1]) *vstride[1]
                                               +   z *vstride[2]
                                               +   t *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5] ]
                                        -gamf[1][s]* v_ptr[    x *vstride[0]
                                                          + ((y-1+vsize[1])%vsize[1]) *vstride[1]
                                                          +   z *vstride[2]
                                                          +   t *vstride[3]
                                                          + gamx[1][s] *vstride[4]
                                                          +   gi*vstride[5] ]
                                    )
                                    + U_ptr[  1 *ustride[0]
                                            + x *ustride[1]
                                            + y *ustride[2]
                                            + z *ustride[3]
                                            + t *ustride[4]
                                            + g *ustride[5]
                                            + gi*ustride[6] ]
                                    * (
                                        -v_ptr[    x *vstride[0]
                                               + ((y+1)%vsize[1]) *vstride[1]
                                               +   z *vstride[2]
                                               +   t *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5]]
                                        +gamf[1][s] * v_ptr[    x *vstride[0]
                                                           + ((y+1)%vsize[1]) *vstride[1]
                                                           +   z *vstride[2]
                                                           +   t *vstride[3]
                                                           + gamx[1][s] *vstride[4]
                                                           +   gi*vstride[5]]
                                    )

                                    // mu = 2 term
                                    +std::conj(
                                        U_ptr[    2 *ustride[0]
                                              +   x *ustride[1]
                                              +   y *ustride[2]
                                              + ((z-1+u_size[3])%u_size[3]) *ustride[3]
                                              +   t *ustride[4]
                                              +   gi*ustride[5]
                                              +   g *ustride[6] ]
                                    ) * (
                                        -v_ptr[    x *vstride[0]
                                               +   y *vstride[1]
                                               + ((z-1+vsize[2])%vsize[2]) *vstride[2]
                                               +   t *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5] ]
                                        -gamf[2][s]* v_ptr[    x *vstride[0]
                                                          +   y *vstride[1]
                                                          + ((z-1+vsize[2])%vsize[2]) *vstride[2]
                                                          +   t *vstride[3]
                                                          + gamx[2][s] *vstride[4]
                                                          +   gi*vstride[5] ]
                                    )
                                    + U_ptr[  2 *ustride[0]
                                            + x *ustride[1]
                                            + y *ustride[2]
                                            + z *ustride[3]
                                            + t *ustride[4]
                                            + g *ustride[5]
                                            + gi*ustride[6] ]
                                    * (
                                        -v_ptr[    x *vstride[0]
                                               +   y *vstride[1]
                                               + ((z+1)%vsize[2]) *vstride[2]
                                               +   t *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5]]
                                        +gamf[2][s] * v_ptr[    x *vstride[0]
                                                           +   y *vstride[1]
                                                           + ((z+1)%vsize[2]) *vstride[2]
                                                           +   t *vstride[3]
                                                           + gamx[2][s] *vstride[4]
                                                           +   gi*vstride[5]]
                                    )

                                    // mu = 3 term
                                    +std::conj(
                                        U_ptr[    3 *ustride[0]
                                              +   x *ustride[1]
                                              +   y *ustride[2]
                                              +   z *ustride[3]
                                              + ((t-1+u_size[4])%u_size[4]) *ustride[4]
                                              +   gi*ustride[5]
                                              +   g *ustride[6] ]
                                    ) * (
                                        -v_ptr[    x *vstride[0]
                                               +   y *vstride[1]
                                               +   z *vstride[2]
                                               + ((t-1+vsize[3])%vsize[3]) *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5] ]
                                        -gamf[3][s]* v_ptr[    x *vstride[0]
                                                          +   y *vstride[1]
                                                          +   z *vstride[2]
                                                          + ((t-1+vsize[3])%vsize[3]) *vstride[3]
                                                          + gamx[3][s] *vstride[4]
                                                          +   gi*vstride[5] ]
                                    )
                                    + U_ptr[  3 *ustride[0]
                                            + x *ustride[1]
                                            + y *ustride[2]
                                            + z *ustride[3]
                                            + t *ustride[4]
                                            + g *ustride[5]
                                            + gi*ustride[6] ]
                                    * (
                                        -v_ptr[    x *vstride[0]
                                               +   y *vstride[1]
                                               +   z *vstride[2]
                                               + ((t+1)%vsize[3]) *vstride[3]
                                               +   s *vstride[4]
                                               +   gi*vstride[5]]
                                        +gamf[3][s] * v_ptr[    x *vstride[0]
                                                           +   y *vstride[1]
                                                           +   z *vstride[2]
                                                           + ((t+1)%vsize[3]) *vstride[3]
                                                           + gamx[3][s] *vstride[4]
                                                           +   gi*vstride[5]]
                                    )
                                ) *0.5
                                
                                // dirac wilson clover improvement
                                -(
                                     F10_ptr[  x *fstride[0]
                                             + y *fstride[1]
                                             + z *fstride[2]
                                             + t *fstride[3]
                                             + g *fstride[4]
                                             + gi*fstride[5]]*sigf[0][s]*v_ptr[  x *vstride[0]
                                                                               + y *vstride[1]
                                                                               + z *vstride[2]
                                                                               + t *vstride[3]
                                                                               + sigx[0][s] *vstride[4]
                                                                               + gi*vstride[5]]
                                    +F20_ptr[  x *fstride[0]
                                             + y *fstride[1]
                                             + z *fstride[2]
                                             + t *fstride[3]
                                             + g *fstride[4]
                                             + gi*fstride[5]]*sigf[1][s]*v_ptr[  x *vstride[0]
                                                                               + y *vstride[1]
                                                                               + z *vstride[2]
                                                                               + t *vstride[3]
                                                                               + sigx[1][s] *vstride[4]
                                                                               + gi*vstride[5]]
                                    +F21_ptr[  x *fstride[0]
                                             + y *fstride[1]
                                             + z *fstride[2]
                                             + t *fstride[3]
                                             + g *fstride[4]
                                             + gi*fstride[5]]*sigf[2][s]*v_ptr[  x *vstride[0]
                                                                               + y *vstride[1]
                                                                               + z *vstride[2]
                                                                               + t *vstride[3]
                                                                               + sigx[2][s] *vstride[4]
                                                                               + gi*vstride[5]]
                                    +F30_ptr[  x *fstride[0]
                                             + y *fstride[1]
                                             + z *fstride[2]
                                             + t *fstride[3]
                                             + g *fstride[4]
                                             + gi*fstride[5]]*sigf[3][s]*v_ptr[  x *vstride[0]
                                                                               + y *vstride[1]
                                                                               + z *vstride[2]
                                                                               + t *vstride[3]
                                                                               + sigx[3][s] *vstride[4]
                                                                               + gi*vstride[5]]
                                    +F31_ptr[  x *fstride[0]
                                             + y *fstride[1]
                                             + z *fstride[2]
                                             + t *fstride[3]
                                             + g *fstride[4]
                                             + gi*fstride[5]]*sigf[4][s]*v_ptr[  x *vstride[0]
                                                                               + y *vstride[1]
                                                                               + z *vstride[2]
                                                                               + t *vstride[3]
                                                                               + sigx[4][s] *vstride[4]
                                                                               + gi*vstride[5]]
                                    +F32_ptr[  x *fstride[0]
                                             + y *fstride[1]
                                             + z *fstride[2]
                                             + t *fstride[3]
                                             + g *fstride[4]
                                             + gi*fstride[5]]*sigf[5][s]*v_ptr[  x *vstride[0]
                                                                               + y *vstride[1]
                                                                               + z *vstride[2]
                                                                               + t *vstride[3]
                                                                               + sigx[5][s] *vstride[4]
                                                                               + gi*vstride[5]]
                                )*csw*0.5;
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

