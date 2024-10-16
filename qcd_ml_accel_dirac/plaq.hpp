#include <torch/extension.h>
#include <vector>

// #ifndef _OPENMP
// #define _OPENMP
// #endif
// #include <ATen/ParallelOpenMP.h>
// #include <omp.h>

// file for the plaquette action

namespace qcd_ml_accel_dirac{

/**
 * @brief compute the plaquette action
 * 
 * @param U the gauge field configuration, (4,Lx,Ly,Lz,Lt,3,3)-tensor
 * @param g the gauge coupling constant
 * @return double, the normalized action
 */
double plaq_action_cpu (const at::Tensor& U, double g);

}

