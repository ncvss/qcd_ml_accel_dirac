/// file for inline functions to access indices of arrays
// that treat complex numbers as 2 doubles in riri format

namespace qcd_ml_accel_dirac {

// address for complex gauge field pointer
inline int uixo (int t, int mu, int g, int gi, int vol){
    return mu*vol*18 + t*18 + g*6 + gi*2;
}
// address for complex vector field pointer
inline int vixo (int t, int g, int s){
    return t*24 + s*6 + g*2;
}
// address for hop lookup array pointer
inline int hixd (int t, int h, int d){
    return t*8 + h*2 + d;
}
// address for the gamma matrix prefactor
inline int gixd (int mu, int s){
    return mu*8 + s*2;
}
// address for field strength tensors (index order F[t,munu,g,gi])
inline int fixd (int t, int munu, int g, int gi){
    return t*108 + munu*18 + g*6 + gi*2;
}
// address for the sigma matrix prefactor
inline int sixd (int munu, int s){
    return munu*8 + s*2;
}

}
