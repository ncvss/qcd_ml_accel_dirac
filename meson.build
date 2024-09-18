project('qcd_ml_accel_dirac', 'cpp', version:'0.0.1'
  , default_options : ['cpp_std=c++17'])

fs = import('fs')

add_global_arguments('-DTORCH_API_INCLUDE_EXTENSION_H', language:'cpp')

CXX = meson.get_compiler('cpp')

pymod = import('python')
python = pymod.find_installation('python3')
pydep = python.dependency()


#--------------------------------------------------------------------------------#
# Headers and shared objects from torch                                          #
# Obtain include and library paths from torch installation.
torch_include_dirs = []
foreach tin: run_command(python, ['-c', 'import torch.utils.cpp_extension; print("\\n".join(torch.utils.cpp_extension.include_paths()))'], check: true).stdout().split('\n')
    tins = tin.strip()
    if tins != ''
        if fs.is_dir(tins)
            torch_include_dirs += tins
        endif
    endif
endforeach
torch_includes = include_directories(torch_include_dirs)

torch_ld_paths = []
foreach tlp: run_command(python, ['-c', 'import torch.utils.cpp_extension; print("\\n".join(torch.utils.cpp_extension.library_paths()))'], check: true).stdout().split('\n')
    tpls =  tlp.strip()
    if tpls != ''
        torch_ld_paths += tpls
    endif
endforeach

needed_by_torch = ['c10', 'torch', 'torch_cpu', 'torch_python']
torch_deps = []
foreach libname: needed_by_torch
    torch_deps += CXX.find_library(libname, dirs: torch_ld_paths)
endforeach

#--------------------------------------------------------------------------------#
# PYBIND11 related configs                                                       #

pybind11_attrs = ['PYBIND11_BUILD_ABI', 'PYBIND11_COMPILER_TYPE', 'PYBIND11_STDLIB']
pybind11_attr_vals = []

foreach attr: pybind11_attrs
    pybind11_attr_vals += run_command(python, ['-c', 'import torch, sys; val = getattr(torch._C, "_" + sys.argv[1]); print(val if val is not None else "")', attr], check: true).stdout().strip()
endforeach

foreach i: range(3)
    val = pybind11_attr_vals[i]
    if val != ''
        add_global_arguments('-D@0@="@1@"'.format(pybind11_attrs[i], val), language: 'cpp')
    endif
endforeach

#--------------------------------------------------------------------------------#
# Defines that are required for torch compatibility                              #

# ABI compatibility
cxx11abi = run_command(python, ['-c', 'import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)'], check: true).stdout().strip()
add_global_arguments('-D_GLIBCXX_USE_CXX11_ABI=@0@'.format(cxx11abi), language: 'cpp')



python.extension_module(
    '_C'
    , ['src/qcd_ml_accel_dirac_cpp/cppops.cpp']
    , dependencies: torch_deps
    , install:true
    , subdir: 'qcd_ml_accel_dirac_cpp'
    , cpp_args: ['-DTORCH_EXTENSION_NAME=_C']
    , include_directories: torch_includes)

python.install_sources('src/qcd_ml_accel_dirac_cpp/__init__.py'
    , pure:false
    , subdir: 'qcd_ml_accel_dirac_cpp')


python.install_sources('src/qcd_ml_accel_dirac/__init__.py', 'src/qcd_ml_accel_dirac/dirac.py'
    , pure:false
    , subdir: 'qcd_ml_accel_dirac')
