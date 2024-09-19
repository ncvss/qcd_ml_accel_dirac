project('qcd_ml_accel_dirac', 'cpp', version:'0.0.1',
    subproject_dir: 'subprojects',
    default_options : ['cpp_std=c++17'],)

pymod = import('python')
python = pymod.find_installation('python3')
pydep = python.dependency()

subdir('csrc')
subdir('pysrc')

#dirac_cpp_dep = dependency('qcd_ml_accel_dirac_cpp', fallback: ['qcd_ml_accel_dirac_cpp', 'dirac_cpp_dep'])

#python.extension_module(
#    'qcd_ml_accel_dirac',
#    'src/qcd_ml_accel_dirac/__init__.py', 'src/qcd_ml_accel_dirac/dirac.py',
#    dependencies: cppdep,
#    )
#python.install_sources('src/qcd_ml_accel_dirac/__init__.py', 'src/qcd_ml_accel_dirac/dirac.py'
#    , pure:true
#    , subdir: 'qcd_ml_accel_dirac')
