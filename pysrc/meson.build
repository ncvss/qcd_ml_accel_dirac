project('qcd_ml_accel_dirac', 'cpp', version:'0.0.1')

pymod = import('python')
python = pymod.find_installation('python3')
pydep = python.dependency()

python.install_sources('qcd_ml_accel_dirac/__init__.py', 'qcd_ml_accel_dirac/dirac.py'
    , pure:true
    , subdir: 'qcd_ml_accel_dirac')
