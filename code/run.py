"""Run from repository root: python code/run.py."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
from gjr_var.cli import main
if __name__=='__main__': main()
