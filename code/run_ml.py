"""Run the eight ML models only; all run_all.py flags remain available."""
import sys
from ml_models import ML_CATALOG
from run_all import main
if __name__=='__main__':
    if '--models' not in sys.argv:sys.argv+=['--models',*ML_CATALOG]
    main()
