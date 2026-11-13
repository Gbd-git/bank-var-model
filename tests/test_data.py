import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
class DataTests(unittest.TestCase):
    def test_csv_npz_and_source_alignment(self):
        d=np.load(ROOT/'data/bank_returns.npz',allow_pickle=False)
        csv=pd.read_csv(ROOT/'data/bank_returns.csv',index_col=0,float_precision='round_trip')
        raw=pd.read_csv(ROOT/'data/raw/source_returns.csv',index_col=0,float_precision='round_trip')
        np.testing.assert_array_equal(csv.to_numpy(),d['y'])
        np.testing.assert_array_equal(csv.index.to_numpy(str),d['dates'])
        np.testing.assert_array_equal(csv.columns.to_numpy(str),d['banks'])
        np.testing.assert_allclose(raw.loc[d['dates'],d['banks']].to_numpy()*100,d['y'],atol=1e-10,rtol=0)
        self.assertEqual(d['y'].shape,(4575,14))
