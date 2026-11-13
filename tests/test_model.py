import unittest
import numpy as np
from arch.univariate import SkewStudent
from gjr_var.model import RollingGJR, quantiles, TAU

class ModelTests(unittest.TestCase):
    def test_skewt_quantiles(self):
        for eta,lam in [(8.,0.),(6.,-.4),(10.,.7)]:
            np.testing.assert_allclose(quantiles([eta],[lam])[0],SkewStudent().ppf(TAU,[eta,lam]),rtol=1e-12)
    def test_refit_schedule_and_causality(self):
        rng=np.random.default_rng(8)
        data=rng.standard_t(8,size=(110,1))
        model=RollingGJR(1,window=100,warmup=100,refit_every=5)
        for t,row in enumerate(data):
            due=model.refit_if_due()
            self.assertEqual(due,t in [100,105])
            prediction=model.forecast(str(t)).copy()
            if t>=100:
                mu,o,a,g,b=model.params[0,:5];h=model.h[0];eps=row[0]-mu
                expected=max(o+(a+g*(eps<0))*eps**2+b*h,1e-8)
            model.update(str(t),row,None)
            if t>=100: self.assertAlmostEqual(model.h[0],expected)
            self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(len(model.history),100)
    def test_invalid_configuration(self):
        with self.assertRaises(ValueError): RollingGJR(1,1500,1000)

if __name__=='__main__': unittest.main()
