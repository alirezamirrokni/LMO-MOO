from __future__ import annotations
import unittest
from src.experiments.common import train_method

class SmokeTests(unittest.TestCase):
    def test_one_epoch_alg1_synthetic_city(self):
        cfg={'experiment':{'epochs':1,'average_last_epochs':1,'checkpoint_interval':1,'device':'cpu','deterministic':True},'model':{'width':0.0625},
             'data':{'cityscapes':{'root':'__synthetic__','batch_size':2,'workers':0,'pin_memory':False,'synthetic_train':2,'synthetic_val':2,'synthetic_h':32,'synthetic_w':32}},
             'methods':{'Alg1-L2':{'kind':'alg1','lmo':'l2','gamma':1e-4,'eta':.1,'radius':1.}}}
        r=train_method(cfg,'cityscapes','Alg1-L2',123,progress=False); self.assertTrue(r['complete']); self.assertIn('miou',r['final']); self.assertEqual(len(r['history']),1)

    def test_one_epoch_alg4_spectral_synthetic_nyu(self):
        cfg={'experiment':{'epochs':1,'average_last_epochs':1,'checkpoint_interval':1,'device':'cpu','deterministic':True},'model':{'width':0.0625},
             'data':{'nyuv2':{'root':'__synthetic__','batch_size':2,'workers':0,'pin_memory':False,'synthetic_train':2,'synthetic_val':2,'synthetic_h':32,'synthetic_w':32}},
             'methods':{'Alg4-Spectral':{'kind':'alg4','lmo':'spectral','gamma':1e-4,'beta':.1,'rho':.1,'radius':1.,'spectral_backend':'newton_schulz','ns_steps':3,'width_scaling':'scion'}}}
        r=train_method(cfg,'nyuv2','Alg4-Spectral',124,progress=False); self.assertTrue(r['complete']); self.assertIn('normal_mean',r['final']); self.assertEqual(len(r['history']),1)

if __name__=='__main__': unittest.main()
