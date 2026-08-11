from __future__ import annotations
import unittest
import torch
from torch import nn
from src.methods.baselines import BaselineController

class Tiny(nn.Module):
    def __init__(self): super().__init__(); self.shared=nn.Linear(2,3); self.a=nn.Linear(3,1); self.b=nn.Linear(3,1)
    def forward(self,x): h=torch.tanh(self.shared(x)); return self.a(h),self.b(h)

def loss_dict(m,x):
    a,b=m(x); return {'a':(a**2).mean()+.1,'b':((b-1)**2).mean()+.1}

class BaselineTests(unittest.TestCase):
    def test_runnable_baselines_one_step(self):
        names=['ls','si','rlw','dwa','uw','mgda','pcgrad','graddrop','cagrad','nash-mtl','moco','mgda-warm','famo']
        for name in names:
            with self.subTest(name=name):
                torch.manual_seed(1); m=Tiny(); x=torch.randn(4,2); ctl=BaselineController(name,m,('a','b'),{'gamma':1e-3,'weight_lr':1e-2,'rho':.1,'warm_steps':2},torch.device('cpu')); losses=loss_dict(m,x); ctl.step(losses)
                if name=='famo': ctl.famo_post_step(loss_dict(m,x))
                self.assertTrue(torch.isfinite(ctl.w).all()); self.assertAlmostEqual(float(ctl.w.sum()),1.,places=5)
    def test_modo_three_samples(self):
        m=Tiny(); ctl=BaselineController('modo',m,('a','b'),{'gamma':1e-3,'weight_lr':1e-2,'rho':.1},torch.device('cpu'))
        xs=[torch.randn(4,2) for _ in range(3)]; ctl.step(loss_dict(m,xs[0]),extra_loss_dicts=[loss_dict(m,xs[1]),loss_dict(m,xs[2])]); self.assertAlmostEqual(float(ctl.w.sum()),1.,places=5)
if __name__=='__main__': unittest.main()
