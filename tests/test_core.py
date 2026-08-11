from __future__ import annotations
import math, tempfile, unittest
from pathlib import Path
import torch
from torch import nn
from src.methods.common import simplex_projection,named_trainable,task_gradients,blend_grads
from src.methods.lmo import build_lmo_direction,zeropower_newton_schulz5
from src.methods.alg1 import EntropicLMOMGDA,entropic_weight_update
from src.methods.alg4 import QuadraticLMOMGDA,quadratic_weight_gradient,quadratic_weight_update
from src.methods.baselines import min_norm_weights,BaselineController
from src.metrics import delta_m,DenseMetricAccumulator
from src.reported import ZHANG_REPORTED
from src.data.dense import DenseNPYDataset
from src.report import aggregate_results,add_ranks

class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.hidden=nn.Linear(3,4,bias=True); self.pred_a=nn.Linear(4,1); self.pred_b=nn.Linear(4,1)
        self.output_parameter_names={n for n,_ in self.named_parameters() if n.startswith('pred_')}
    def forward(self,x):
        h=torch.tanh(self.hidden(x)); return self.pred_a(h),self.pred_b(h)

class CoreTests(unittest.TestCase):
    def test_empty_report_is_plot_only_safe(self):
        frame=aggregate_results([],None)
        self.assertIn("dataset",frame.columns); self.assertIn("method",frame.columns)
        ranked=add_ranks(frame)
        self.assertTrue(ranked.empty); self.assertIn("mr",ranked.columns)

    def test_simplex_projection(self):
        y=torch.tensor([-2.,.2,3.,.7]); w=simplex_projection(y)
        self.assertTrue(torch.all(w>=0)); self.assertAlmostEqual(float(w.sum()),1.0,places=6)
    def test_l2_lmo_exact_ball_and_support(self):
        m=Tiny(); params=named_trainable(m); grads=[torch.randn_like(p) for _,p in params]; R=2.5
        info=build_lmo_direction(grads,params,geometry='l2',radius=R)
        norm=math.sqrt(sum(float((d*d).sum()) for d in info.direction)); self.assertAlmostEqual(norm,R,places=5)
        gnorm=math.sqrt(sum(float((g*g).sum()) for g in grads)); self.assertAlmostEqual(float(info.support),R*gnorm,places=4)
    def test_sign_lmo(self):
        m=Tiny(); params=named_trainable(m); grads=[torch.randn_like(p) for _,p in params]; info=build_lmo_direction(grads,params,geometry='sign',radius=3.)
        for g,d in zip(grads,info.direction): self.assertTrue(torch.equal(d,-3*torch.sign(g)))
    def test_newton_schulz_direction(self):
        g=torch.randn(7,5); z=zeropower_newton_schulz5(g,steps=7)
        self.assertEqual(z.shape,g.shape); self.assertTrue(torch.isfinite(z).all()); self.assertGreater(float((g*z).sum()),0.0)
    def test_min_norm_weights(self):
        grads=[[torch.tensor([1.,0.])],[torch.tensor([-1.,0.])]]; w=min_norm_weights(grads)
        self.assertAlmostEqual(float(w.sum()),1.0,places=6); self.assertAlmostEqual(float(w[0]),0.5,places=3)
    def test_algorithm1_positive_sign_update_exact(self):
        w=torch.tensor([0.5,0.5]); scores=torch.tensor([-0.25,-1.25]); eta=0.7
        got=entropic_weight_update(w,scores,eta)
        expected=torch.softmax(torch.log(w)+eta*scores,dim=0)
        self.assertTrue(torch.allclose(got,expected,atol=1e-7,rtol=1e-7))
        # Positive score sign means the less-negative first score gains relative weight.
        self.assertGreater(float(got[0]),0.5)

    def test_algorithm4_centered_gradient_and_projection_exact(self):
        w=torch.tensor([0.6,0.4]); scores=torch.tensor([-0.4,-1.2]); dual=torch.tensor(2.0)
        grad=quadratic_weight_gradient(w,scores,dual_norm=dual,radius=2.0,rho=0.5)
        expected=-(dual*scores/2.0)+0.5*(w-torch.tensor([0.5,0.5]))
        self.assertTrue(torch.allclose(grad,expected,atol=1e-7,rtol=1e-7))
        got,got_grad=quadratic_weight_update(w,scores,dual_norm=dual,radius=2.0,beta=0.3,rho=0.5)
        expected_w=simplex_projection(w-0.3*expected)
        self.assertTrue(torch.allclose(got_grad,expected,atol=1e-7,rtol=1e-7))
        self.assertTrue(torch.allclose(got,expected_w,atol=1e-7,rtol=1e-7))

    def test_algorithm1_and_4_stay_on_simplex(self):
        x=torch.randn(5,3)
        for cls,cfg in [(EntropicLMOMGDA,{'lmo':'l2','radius':1.,'gamma':1e-3,'eta':.5}), (QuadraticLMOMGDA,{'lmo':'l2','radius':1.,'gamma':1e-3,'beta':.5,'rho':.1})]:
            m=Tiny(); a,b=m(x); losses={'a':(a**2).mean(),'b':((b-1)**2).mean()}; ctl=cls(m,('a','b'),cfg,torch.device('cpu')); info=ctl.step(losses)
            self.assertTrue(torch.isfinite(ctl.w).all()); self.assertTrue(torch.all(ctl.w>=0)); self.assertAlmostEqual(float(ctl.w.sum()),1.,places=6); self.assertGreaterEqual(info['lmo_support'],-1e-6)

    def test_scion_conv_radius_scaling(self):
        # With the exact SVD polar factor, the flattened spectral norm equals the configured scale.
        g=torch.randn(8,4,3,3); p=nn.Parameter(torch.zeros_like(g)); params=[("conv.weight",p)]
        info=build_lmo_direction([g],params,geometry="spectral",radius=1.0,spectral_backend="svd",width_scaling="scion")
        spec=float(torch.linalg.matrix_norm(info.direction[0].reshape(8,-1).float(),ord=2))
        expected=math.sqrt(8/4)/(3*3)
        self.assertAlmostEqual(spec,expected,places=5)

    def test_cityscapes_label7_layout(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for split in ("train","val"):
                for sub in ("image","label_7","depth"):
                    (root/split/sub).mkdir(parents=True,exist_ok=True)
                np.save(root/split/"image"/"0.npy",np.zeros((8,8,3),dtype=np.float32))
                np.save(root/split/"label_7"/"0.npy",np.zeros((8,8),dtype=np.int64))
                np.save(root/split/"depth"/"0.npy",np.ones((8,8,1),dtype=np.float32))
            ds=DenseNPYDataset(root,"cityscapes",train=True)
            item=ds[0]
            self.assertEqual(tuple(item["image"].shape),(3,8,8))
            self.assertEqual(tuple(item["semantic"].shape),(8,8))
            self.assertEqual(tuple(item["depth"].shape),(1,8,8))

    def test_city_depth_relative_error_is_not_percent_scaled(self):
        acc=DenseMetricAccumulator("cityscapes")
        pred={"semantic":torch.zeros(1,7,1,1),"depth":torch.tensor([[[[1.5]]]])}
        pred["semantic"][:,0]=1.0
        batch={"semantic":torch.zeros(1,1,1,dtype=torch.long),"depth":torch.tensor([[[[1.0]]]])}
        acc.update(pred,batch); out=acc.compute()
        self.assertAlmostEqual(out["abs_err"],0.5,places=7)
        self.assertAlmostEqual(out["rel_err"],0.5,places=7)

    def test_delta_formula_matches_reported_city_mgda(self):
        stl=ZHANG_REPORTED['cityscapes']['STL']; mgda=ZHANG_REPORTED['cityscapes']['MGDA']; val=delta_m({k:mgda[k] for k in stl},stl)
        self.assertAlmostEqual(val,mgda['delta_m'],places=1)

if __name__=='__main__': unittest.main()
