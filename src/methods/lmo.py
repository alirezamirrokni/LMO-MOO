from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence
import torch
from src.methods.common import NamedParam, GradList, inner_lists

@dataclass
class LMOInfo:
    direction:GradList
    support:torch.Tensor   # -<G,D> >= 0
    normalized_dual_proxy:torch.Tensor # support / R

def zeropower_newton_schulz5(g:torch.Tensor,steps:int=5,eps:float=1e-7)->torch.Tensor:
    """Approximate U V^T for a 2-D matrix G=U S V^T using Muon-style quintic NS iteration."""
    if g.ndim!=2: raise ValueError("Newton-Schulz zeropower expects a matrix")
    if g.numel()==0: return torch.zeros_like(g)
    orig_dtype=g.dtype; x=g.float()
    transposed=x.shape[0]>x.shape[1]
    if transposed: x=x.mT
    x=x/(x.norm()+eps)
    a,b,c=3.4445,-4.7750,2.0315
    for _ in range(int(steps)):
        A=x@x.mT; B=b*A+c*(A@A); x=a*x+B@x
    if transposed: x=x.mT
    return x.to(orig_dtype)

def zeropower_svd(g:torch.Tensor)->torch.Tensor:
    if g.ndim!=2: raise ValueError("SVD zeropower expects a matrix")
    if not torch.count_nonzero(g): return torch.zeros_like(g)
    U,_,Vh=torch.linalg.svd(g.float(),full_matrices=False)
    return (U@Vh).to(g.dtype)

def _matrix_view(g:torch.Tensor):
    if g.ndim<2: raise ValueError("not a matrix-shaped parameter")
    return g.reshape(g.shape[0],-1),g.shape

def _spectral_direction(g:torch.Tensor,radius:float,backend:str,steps:int,width_scaling:str)->torch.Tensor:
    mat,shape=_matrix_view(g)
    if torch.count_nonzero(mat)==0: return torch.zeros_like(g)
    z=zeropower_svd(mat) if backend=="svd" else zeropower_newton_schulz5(mat,steps)
    scale=1.0
    if width_scaling=="scion":
        # Scion normalized spectral radius. For conv kernels Scion includes the
        # spatial kernel factor in addition to sqrt(out_channels/in_channels).
        if g.ndim==4:
            out_ch,in_ch,kh,kw=g.shape
            scale=math.sqrt(max(out_ch,1)/max(in_ch,1))/max(kh*kw,1)
        elif g.ndim==3:
            out_ch,in_ch,k=g.shape
            scale=math.sqrt(max(out_ch,1)/max(in_ch,1))/max(k,1)
        else:
            scale=math.sqrt(max(mat.shape[0],1)/max(mat.shape[1],1))
    elif width_scaling not in ("none",None): raise ValueError(f"unknown width_scaling={width_scaling}")
    return (-float(radius)*scale*z).reshape(shape)

def build_lmo_direction(blended:GradList, params:Sequence[NamedParam], *, geometry:str, radius:float,
                        spectral_backend:str="newton_schulz", ns_steps:int=5, width_scaling:str="none",
                        output_parameter_names:set[str]|None=None)->LMOInfo:
    geometry=geometry.lower(); R=float(radius)
    if R<=0: raise ValueError("LMO radius must be positive")
    if len(blended)!=len(params): raise ValueError("gradient/parameter length mismatch")
    output_parameter_names=output_parameter_names or set()
    if geometry=="l2":
        norm=torch.sqrt(sum((g.float()**2).sum() for g in blended)).to(blended[0].dtype)
        if float(norm)<=1e-20: direction=[torch.zeros_like(g) for g in blended]
        else: direction=[-R*g/norm for g in blended]
    elif geometry=="sign":
        direction=[-R*torch.sign(g) for g in blended]
    elif geometry in ("spectral","mixed"):
        direction=[]
        for g,(name,_) in zip(blended,params):
            is_output=name in output_parameter_names or name.startswith("pred_") or ".pred_" in name
            if g.ndim>=2 and not (geometry=="mixed" and is_output):
                direction.append(_spectral_direction(g,R,spectral_backend,ns_steps,width_scaling))
            else:
                # Sign is the robust vector/output fallback and matches the mixed Sign→Spectral→Sign configuration.
                direction.append(-R*torch.sign(g))
    else: raise ValueError(f"unknown LMO geometry: {geometry}")
    support=-inner_lists(blended,direction)
    return LMOInfo(direction=direction,support=support,normalized_dual_proxy=support/R)
