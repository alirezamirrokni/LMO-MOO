from __future__ import annotations
from typing import Iterator
import torch
from torch import nn
import torch.nn.functional as F


def _c(v:int,width:float)->int:
    return max(4,int(round(v*width)))


def _unit(cin:int,cout:int)->nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin,cout,kernel_size=3,padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class MultiTaskSegNet(nn.Module):
    """SegNet hard-sharing backbone matching the MTAN/CAGrad/FAMO dense benchmark.

    The standard benchmark has stage depths [2,2,3,3,3] in the encoder and the
    corresponding decoder construction. ``width`` scales channel counts only;
    width=1.0 is the benchmark network and is used for Experiments 1--5.
    """
    def __init__(self,dataset:str,width:float=1.0):
        super().__init__()
        if dataset not in ("cityscapes","nyuv2"):
            raise ValueError(f"unknown dataset: {dataset}")
        self.dataset=dataset
        self.tasks=("semantic","depth") if dataset=="cityscapes" else ("semantic","depth","normal")
        classes=7 if dataset=="cityscapes" else 13
        filt=[_c(v,width) for v in (64,128,256,512,512)]

        # This mirrors SegNetSplit in the public dense-prediction benchmark code.
        self.encoder_block=nn.ModuleList([_unit(3,filt[0])])
        self.decoder_block=nn.ModuleList([_unit(filt[0],filt[0])])
        for i in range(4):
            self.encoder_block.append(_unit(filt[i],filt[i+1]))
            self.decoder_block.append(_unit(filt[i+1],filt[i]))

        self.conv_block_enc=nn.ModuleList([_unit(filt[0],filt[0])])
        self.conv_block_dec=nn.ModuleList([_unit(filt[0],filt[0])])
        for i in range(4):
            if i==0:
                self.conv_block_enc.append(_unit(filt[i+1],filt[i+1]))
                self.conv_block_dec.append(_unit(filt[i],filt[i]))
            else:
                self.conv_block_enc.append(nn.Sequential(_unit(filt[i+1],filt[i+1]),_unit(filt[i+1],filt[i+1])))
                self.conv_block_dec.append(nn.Sequential(_unit(filt[i],filt[i]),_unit(filt[i],filt[i])))

        def head(out_channels:int):
            return nn.Sequential(
                nn.Conv2d(filt[0],filt[0],kernel_size=3,padding=1),
                nn.Conv2d(filt[0],out_channels,kernel_size=1,padding=0),
            )
        self.pred_semantic=head(classes)
        self.pred_depth=head(1)
        if "normal" in self.tasks:
            self.pred_normal=head(3)

        self.pool=nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
        self.unpool=nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.reset_parameters()
        self.output_parameter_names={n for n,_ in self.named_parameters() if n.startswith("pred_")}

    def reset_parameters(self):
        # Public benchmark initialization.
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                if m.weight is not None: nn.init.constant_(m.weight,1)
                if m.bias is not None: nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias,0)

    def forward(self,x,return_representation:bool=False):
        g_maxpool=[]; indices=[]; sizes=[]
        h=x
        for i in range(5):
            h=self.encoder_block[i](h)
            h=self.conv_block_enc[i](h)
            sizes.append(h.size())
            h,idx=self.pool(h)
            g_maxpool.append(h); indices.append(idx)
        for i in range(5):
            src=g_maxpool[-1] if i==0 else h
            stage=4-i
            h=self.unpool(src,indices[stage],output_size=sizes[stage])
            h=self.decoder_block[stage](h)
            h=self.conv_block_dec[stage](h)
        rep=h
        out={
            "semantic":F.log_softmax(self.pred_semantic(rep),dim=1),
            "depth":self.pred_depth(rep),
        }
        if "normal" in self.tasks:
            out["normal"]=F.normalize(self.pred_normal(rep),p=2,dim=1,eps=1e-8)
        return (out,rep) if return_representation else out

    def shared_parameters(self)->Iterator[nn.Parameter]:
        return (p for n,p in self.named_parameters() if not n.startswith("pred_"))
    def task_specific_parameters(self)->Iterator[nn.Parameter]:
        return (p for n,p in self.named_parameters() if n.startswith("pred_"))
    def last_shared_parameters(self)->Iterator[nn.Parameter]:
        # Same conceptual last shared decoder block used by the benchmark wrappers.
        return self.conv_block_dec[0].parameters()
