from __future__ import annotations
import fnmatch, os, random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

TASKS={"cityscapes":("semantic","depth"),"nyuv2":("semantic","depth","normal")}
CLASSES={"cityscapes":7,"nyuv2":13}

class DenseNPYDataset(Dataset):
    """NYUv2/CityScapes loader compatible with the MTAN/FAMO prepared .npy layout.

    Expected layout: NYUv2 uses ROOT/{train,val}/{image,label,depth,normal}/<index>.npy;
    the public CityScapes preparation uses ROOT/{train,val}/{image,label_7,depth}/<index>.npy.
    A CityScapes ``label`` directory is also accepted for compatibility. Images/normal maps may
    be HWC or CHW; semantic labels are HW.
    """
    def __init__(self, root: str|Path, dataset: str, train: bool=True, augmentation: bool=False):
        dataset=dataset.lower()
        if dataset not in TASKS: raise ValueError(f"unknown dense dataset: {dataset}")
        self.dataset=dataset; self.tasks=TASKS[dataset]; self.augmentation=bool(augmentation and train)
        split="train" if train else "val"
        self.data_path=Path(root).expanduser()/split
        if not self.data_path.exists() and not train and (Path(root).expanduser()/"test").exists():
            self.data_path=Path(root).expanduser()/"test"
        image_dir=self.data_path/"image"
        if not image_dir.is_dir(): raise FileNotFoundError(f"missing prepared image directory: {image_dir}")
        self.files=sorted(image_dir.glob("*.npy"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        if not self.files: raise FileNotFoundError(f"no .npy images found in {image_dir}")
        # The public CityScapes prepared split calls its semantic directory ``label_7``;
        # NYUv2 uses ``label``. Accept both to remain compatible with existing copies.
        label_candidates=("label_7","label") if self.dataset=="cityscapes" else ("label",)
        self.label_dir=next((x for x in label_candidates if (self.data_path/x).is_dir()),None)
        if self.label_dir is None:
            raise FileNotFoundError(f"missing semantic-label directory under {self.data_path}; tried {label_candidates}")
        for sub in ("depth",) + (("normal",) if "normal" in self.tasks else ()):
            if not (self.data_path/sub).is_dir(): raise FileNotFoundError(f"missing directory: {self.data_path/sub}")
    def __len__(self): return len(self.files)
    @staticmethod
    def _chw(arr: np.ndarray) -> np.ndarray:
        if arr.ndim==2: return arr[None,...]
        if arr.ndim==3 and arr.shape[0] not in (1,3) and arr.shape[-1] in (1,3): return np.moveaxis(arr,-1,0)
        return arr
    def __getitem__(self,index:int)->dict[str,torch.Tensor]:
        stem=self.files[index].stem
        image=self._chw(np.load(self.data_path/"image"/f"{stem}.npy")).astype(np.float32)
        label=np.load(self.data_path/self.label_dir/f"{stem}.npy").astype(np.int64)
        if label.ndim==3: label=np.squeeze(label)
        depth=self._chw(np.load(self.data_path/"depth"/f"{stem}.npy")).astype(np.float32)
        out={"image":torch.from_numpy(image),"semantic":torch.from_numpy(label),"depth":torch.from_numpy(depth)}
        if "normal" in self.tasks:
            normal=self._chw(np.load(self.data_path/"normal"/f"{stem}.npy")).astype(np.float32)
            out["normal"]=torch.from_numpy(normal)
        if self.augmentation: out=self._augment(out)
        return out
    def _augment(self,b:dict[str,torch.Tensor])->dict[str,torch.Tensor]:
        # Public FAMO/CAGrad NYUv2 order: random scale/crop first, then horizontal flip.
        if self.dataset=="nyuv2":
            scale=random.choice((1.0,1.2,1.5))
            if scale!=1.0:
                h,w=b["image"].shape[-2:]; nh,nw=max(1,int(h/scale)),max(1,int(w/scale))
                i=random.randint(0,h-nh); j=random.randint(0,w-nw)
                def crop(x): return x[...,i:i+nh,j:j+nw]
                b["image"]=F.interpolate(crop(b["image"])[None],size=(h,w),mode="bilinear",align_corners=True)[0]
                b["semantic"]=F.interpolate(crop(b["semantic"])[None,None].float(),size=(h,w),mode="nearest")[0,0].long()
                b["depth"]=F.interpolate(crop(b["depth"])[None],size=(h,w),mode="nearest")[0]/scale
                if "normal" in b: b["normal"]=F.interpolate(crop(b["normal"])[None],size=(h,w),mode="bilinear",align_corners=True)[0]
        if random.random()<0.5:
            for k in list(b): b[k]=torch.flip(b[k],dims=[-1])
            if "normal" in b: b["normal"][0].mul_(-1.0)
        return b

class SyntheticDenseDataset(Dataset):
    """Small deterministic dataset used only for tests/smoke runs."""
    def __init__(self,dataset:str,n:int=8,h:int=32,w:int=32,seed:int=0):
        self.dataset=dataset; self.tasks=TASKS[dataset]; self.n=n; self.h=h; self.w=w; self.seed=seed
    def __len__(self): return self.n
    def __getitem__(self,index:int):
        g=torch.Generator().manual_seed(self.seed+index)
        image=torch.randn(3,self.h,self.w,generator=g)
        semantic=torch.randint(0,CLASSES[self.dataset],(self.h,self.w),generator=g)
        depth=torch.rand(1,self.h,self.w,generator=g)+0.1
        out={"image":image,"semantic":semantic,"depth":depth}
        if "normal" in self.tasks:
            normal=torch.randn(3,self.h,self.w,generator=g); normal=F.normalize(normal,dim=0)
            out["normal"]=normal
        return out

def build_loaders(config:dict[str,Any],dataset:str,seed:int):
    dcfg=config["data"][dataset]; root=dcfg.get("root","")
    bs=int(dcfg.get("batch_size",8 if dataset=="cityscapes" else 2))
    workers=int(dcfg.get("workers",0)); pin=bool(dcfg.get("pin_memory",True))
    if str(root)=="__synthetic__":
        train=SyntheticDenseDataset(dataset,int(dcfg.get("synthetic_train",8)),int(dcfg.get("synthetic_h",32)),int(dcfg.get("synthetic_w",32)),seed)
        val=SyntheticDenseDataset(dataset,int(dcfg.get("synthetic_val",4)),int(dcfg.get("synthetic_h",32)),int(dcfg.get("synthetic_w",32)),seed+10000)
    else:
        train=DenseNPYDataset(root,dataset,True,bool(dcfg.get("augmentation",False)))
        val=DenseNPYDataset(root,dataset,False,False)
    gen=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train,batch_size=bs,shuffle=True,num_workers=workers,pin_memory=pin,generator=gen,drop_last=False)
    val_loader=DataLoader(val,batch_size=bs,shuffle=False,num_workers=workers,pin_memory=pin,drop_last=False)
    return train_loader,val_loader
