"""MultiMNIST/ViT: reference MOON, six table baselines, and paper LMO-MOO."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from experiments.checkpointing import atomic_torch_save, load_torch, capture_rng_state, restore_rng_state
from experiments.utils import set_seed
from experiments.multimnist.models_vit import MultiMNISTViT
from experiments.multimnist.data import MultiMNISTDataset
from methods.moon import ReferenceMOON, make_reference_optimizer, previous_update_scores
from methods.moon_reference_weights import MGDA, FAMO, LinearScalarization
from methods.paper_ablation import PaperAblation

METHODS = ['ours','mgda','mgda_muon','famo','famo_muon','muon_ls','moon']
CONFIG = json.loads(Path(__file__).with_name('configs.json').read_text())


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', type=Path, default=Path('datasets/multimnist'))
    p.add_argument('--output-root', type=Path, default=Path('results/multimnist'))
    p.add_argument('--method', type=lambda x: {'entropic-lmo-mgda':'ours','entropic_lmo_mgda':'ours'}.get(x,x), choices=METHODS, default='ours')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=CONFIG['common']['epochs'])
    p.add_argument('--batch-size', type=int, default=CONFIG['common']['batch_size'])
    p.add_argument('--lr', type=float, default=CONFIG['common']['lr'])
    p.add_argument('--weight-lr', '--method-params-lr', type=float)
    p.add_argument('--gamma', type=float)
    p.add_argument('--eta', type=float, default=CONFIG['ours']['eta'])
    p.add_argument('--alpha', type=float, default=CONFIG['ours']['alpha'])
    p.add_argument('--oracle', choices=['spectral','l2','sign'], default='spectral')
    p.add_argument('--weights', choices=['entropic','projected'], default='entropic')
    p.add_argument('--momentum', choices=['blended','per-task','none'], default='blended')
    p.add_argument('--ns-steps', type=int, default=5)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--tag', default='main')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Two batches on synthetic data, one epoch; NEVER a paper result.')
    p.add_argument('--stop-after-epoch', type=int, help='Graceful stop for interruption/resume testing; total epochs unchanged.')
    p.add_argument('--metric', choices=['upstream','sample'], default='upstream', help='Upstream averages batch accuracies equally. Both statistics are always saved.')
    return p


def make_method(model, args, device):
    if args.method == 'ours':
        method = PaperAblation(2, device, eta=args.eta, alpha=args.alpha, ns_steps=args.ns_steps,
                              entropy_tau=0, weight_kappa=0, oracle=args.oracle,
                              weight_update=args.weights, momentum=args.momentum, log_every=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        if args.method == 'moon':
            method = ReferenceMOON(2,device,w_lr=args.weight_lr,gamma=args.gamma)
        elif args.method.startswith('famo'):
            method = FAMO(2,device,w_lr=args.weight_lr,gamma=args.gamma)
        elif args.method.startswith('mgda'):
            method = MGDA(2,device)
        else:
            method = LinearScalarization(2,device)
        if args.method in {'moon','mgda_muon','famo_muon','muon_ls'}:
            optimizer = make_reference_optimizer(model,'multimnist',args.lr)
        else:
            optimizer = torch.optim.Adam([dict(params=model.parameters(),lr=args.lr),
                                          dict(params=method.parameters(),lr=args.weight_lr)])
    return method, optimizer


def method_state(method):
    if hasattr(method,'state_dict'):
        return method.state_dict()
    if isinstance(method,FAMO):
        return dict(w=method.w.detach().cpu(),min_losses=method.min_losses.cpu(),w_opt=method.w_opt.state_dict())
    return {}


def load_method_state(method,state):
    if hasattr(method,'load_state_dict'):
        method.load_state_dict(state)
    elif isinstance(method,FAMO):
        with torch.no_grad():
            method.w.copy_(state['w'].to(method.device))
            method.min_losses.copy_(state['min_losses'].to(method.device))
        method.w_opt.load_state_dict(state['w_opt'])


def loss_vector(pred, y0,y1):
    return torch.stack([F.cross_entropy(pred[0],y0),F.cross_entropy(pred[1],y1)])


def train_step(model, method, optimizer, batch, method_name):
    model.train()
    optimizer.zero_grad()
    x,y0,y1 = batch
    pred,features = model(x,return_representation=True)
    losses = loss_vector(pred,y0,y1)
    if method_name == 'moon':
        method.update(previous_update_scores(losses,list(model.parameters()),optimizer))
    method.backward(losses, shared_parameters=list(model.shared_parameters()),
                    task_specific_parameters=list(model.task_specific_parameters()),
                    representation=features, model_parameters=list(model.parameters()))
    optimizer.step()
    if method_name.startswith('famo'):
        with torch.no_grad():
            method.update(loss_vector(model(x),y0,y1).detach())
    if not torch.isfinite(losses).all():
        raise FloatingPointError('Nonfinite loss')
    acc = [float((p.argmax(1)==y).float().mean()) for p,y in zip(pred,(y0,y1))]
    return [float(losses[0].detach()), acc[0], float(losses[1].detach()), acc[1]]


def evaluate(model,loader,device):
    model.eval()
    sums=np.zeros(4); batch_means=np.zeros(4); count=0
    with torch.no_grad():
        for batch in loader:
            x,y0,y1=(v.to(device) for v in batch)
            pred=model(x); losses=loss_vector(pred,y0,y1)
            row=np.array([float(losses[0]),float((pred[0].argmax(1)==y0).float().mean()),
                          float(losses[1]),float((pred[1].argmax(1)==y1).float().mean())])
            sums+=row*len(x); batch_means+=row; count+=len(x)
    return batch_means/len(loader),sums/count


def data_fingerprint(root):
    h=hashlib.sha256()
    # Pin the labels and every referenced image, including preprocessing version.
    for split in ['train','test']:
        path=root/split/'labels.csv'; h.update(split.encode()); h.update(path.read_bytes())
        for row in csv.reader(path.open()):
            if row:
                image=root/split/'2'/row[0].strip()
                h.update(image.read_bytes())
    return h.hexdigest()


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(path)


def run(args):
    if args.epochs<=0 or args.batch_size<=0 or args.lr<=0:
        raise ValueError('epochs, batch size, and lr must be positive')
    if args.weight_lr is None:
        args.weight_lr = CONFIG['moon' if args.method=='moon' else 'famo']['weight_lr']
    if args.gamma is None:
        args.gamma=CONFIG['moon' if args.method=='moon' else 'famo']['gamma']
    if args.weight_lr<=0 or args.gamma<0:
        raise ValueError('weight-lr must be positive and gamma nonnegative')
    if args.smoke:
        args.epochs=1; args.batch_size=4
    if Path(args.tag).name!=args.tag or args.tag in {'.','..'}:
        raise ValueError('tag must be a simple directory name')
    torch.set_num_threads(args.threads)
    device=torch.device(args.device); set_seed(args.seed)
    model=MultiMNISTViT().to(device)
    method,optimizer=make_method(model,args,device)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.5)
    if args.smoke:
        gen=torch.Generator().manual_seed(999)
        train_set=torch.utils.data.TensorDataset(torch.rand(8,1,28,43,generator=gen),
                    torch.randint(10,(8,),generator=gen),torch.randint(10,(8,),generator=gen))
        test_set=train_set
        fingerprint='synthetic-smoke-only'
    else:
        train_set=MultiMNISTDataset(str(args.data_path),'train')
        test_set=MultiMNISTDataset(str(args.data_path),'test')
        fingerprint=data_fingerprint(args.data_path)
    # Same loader construction/RNG consumption as upstream. Restore global RNG at epoch boundaries.
    train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    test_loader=DataLoader(test_set,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    prefix='smoke' if args.smoke else args.tag
    out=args.output_root/prefix/args.method/f'seed{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    signature={k:v for k,v in config.items() if k not in {'resume','stop_after_epoch','output_root','data_path','device','workers','threads'}}
    signature.update(data_sha256=fingerprint,implementation='multimnist-v1',source_commit=CONFIG['source_commit'])
    ckpt=out/'checkpoint.pt'; start=0; rows=[]; elapsed=0.0
    if ckpt.exists():
        if not args.resume:
            raise FileExistsError(f'{ckpt} exists; pass --resume or choose another --output-root/--tag')
        state=load_torch(ckpt,'cpu')
        if state['signature']!=signature:
            raise ValueError('Checkpoint configuration or dataset differs')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        load_method_state(method,state['method']); scheduler.load_state_dict(state['scheduler'])
        rows=state['rows']; start=state['epoch']; elapsed=state['elapsed']
        restore_rng_state(state['rng'])
    write_json(out/'config.json',dict(config,signature=signature))
    for epoch in range(start,args.epochs):
        begin=time.monotonic(); train_cost=np.zeros(4)
        for batch in train_loader:
            batch=tuple(v.to(device) for v in batch)
            train_cost+=np.array(train_step(model,method,optimizer,batch,args.method))/len(train_loader)
        scheduler.step()
        upstream,sample=evaluate(model,test_loader,device)
        # Fixed training-only diagnostic at the FINAL model; does not update buffers or weights.
        gap=support=None
        if args.method=='ours':
            probe=torch.utils.data.default_collate([train_set[i] for i in range(min(256,len(train_set)))])
            x,y0,y1=(v.to(device) for v in probe)
            with torch.enable_grad():
                gap,support=method.inner_gap(loss_vector(model(x),y0,y1),list(model.parameters()))
        chosen=upstream if args.metric=='upstream' else sample
        elapsed+=time.monotonic()-begin
        row=dict(epoch=epoch+1,left=100*chosen[1],right=100*chosen[3],avg=50*(chosen[1]+chosen[3]),
                 gap=gap,support=support,train=train_cost.tolist(),test_upstream=upstream.tolist(),
                 test_sample=sample.tolist(),elapsed_seconds=elapsed)
        rows.append(row)
        atomic_torch_save(dict(signature=signature,epoch=epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(),method=method_state(method),rng=capture_rng_state(),rows=rows,elapsed=elapsed),ckpt)
        write_json(out/'history.json',rows)
        write_json(out/'summary.json',dict(row,completed=epoch+1==args.epochs,smoke=args.smoke,method=args.method,
                   seed=args.seed,tag=args.tag,signature=signature,metric=args.metric))
        print(json.dumps({k:row[k] for k in ['epoch','left','right','avg','gap']}),flush=True)
        if args.stop_after_epoch and epoch+1>=args.stop_after_epoch:
            break
    print(f'Outputs: {out}',flush=True)
    return out


def main():
    run(parser().parse_args())

if __name__=='__main__': main()
