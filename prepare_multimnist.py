"""Generate two-digit data with the source linked by MOON, or import its release.

The paper does not identify its generated dataset/seed/hash. Newly generated
images follow the linked source, but are NOT claimed identical to paper data.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import numpy as np
from PIL import Image


def reference_merge():
    from experiments.multimnist.generation import merge
    return merge


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('datasets/multimnist'))
    p.add_argument('--from-directory',type=Path,help='Import upstream generated release; retain only two-digit CSV rows')
    p.add_argument('--mnist-root',type=Path,default=Path('datasets/mnist'))
    p.add_argument('--train-samples',type=int,default=10000,help='Linked generator default for the 2-digit class')
    p.add_argument('--test-samples',type=int,default=1000)
    p.add_argument('--seed',type=int,default=2026,help='Dataset generation seed, fixed across training seeds')
    p.add_argument('--download',action='store_true')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError('Use a new empty output directory; existing datasets are never overwritten')
    if min(a.train_samples,a.test_samples)<=0: raise ValueError('Sample counts must be positive')
    random.seed(a.seed); merge=reference_merge(); counts={}; counter=0
    for split,n in [('train',a.train_samples),('test',a.test_samples)]:
        folder=a.output/split/'2'; folder.mkdir(parents=True,exist_ok=True); count=0
        with (folder.parent/'labels.csv').open('w',newline='') as f:
            writer=csv.writer(f)
            if a.from_directory:
                for row in csv.reader((a.from_directory/split/'labels.csv').open()):
                    if not row: continue
                    name,label=row[0].strip(),row[1].strip()
                    if len(label)!=2 or not label.isdigit(): continue
                    if Path(name).name!=name: raise ValueError('Image names must be filenames')
                    shutil.copy2(a.from_directory/split/'2'/name,folder/name)
                    writer.writerow([name,label]); count+=1
            else:
                from torchvision.datasets import MNIST
                mnist=MNIST(str(a.mnist_root),train=split=='train',download=a.download)
                images=mnist.data.numpy(); labels=mnist.targets.numpy(); indices=list(range(len(images)))
                for _ in range(n):
                    chosen=random.choices(indices,k=2)
                    arr=merge([images[i] for i in chosen],True)
                    arr=np.concatenate([np.zeros((28,2)),arr,np.zeros((28,2))],axis=1)
                    name=f'{counter}.png'; counter+=1
                    Image.fromarray(arr.astype(np.uint8)).save(folder/name)
                    writer.writerow([name,''.join(str(labels[i]) for i in chosen)]); count+=1
            if count==0: raise RuntimeError('No two-digit rows in '+split)
        counts[split]=count
    metadata={'source':'https://github.com/vndee/multi-mnist','counts':counts,'seed':a.seed,
              'source_directory':str(a.from_directory) if a.from_directory else None,
              'paper_identical_data':False,'note':'MOON does not publish generated dataset hash or generation seed.'}
    (a.output/'generation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__': main()
