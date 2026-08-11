from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.runner import run_experiment

def main():
    parser=argparse.ArgumentParser(description="Run the main LMO-MOO experiments.")
    parser.add_argument("--problem",choices=("main_comparison","wall_clock"),required=True)
    parser.add_argument("--config",type=Path,default=None,help="defaults to configs/<problem>.yaml")
    parser.add_argument("--output",type=Path,default=None,help="override config output_dir")
    parser.add_argument("--methods",type=str,default=None,help="comma-separated method names")
    parser.add_argument("--dataset",choices=("cityscapes","nyuv2"),default=None)
    parser.add_argument("--reset-cache",action="store_true")
    parser.add_argument("--plot-only",action="store_true")
    args=parser.parse_args(); cfg=args.config or Path(__file__).resolve().parent/"configs"/f"{args.problem}.yaml"
    methods=None if args.methods is None else [x.strip() for x in args.methods.split(",") if x.strip()]
    run_experiment(args.problem,cfg,args.output,reset_cache=args.reset_cache,plot_only=args.plot_only,methods_override=methods,dataset_override=args.dataset)
if __name__=="__main__": main()
