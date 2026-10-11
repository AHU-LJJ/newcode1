import argparse
import asyncio
import json
from pathlib import Path
from .adapters import load_cases,partition,describe
from .config import METHODS
from .evaluation import frontier
from .runner import atomic_json,run


def parser():
    p=argparse.ArgumentParser(description='SPM API-only situated preference memory')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('inspect','run'):
        q=sub.add_parser(name)
        q.add_argument('--datasets',default='datasets')
        q.add_argument('--benchmark',choices=['all','prefeval','perma','cupid','benchpres'],default='all')
        q.add_argument('--split',choices=['dev','test','all'],default='dev')
        q.add_argument('--seed',type=int,default=42)
        q.add_argument('--perma-mode',choices=['sd_clean','sd_noise','md_clean','md_noise'],default='sd_clean')
        q.add_argument('--perma-types',type=int,nargs='+',choices=[1,2,3],default=[1,2,3])
    q=sub.choices['run']
    q.add_argument('--methods',nargs='+',choices=METHODS,default=['v6_compact'])
    q.add_argument('--limit',type=int,default=30,help='Cases per benchmark; stratified round-robin. 0 means all.')
    q.add_argument('--perma-limit',type=int,help='Optional separate PERMA case limit; 0 means all.')
    q.add_argument('--max-history-episodes',type=int,help='Resource-limited subset selected using public history length BEFORE sampling.')
    q.add_argument('--output',default='runs/spm/pilot')
    q.add_argument('--cache',default='runs/spm/api_cache.sqlite')
    q.add_argument('--case-file',help='JSON list of exact case IDs; all IDs must survive split/resource filters and sampling.')
    q.add_argument('--env',default='.env')
    q.add_argument('--workers',type=int,default=8)
    q.add_argument('--concurrency',type=int,default=16)
    q.add_argument('--max-calls',type=int,default=2000)
    q.add_argument('--max-tokens',type=int,default=5000000)
    q.add_argument('--timeout',type=float,default=180)
    q.add_argument('--model-tpm',action='append',default=[],metavar='MODEL=TPM',
                   help='Per-model token rate shared across roles, with 10%% headroom.')
    q.add_argument('--tpm-tokenizer',help='Local tokenizer.json for ChatML token counting; otherwise conservative UTF-8 byte count.')
    q.add_argument('--apply-threshold',type=float,default=.5)
    q.add_argument('--suppress-threshold',type=float,default=.2)
    q.add_argument('--render-suppressed',action='store_true')
    q.add_argument('--max-context-chars',type=int,default=300000)
    q=sub.add_parser('frontier',help='Offline BenchPreS threshold sweep; no API calls')
    q.add_argument('run_directory');q.add_argument('--thresholds',type=float,nargs='+')
    return p


def main():
    args=parser().parse_args()
    if args.command=='inspect':
        names=['perma','cupid','benchpres'] if args.benchmark=='all' else [args.benchmark]
        print(json.dumps({name:dict(local=describe(c:=load_cases(args.datasets,name,args.perma_mode,args.perma_types)),
                                   selected_split=describe(partition(c,args.split,args.seed))) for name in names},ensure_ascii=False,indent=2))
        return 0
    if args.command=='frontier':
        directory=Path(args.run_directory)
        rows=[json.loads(p.read_text()) for p in sorted((directory/'samples').glob('*.json'))]
        result=frontier(rows,args.thresholds)
        atomic_json(directory/'frontier.json',result)
        print(json.dumps(result,indent=2));return 0
    if args.limit<0 or min(args.workers,args.concurrency,args.max_calls,args.max_tokens)<=0:
        raise ValueError('Workers, concurrency and budgets must be positive; limit must be nonnegative')
    if len(set(args.methods))!=len(args.methods):raise ValueError('Duplicate methods')
    if args.perma_limit is not None and args.perma_limit<0:raise ValueError('PERMA limit must be nonnegative')
    if args.max_history_episodes is not None and args.max_history_episodes<0:raise ValueError('History episode cap must be nonnegative')
    args.limit=args.limit or None
    return asyncio.run(run(args))


if __name__=='__main__':
    raise SystemExit(main())
