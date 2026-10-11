"""Resume original M1 results with their frozen code, preserving successful samples."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import os
from datetime import datetime, timezone


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['dev','test'],required=True)
    parser.add_argument('--max-calls',type=int,default=1200)
    parser.add_argument('--max-tokens',type=int,default=2000000)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--concurrency',type=int,default=16)
    parser.add_argument('--timeout',type=int,default=None,
                        help='Operational HTTP timeout override; does not change the frozen protocol')
    parser.add_argument('--audit-transport',action='store_true',
                        help='Record judge response structure without changing requests or grading')
    parser.add_argument('--recovery-output',help='Separate output directory for a consistently rejudged recovery track')
    parser.add_argument('--recovery-judge',help='Judge model override; requires a separate recovery output')
    parser.add_argument('--check-only',action='store_true',help='Verify frozen code and count pending results without any API calls')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if Path.cwd()!=root:parser.error('Run from the CAPS project root')
    plan=json.loads((root/'runs/spm/m1_plan.json').read_text())
    frozen=root/'runs/spm/m1_frozen'
    for name,expected in plan['code'].items():
        if hashlib.sha256((frozen/'spm'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen M1 source was modified')
    argv=plan['phases'][args.phase]['argv'][:]
    if bool(args.recovery_output)!=bool(args.recovery_judge):
        parser.error('--recovery-output and --recovery-judge must be supplied together')
    if args.recovery_output:
        if (root/args.recovery_output).resolve()==(root/argv[argv.index('--output')+1]).resolve():
            parser.error('Recovery grades must never overwrite the original M1 directory')
        argv[argv.index('--output')+1]=args.recovery_output
        os.environ['JUDGE_MODEL']=args.recovery_judge
    if args.check_only:
        output=root/argv[argv.index('--output')+1]
        manifest=json.loads((output/'manifest.json').read_text())
        if manifest['scientific']['code']!=plan['code']:raise ValueError('M1 manifest does not match frozen source')
        rows=[json.loads(p.read_text()) for p in (output/'samples').glob('*.json')]
        print(json.dumps(dict(phase=args.phase,source_verified=True,samples=len(rows),
                              ok=sum(r['status']=='ok' for r in rows),
                              pending=sum(r['status']!='ok' for r in rows),network_requests=0),indent=2))
        return 0
    for flag,value in [('max-calls',args.max_calls),('max-tokens',args.max_tokens),
                       ('workers',args.workers),('concurrency',args.concurrency)]:
        argv[argv.index('--'+flag)+1]=str(value)
    if args.timeout is not None:
        if args.timeout<=0:parser.error('--timeout must be positive')
        if '--timeout' in argv:argv[argv.index('--timeout')+1]=str(args.timeout)
        else:argv.extend(['--timeout',str(args.timeout)])
    sys.path.insert(0,str(frozen))
    from spm.__main__ import main as frozen_main
    if args.audit_transport:
        from spm.api import Gateway
        original_request=Gateway._request
        audit=root/argv[argv.index('--output')+1]/'transport_audit.jsonl'
        wrapper_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        async def traced_request(gateway,role,path,payload,tag):
            data=await original_request(gateway,role,path,payload,tag)
            if role=='judge':
                choice=(data.get('choices') or [{}])[0]
                message=choice.get('message',{})
                content=message.get('content')
                record=dict(run_id=gateway.run,tag=tag,wrapper_sha256=wrapper_hash,
                    timestamp=datetime.now(timezone.utc).isoformat(),model=data.get('model'),
                    requested_max_tokens=payload.get('max_tokens'),enable_thinking=payload.get('enable_thinking'),
                    finish_reason=choice.get('finish_reason'),usage=data.get('usage'),
                    content_type=type(content).__name__,content_chars=len(content) if isinstance(content,str) else None,
                    content_prefix=content[:200] if isinstance(content,str) else None,
                    content_suffix=content[-200:] if isinstance(content,str) else None,
                    reasoning_chars=len(message.get('reasoning_content') or ''))
                with audit.open('a') as handle:handle.write(json.dumps(record,ensure_ascii=False)+'\n')
            return data
        Gateway._request=traced_request
    sys.argv=[sys.argv[0],*argv]
    return frozen_main()


if __name__=='__main__':
    raise SystemExit(main())
