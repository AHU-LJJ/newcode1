"""Operational extraction telemetry only: does not modify prompts, outputs or validation."""
import argparse
import json
from pathlib import Path
import spm_m2

spm_m2.frozen_import()
from spm.api import Gateway
original=Gateway._request

async def audited(self,role,path,payload,tag):
    data=await original(self,role,path,payload,tag)
    if tag=='extract':
        record=dict(tag=tag,wrapper_sha256=spm_m2.sha(__file__))
        try:
            source=json.loads(payload['messages'][1]['content'])['messages']
            obj=json.loads(data['choices'][0]['message']['content'])
            items=[]
            for unit in obj.get('units',[]):
                idx=unit.get('turn_idx');span=unit.get('span','')
                matches=[m['turn_idx'] for m in source if m['role']=='user' and isinstance(span,str) and span and span in m['content']]
                cited=next((m for m in source if m['turn_idx']==idx),None)
                if idx not in matches:
                    items.append(dict(turn_idx=idx,span=span,exact_matching_user_indices=matches,
                                      cited_turn=cited))
            record['ungrounded_units']=items
        except (ValueError,KeyError,TypeError,IndexError):record['unparseable']=True
        p=spm_m2.BASE/'m2_extract_diagnostic.jsonl'
        with p.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    return data

Gateway._request=audited
args=argparse.Namespace(phase='test',workers=4,concurrency=8,max_calls=100,max_tokens=300000,timeout=180)
raise SystemExit(spm_m2.run(args))
