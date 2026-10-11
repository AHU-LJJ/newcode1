"""A small real-API integration demo. Run from the CAPS project root."""
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from spm import SPM,Episode,Message
from spm.api import Gateway
from spm.config import endpoints
from spm.models import fingerprint
from spm.store import Store


async def main():
    root=Path('runs/spm/demo');root.mkdir(parents=True,exist_ok=True)
    api=Gateway(endpoints(),Path('runs/spm/api_cache.sqlite'),concurrency=4,max_calls=80,max_tokens=200000,timeout=180)
    store=Store(root/'memory.sqlite','demo-user')
    memory=SPM('demo-user',store,api)
    sessions=[
        'I cannot eat gluten: my doctor diagnosed me with coeliac disease.',
        'When writing to Acme Leasing I prefer a formal, concise tone. For my friend Mia I prefer playful messages.',
    ]
    traces=[]
    try:
        for i,text in enumerate(sessions):
            await memory.ingest(Episode(id=fingerprint([i,text])[:24],order=i,
                messages=[Message(role='user',content=text)]))
        for query in ['Recommend restaurants for my trip to Rome.',
                      'Draft a complaint email to Acme Leasing about a broken heater.']:
            context,trace=await memory.query(query)
            traces.append(trace)
            print(json.dumps(dict(request=query,memory_context=context),ensure_ascii=False,indent=2))
        # Inspect a physiological restriction relevant across locations and suppressed for writing.
        health={u.id for u in store.all_units() if any(w in u.content.casefold() for w in ('gluten','coeliac','celiac'))}
        if not health:raise RuntimeError('Demo failed: no grounded coeliac/gluten constraint was extracted')
        if not any(d['unit_id'] in health and d['decision']=='apply' for d in traces[0]['decisions']):
            raise RuntimeError('Demo failed: relevant dietary constraint was not applied for restaurants')
        if any(d['unit_id'] in health and d['decision']=='apply' for d in traces[1]['decisions']):
            raise RuntimeError('Demo failed: dietary constraint leaked into unrelated complaint writing')
        (root/'traces.json').write_text(json.dumps(traces,ensure_ascii=False,indent=2)+'\n')
        (root/'snapshot.json').write_text(json.dumps(store.snapshot(),ensure_ascii=False,indent=2)+'\n')
        print('Context generalization/suppression checks passed.')
    finally:
        store.close()
        usage=await api.close()
        (root/'usage.json').write_text(json.dumps(usage,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':asyncio.run(main())
