"""Blind, additive CUPID response alignment evaluation of saved M8 answers."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT/'runs/spm'
OUT = BASE/'m8_alignment'
PLAN = BASE/'m8_alignment_plan.json'
PROMPT = ROOT/'paper/cupid_official/prompts/evaluation/response_judger.yaml'
METHODS = ['v6_m8', 'typed_m8', 'flat_m8', 'raw_m8', 'b1']
SEED = 20261017


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    from spm.config import endpoints
    from spm.models import fingerprint
    if PLAN.exists():
        raise ValueError('Refuse to overwrite alignment plan')
    source = read(BASE/'m8_test_plan.json')
    assert source['methods'] == ['flat_m8', 'v6_m8', 'typed_m8', 'raw_m8', 'b1']
    judge = endpoints()['judge']
    assert judge.model == 'Pro/deepseek-ai/DeepSeek-V3.2'
    selected = []
    for path in sorted((BASE/'m8_test/samples').glob('*.json')):
        row = read(path)
        assert row['benchmark'] == 'cupid' and row['method'] in METHODS
        if row['status'] != 'ok':
            continue
        answer = row['evaluation']['response']
        assert isinstance(answer, str) and answer.strip()
        selected.append(dict(case_id=row['case_id'], case_digest=row['case_digest'],
                             group=row['group'], stratum=row['stratum'], method=row['method'],
                             path=str(path.relative_to(ROOT)), sha256=sha(path),
                             response_sha256=fingerprint(answer)))
    assert len(selected) == 93
    protected = list(BASE.glob('*/samples/*.json'))
    for phase in ['dev', 'test']:
        protected += list((BASE/f'm8_{phase}/memory').rglob('*.json'))
    code = list((ROOT/'spm').glob('*.py')) + [Path(__file__)]
    upstream = read(ROOT/'paper/cupid_official/source.json')
    references = [ROOT/'paper/cupid_official'/p for p in upstream['files']]
    references += [ROOT/'paper/cupid_official/source.json', ROOT/'paper/CUPID.pdf', BASE/'m8_test_plan.json']
    write(PLAN, dict(protocol='CUPID official released response alignment prompt; additive M8 re-evaluation',
                     seed=SEED, judge=dict(model=judge.model, base_url=judge.base_url),
                     tests_before_freeze='14 alignment-specific tests passed',
                     official_revision=upstream['revision'], paper_sections=['2.3', 'A.2', 'Figure 16'],
                     methods=METHODS, rows=selected, sources=source['sources'],
                     protected={str(p.relative_to(ROOT)):sha(p) for p in protected},
                     code={str(p.relative_to(ROOT)):sha(p) for p in code},
                     references={str(p.relative_to(ROOT)):sha(p) for p in references},
                     controls='Judge sees only request, saved response, ground-truth preference/checklist; no method, memory, inferred checklist or prior scores.',
                     sampling='All 93 previously successful M8 main-test answers. Main comparison uses all five methods jointly valid; no cherry-picked scores.',
                     missing='27 original execution failures remain missing, never assigned zero.',
                     budget=dict(max_calls=220, max_tokens=2_000_000, concurrency=16, timeout=180,
                                 max_output=8192, judge_tpm=2_000_000),
                     resume='At most one failed-only resume; successful grade bytes protected. No answer regeneration or replacement of old metrics.',
                     differences='Judge substituted for paper GPT-4o. Released prompt adds rubric labels to Figure 16. M8 answers already use inferred preference before generation; not the paper generation-only pipeline.'))
    print(json.dumps(dict(prepared_answers=len(selected), protected_files=len(protected))), flush=True)


def checked_inputs():
    from spm.adapters import cupid
    from spm.cupid_alignment import load_prompt
    plan = read(PLAN)
    for section in ['protected', 'code', 'references', 'sources']:
        for name, digest in plan[section].items():
            assert sha(ROOT/name) == digest, (section, name)
    cases = {case.id:case for case in cupid(ROOT/'datasets')}
    for item in plan['rows']:
        assert cases[item['case_id']].digest() == item['case_digest']
    return plan, cases, load_prompt(PROMPT)


def grade_path(item):
    return OUT/'grades'/f"{item['method']}_{item['case_id'].replace('/', '_')}.json"


async def run(resume=False, replay=False):
    from spm.api import Gateway, REQUEST_CONTEXT
    from spm.config import endpoints
    from spm.cupid_alignment import judge_inputs, parse_score
    plan, cases, templates = checked_inputs()
    ep = endpoints()['judge']
    assert dict(model=ep.model, base_url=ep.base_url) == plan['judge']
    budget = plan['budget']
    api = Gateway({'judge':ep}, OUT/'api_cache.sqlite', concurrency=budget['concurrency'],
                  max_calls=budget['max_calls'], max_tokens=budget['max_tokens'],
                  timeout=budget['timeout'], max_output=budget['max_output'],
                  tpm_limits={ep.model:budget['judge_tpm']})
    if replay:
        async def forbidden(*args, **kwargs):
            raise AssertionError('Offline replay forbids HTTP')
        api._request = forbidden
    invocation_path = OUT/'invocations.json'
    invocations = read(invocation_path) if invocation_path.exists() else []
    if not replay:
        if invocations and not resume:
            raise ValueError('Use --resume for the single bounded failed-only retry')
        if len(invocations) >= 2:
            raise ValueError('One failed-only resume already used')
        invocations.append(dict(run_id=api.run, resume=resume))
        write(invocation_path, invocations)
    success_hashes = {str(p):sha(p) for p in (OUT/'grades').glob('*.json') if read(p)['status']=='ok'}
    completed = 0
    replayed = 0

    async def one(item):
        nonlocal completed, replayed
        path = grade_path(item)
        previous = read(path) if path.exists() else None
        if previous and previous['status'] == 'ok' and not replay:
            return
        if replay and (not previous or previous['status'] != 'ok'):
            return
        case = cases[item['case_id']]
        source = read(ROOT/item['path'])
        system, user = judge_inputs(templates, case.public.query, source['evaluation']['response'], case.gold)
        token = REQUEST_CONTEXT.set(dict(experiment='m8_alignment', case_id=item['case_id'], method=item['method']))
        raw = None
        try:
            raw = await api.ask('judge', system, user, tag='cupid_alignment_official_v1', max_tokens=8192)
            score = parse_score(raw)
            if replay:
                assert raw == previous['raw_judge'] and score == previous['preference_alignment_score']
                replayed += 1
                return
            write(path, dict(**item, status='ok', preference_alignment_score=score, raw_judge=raw,
                             judge_model=ep.model, protocol=plan['protocol']))
        except Exception as error:
            if replay:
                raise
            write(path, dict(**item, status='failed', error=dict(type=type(error).__name__, message=str(error)),
                             raw_judge=raw, judge_model=ep.model, protocol=plan['protocol']))
        finally:
            REQUEST_CONTEXT.reset(token)
            completed += 1
            if not replay:
                print(json.dumps(dict(completed=completed, case=item['case_id'], method=item['method'],
                                      status=read(path)['status'])), flush=True)
    try:
        await asyncio.gather(*(one(item) for item in plan['rows']))
    finally:
        usage = await api.close()
        if not replay:
            write(OUT/'usage'/f"{api.run}.json", usage)
    for name, digest in success_hashes.items():
        assert sha(name) == digest, name
    checked_inputs()
    if replay:
        assert usage['http_requests'] == 0
        write(OUT/'replay_verification.json', dict(replayed=replayed, http_requests=0,
                                                  successful_grade_bytes_unchanged=True,
                                                  original_artifacts_unchanged=True))
        print(json.dumps(dict(replayed=replayed, http_requests=0)), flush=True)


def summarize():
    import numpy as np
    plan, cases, _ = checked_inputs()
    rows = [read(grade_path(item)) for item in plan['rows']]
    good = {(r['case_id'],r['method']):r for r in rows if r['status']=='ok'}
    planned_ids = sorted({r['case_id'] for r in plan['rows']})
    common = [cid for cid in planned_ids if all((cid,m) in good for m in METHODS)]
    def interval(values):
        if len(values) < 2:
            return None
        rng = np.random.default_rng(SEED)
        draws = rng.choice(values, (2000,len(values)), replace=True).mean(axis=1)
        return np.quantile(draws, [.025,.975]).tolist()
    metrics = {}
    for method in METHODS:
        all_scores = [r['preference_alignment_score'] for r in rows if r['method']==method and r['status']=='ok']
        scores = [good[cid,method]['preference_alignment_score'] for cid in common]
        metrics[method] = dict(original_planned=24, saved_answers=sum(r['method']==method for r in rows),
                               graded=len(all_scores), judge_failed=sum(r['method']==method and r['status']!='ok' for r in rows),
                               mean_on_all_saved_answers=float(np.mean(all_scores)) if all_scores else None,
                               common_questions=len(scores), common_mean=float(np.mean(scores)) if scores else None,
                               common_95ci=interval(scores), distribution=dict(Counter(scores)),
                               strata={s:dict(n=sum(cases[c].stratum==s for c in common),
                                              mean=float(np.mean([good[c,method]['preference_alignment_score'] for c in common if cases[c].stratum==s]))
                                              if any(cases[c].stratum==s for c in common) else None)
                                       for s in ['changing','consistent','contrastive']})
    comparisons = {}
    for method in METHODS:
        if method == 'flat_m8':
            continue
        values = [good[c,method]['preference_alignment_score']-good[c,'flat_m8']['preference_alignment_score'] for c in common]
        comparisons[method+'_vs_flat_m8'] = dict(paired_questions=len(values), delta=float(np.mean(values)) if values else None,
                                                 paired_cluster_95ci=interval(values))
    usage = [read(p) for p in (OUT/'usage').glob('*.json')]
    totals = {key:sum(r[key] for r in usage) for key in ['http_requests','known_tokens','unknown_usage_requests']}
    summary = dict(eligible_answers=len(rows), graded=len(good), failed=len(rows)-len(good),
                   originally_planned=120, original_execution_missing=27, common_ids=common,
                   original_common_ids=read(BASE/'m8_test/common_success_metrics.json')['case_ids'],
                   metrics=metrics, comparisons_vs_flat=comparisons,
                   common_strata=dict(Counter(cases[c].stratum for c in common)), usage=totals,
                   original_artifacts_unchanged=True)
    write(OUT/'summary.json', summary)
    write(OUT/'pending.json', [r for r in rows if r['status']!='ok'])
    names={'v6_m8':'原图','typed_m8':'对象/维度图','flat_m8':'平铺记忆','raw_m8':'原文 RAG','b1':'完整历史'}
    lines=['# CUPID M8：Preference Alignment Score 补测','',
           '根据本地 [CUPID 原论文](paper/CUPID.pdf) §2.3、附录 A.2 与图 16，以真实情境偏好及真实 checklist 逐项分析已有回答，只评价 checklist 内的偏好满足程度，再给出 1–10 分。', '',
           f"采用 [官方评分提示](https://github.com/kixlab/CUPID/blob/{plan['official_revision']}/prompts/evaluation/response_judger.yaml)，固定版本 `{plan['official_revision']}`；[参考文件及摘要](paper/cupid_official/source.json)。官方发布提示较论文图 16 增加了各分数的标签及 Evaluation Label，保留其原始文本，不改成 JSON。", '',
           '裁判为 Pro/deepseek-ai/DeepSeek-V3.2，temperature=0；论文使用 gpt-4o-2024-11-20。因此这里是遵循官方评分提示的 API 裁判替代实验，不能直接对标论文中的绝对分数。沿用 M8 已冻结回答：生成前已先推断偏好，也不同于论文独立生成任务的流程。', '',
           '裁判仅看到当前请求、已有回答、真实偏好与真实 checklist，不看到方法名、检索上下文、模型推断的 checklist、F1 或旧代理评分。按官方格式保留逐项分析与原始输出，宿主严格解析最终 1–10 整数。来源与范围非法则记缺失，不人工补分。', '',
           f"原计划 24 题 × 5 方法；原执行缺失 27 项。已有 {len(rows)} 份回答，补评成功 {len(good)} 份，新增裁判失败 {len(rows)-len(good)} 份。五方法共同有效 {len(common)}/24 题，层分布 `{summary['common_strata']}`；主表严格使用同题子集，不将失败计为 0 或 1。", '',
           '| 方法 | 原有效回答 / 24 | 本次成功评分 | 同题均分（1–10） | 同题均值 95% 区间 |',
           '| --- | ---: | ---: | ---: | --- |']
    fmt=lambda value:'—' if value is None else f'{value:.3f}'
    ci=lambda value:'—' if value is None else '['+', '.join(f'{v:.3f}' for v in value)+']'
    for method in METHODS:
        r=metrics[method]
        lines.append(f"| {names[method]} | {r['saved_answers']} | {r['graded']} | {fmt(r['common_mean'])} | {ci(r['common_95ci'])} |")
    lines += ['', '区间按 persona 配对 bootstrap 2000 次、seed 20261017；每题一 persona。均值区间和方法差值区间不同，不能用均值区间重叠判断差值。补测发生在观察 M8 结果之后，以下均为探索性，未校正多重比较，不能替代原 F1 主比较。', '',
              '| 方法减平铺 | 同题差值 | 配对 95% 区间 |', '| --- | ---: | --- |']
    for method in METHODS:
        if method=='flat_m8':continue
        r=comparisons[method+'_vs_flat_m8']
        lines.append(f"| {names[method]} | {fmt(r['delta'])} | {ci(r['paired_cluster_95ci'])} |")
    lines += ['', '| 原生层 | 同题数量 | 原图 | 对象/维度图 | 平铺 | 原文 RAG | 完整历史 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in ['changing','consistent','contrastive']:
        lines.append(f"| {s} | {summary['common_strata'].get(s,0)} | "+' | '.join(fmt(metrics[m]['strata'][s]['mean']) for m in METHODS)+' |')
    lines += ['', '各方法全部已有回答的均值使用不同题集，仅用于执行诊断，不能作为公平排名：', '',
              '| 方法 | 不同题集数量 | 全部成功补评均值 |', '| --- | ---: | ---: |']
    for m in METHODS:
        lines.append(f"| {names[m]} | {metrics[m]['graded']} | {fmt(metrics[m]['mean_on_all_saved_answers'])} |")
    lines += ['', f"补测 HTTP {totals['http_requests']} 次，已知 provider tokens {totals['known_tokens']:,}，未知用量 {totals['unknown_usage_requests']} 次。仅为这次新增裁判调用，不包括原实验生成、记忆写入和 F1 评分。", '',
              'M8 的 F1、旧 response_score、生成文本和记忆快照保持不变；旧 response_score 混合通用有用性约束，不能重命名为本次偏好对齐分。默认仍 Qwen2.5-7B + v6_compact，不根据补测改默认或调参；没有开展 M9 API 实验。', '',
              '[冻结补测方案](runs/spm/m8_alignment_plan.json)、[指标和配对比较](runs/spm/m8_alignment/summary.json)、[逐项裁判原文](runs/spm/m8_alignment/grades/)、[新增缺失](runs/spm/m8_alignment/pending.json)、[原 M8 报告](SPM_M8_RESULTS.md)。', '']
    replay=OUT/'replay_verification.json'
    if replay.exists():
        r=read(replay)
        lines += [f"{r['replayed']} 份裁判输出与数值从验证缓存精确离线回放，0 HTTP；原样本、原记忆及成功补评文件字节不变。[回放核验](runs/spm/m8_alignment/replay_verification.json)。", '']
    (ROOT/'SPM_M8_ALIGNMENT_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','run','summarize','replay'])
    parser.add_argument('--resume', action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='summarize':summarize()
    else:asyncio.run(run(resume=args.resume, replay=args.action=='replay'))
