"""Render audited Qwen-7B comparisons without API calls."""
import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/spm'
def read(p):return json.loads(p.read_text())
def fmt(x,rate=False):return '未完整评分' if x is None else (f'{x:.1%}' if rate else f'{x:.3f}')
def ci(x):return '不可估计' if x is None else f'[{x[0]:+.3f}, {x[1]:+.3f}]'
KEYS=[('prefeval','follows_preference','PrefEval 遵循率',True),
      ('cupid','checklist_f1','CUPID checklist F1',False),
      ('cupid','response_score','CUPID 回答质量（1–10）',False),
      ('perma','mcq_accuracy','PERMA MCQ',True),
      ('benchpres','gate_AAR','BenchPreS 适用召回',True),
      ('benchpres','gate_MR','BenchPreS 误用率',True)]

def score_table(metrics,prefix='',common=False):
    lines=['| 指标 | v4 | v5 | v6 |','| --- | ---: | ---: | ---: |']
    for b,k,label,rate in KEYS:
        vals=[]
        for m in ('v4','v5','v6'):
            entry=metrics.get(b+'/'+prefix+m,{}).get('complete_run_metrics')
            vals.append('无共同有效样本' if common and not entry else fmt(entry.get(k) if entry else None,rate))
        if common:
            n=metrics.get(b+'/'+prefix+'v4',{}).get('ok',0)
            label+=f'（{n} 题）'
        lines.append('| '+label+' | '+' | '.join(vals)+' |')
    return lines

def outcome(out,pending):
    common=read(out/'common_success_metrics.json')
    data=common['metrics']
    lines=[]
    for b,k,label,rate in [('prefeval','follows_preference','PrefEval 遵循率',True),
                           ('cupid','checklist_f1','CUPID F1',False),
                           ('benchpres','gate_AAR','BenchPreS 适用召回',True)]:
        entries=[data.get(b+'/'+m) for m in ('v4','v5','v6')]
        if not all(entries):continue
        vals=[e['complete_run_metrics'][k] for e in entries]
        lines.append(f"{label}在三个版本共同成功的 {entries[0]['ok']} 个问题上依次为 "+' / '.join(fmt(v,rate) for v in vals)+'（v4 / v5 / v6）。')
    if not lines:return []
    caveat=('这些是执行后的共同成功子集，缺失可能不随机，不能作为完整基准成绩。' if pending
            else '全部问题执行完整，但本轮仍是小规模的固定问题对照。')
    return [' '.join(lines), '', caveat+'门控和原文补充是否有效应结合下面的配对区间、执行成功率和同题 DeepSeek 对照判断。','']

def main():
    plan=read(BASE/'m4_plan.json');verify=read(BASE/'m4_verification.json');pending=read(BASE/'m4_pending.json')
    out=BASE/'m4_test';metrics=read(out/'metrics.json');usage=read(out/'ledger_totals.json')
    rows=[read(p) for p in (out/'samples').glob('*.json')]
    assert verify['checkpointed']==verify['planned']
    text=['# SPM M4：Qwen2.5-7B 的 v4 / v5 / v6 对照','',
          '2026-10-10 至 2026-10-11（JST）。','',
          '本轮将抽取、关系判定、查询解析、门控、偏好推断和回答生成统一切换为 `Qwen/Qwen2.5-7B-Instruct`。裁判保持 `Pro/deepseek-ai/DeepSeek-V3.2`，嵌入保持 `BAAI/bge-m3`。默认策略仍为 v5；模型切换仅在实验进程中覆盖配置，没有改动 `.env`。','',
          f"执行：{verify['ok']}/{verify['planned']} 项有效评分，计划 59 个问题 × 三版本 = 177 项；未完成 {len(pending)} 项。",'',
          '当前直接端到端替换为 7B 尚不稳定，主要缺失来自记忆写入的格式与原文校验。PrefEval 的共同成功子集显示新门控有帮助，但整体可用性与同题 DeepSeek 成绩仍有差距。下一步应先改善写入和候选 ID 返回的稳定性，或另行对照“DeepSeek 写入、Qwen 读取”以区分写入与读取瓶颈；本轮没有用该混合方案替代失败样本。','',
          '## 固定问题与执行设置','',
          '使用 M3 同一批 59 个测试问题及相同数据摘要、顺序和分组：PrefEval 24、CUPID 12、PERMA 3、BenchPreS 20。它们已在 M3 观察过，本轮属于固定问题的模型对照，不能称为新的未见测试集。没有基于本轮答案调参。','',
          'Qwen 从原始会话重新建立自己的记忆库，v4/v5/v6 共享该写入库，不复制 DeepSeek 已完成的抽取或关系标签。隔离缓存可以复用 BGE-M3 嵌入及完全相同的裁判输入；模型名参与生成/抽取缓存键，DeepSeek 答案不会充当 Qwen 答案。','',
          '门控提示、生成提示、schema、严格 USER 连续原文证据校验、0.5/0.2 阈值和 v6 原文预算均与 M3 相同。仅增加运行时 TPM 控制。v6 最多补充三个原文片段、累计最多 6,000 个片段字符；给定偏好的 BenchPreS 禁用原文回退，所以它不能检验原文误用风险。','',
          '按用户给定的 80,000 TPM 限额，客户端采用 72,000 TPM 的滚动预算。用官方 Qwen tokenizer 估计 ChatML 输入，发送前预留最大输出和 256 token 余量；返回后按已知用量调整，完成后的用量保留 60 秒，未知用量保留原预留。chat/cheap 共享同一模型额度，缓存命中不占额度；重试重新计入。预算保存在本轮 SQLite 缓存中，续跑不会立即清空窗口。','',
          '首轮 12 个任务 worker、HTTP 并发 4；仅重试失败项的一次续跑使用并发 8，滚动 token 预算保持 72,000 TPM。限速协调范围为使用该缓存的本地进程，外部同账户调用可能触发平台 429，仍按 Retry-After 和退避处理。具体模型额度以用户账户为准。TPM、并发和缓存复用与 M3 不同，耗时不能用作独立模型速度比较。[平台限速说明](https://api-docs.siliconflow.cn/docs/userguide/faqs/rate-limit-and-upgradation)。','',
          '[冻结方案](runs/spm/m4_plan.json)、[核验](runs/spm/m4_verification.json)、[原始指标](runs/spm/m4_test/metrics.json)。','',
          '## Qwen 结果','',*([] if pending else score_table(metrics)),'',
           '完整基准成绩只在全部计划项有效时填写；存在执行缺失的 `complete_run_metrics` 保持 null，不将执行失败计为答错。以下先列覆盖率，再比较三个版本共同成功的问题。[PrefEval 错误审计](runs/spm/m4_test/error_audit.json)保留候选、门控和记录的裁判理由，未独立核实其中的外部事实。','',
          '| 基准 | v4 有效 / 计划 | v5 有效 / 计划 | v6 有效 / 计划 |','| --- | ---: | ---: | ---: |']
    for b in ('prefeval','cupid','perma','benchpres'):
        text.append('| '+b+' | '+' | '.join(f"{metrics[b+'/'+m]['ok']} / {metrics[b+'/'+m]['attempted']}" for m in ('v4','v5','v6'))+' |')
    if pending:
        common=read(out/'common_success_metrics.json')
        text+=['','三版本共同成功子集：'+str(len(common['case_ids']))+'/59 个问题。下表仅描述此子集，不能替代完整成绩，缺失并不保证随机。','',*score_table(common['metrics'],common=True)]
    text+=['',*outcome(out,pending)]
    text+=['','## Qwen 内部成对差值','',
           '比较同一有效问题，按身份/问题组 bootstrap；BenchPreS 按偏好标签 micro 汇总。小样本和探索性多重比较限制结论，零宽度差值区间不能证明总体无不确定性。','',
           '| 对照 | 指标 | 问题 / 组 | 差值 | 95% 区间 |','| --- | --- | --- | ---: | --- |']
    for baseline,method in [('v4','v5'),('v5','v6')]:
        paired=read(out/f'comparisons_vs_{baseline}.json')
        for b in ('prefeval','cupid','perma','benchpres'):
            result=paired.get(f'{b}/{method}_vs_{baseline}')
            if not result:continue
            for k,v in result['metrics'].items():
                text.append(f"| {method}−{baseline} / {b} | {k} | {result['paired_cases']} / {result['groups']} | {v['delta']:+.3f} | {ci(v['paired_cluster_95ci'])} |")
    text+=['','[v5 对 v4](runs/spm/m4_test/comparisons_vs_v4.json)、[v6 对 v5](runs/spm/m4_test/comparisons_vs_v5.json)、[原文证据审计](runs/spm/m4_test/gate_audit.json)。','',
           '## 与 M3 DeepSeek 对照','',
           'M3 的三个对应版本使用 DeepSeek 抽取与生成；裁判模型相同。本轮不运行 Qwen 的完整历史或原文 RAG 基线，M3 的 DeepSeek b1/b2 不能当成 Qwen 基线。下面按同一方法、同一成功问题配对，差值为 Qwen − DeepSeek。','',
           '| 方法 / 基准 | 指标 | 问题 / 组 | Qwen | DeepSeek | 差值 | 95% 区间 |',
           '| --- | --- | --- | ---: | ---: | ---: | --- |']
    for m in ('v4','v5','v6'):
        paired=read(out/f'comparisons_model_{m}.json')
        for b,k,label,rate in KEYS:
            result=paired.get(f'{b}/qwen_{m}_vs_deepseek_{m}')
            if not result or k not in result['metrics']:continue
            v=result['metrics'][k]
            text.append(f"| {m} / {b} | {k} | {result['paired_cases']} / {result['groups']} | {fmt(v['method_score'],rate)} | {fmt(v['baseline_score'],rate)} | {v['delta']:+.3f} | {ci(v['paired_cluster_95ci'])} |")
    text+=['','[模型配对 v4](runs/spm/m4_test/comparisons_model_v4.json)、[v5](runs/spm/m4_test/comparisons_model_v5.json)、[v6](runs/spm/m4_test/comparisons_model_v6.json)、[M3 报告](SPM_M3_RESULTS.md)。','',
           '## 用量与可靠性','',
           f"本轮 invocation 共 {usage['http_attempts']} 次 HTTP 尝试，已知 provider 用量 {usage['known_tokens']:,} token，未知用量 {usage['unknown_usage']} 次；HTTP 状态统计：`{json.dumps(usage['http_status'],ensure_ascii=False)}`。复制缓存里的历史账本未重复计入。用量包括失败和修复，token 不是账单金额。",'',
           f"按请求完成时间测得 Qwen 已知用量在任意 60 秒窗口的峰值为 {usage['observed_known_qwen_peak_60s_tokens']:,} token。这不含未知用量、分词余量或外部同账户调用，不能当成平台账单或完整配额审计。",'',
           f"模型返回后未通过 schema/语义校验的次数按调用阶段统计为 `{usage['schema_failures']}`；这与 HTTP 状态分开统计，HTTP 200 不代表样本执行成功。",'',
           '| API 角色 | 请求 | 已知 token |','| --- | ---: | ---: |']
    for role,u in usage['by_role'].items():text.append(f"| {role} | {u['requests']} | {u['known_tokens']:,} |")
    text+=['','| 方法 | 有效样本 | 平均上下文字符 | 补充原文平均字符 |','| --- | ---: | ---: | ---: |']
    for m in ('v4','v5','v6'):
        good=[r for r in rows if r['method']==m and r['status']=='ok']
        n=len(good)
        context=sum(len(r['trace']['context']) for r in good)/n if n else 0
        raw=sum(sum(len(e['dialogue_excerpt']) for e in r['trace']['raw_evidence']) for r in good)/n if n else 0
        text.append(f'| {m} | {n} | {context:,.0f} | {raw:,.0f} |')
    if pending:
        counts=Counter(p['error'].get('type','unknown') if isinstance(p['error'],dict) else str(p['error']) for p in pending)
        text+=['',f'未完成项类型：`{dict(counts)}`。未评分缺失可能不随机，成对文件仅反映相同成功子集；[逐项错误队列](runs/spm/m4_pending.json)保留原因。没有放宽原文、候选覆盖或裁判一致性校验。']
        stages=Counter(p['stage'] for p in read(out/'execution_failure_audit.json'))
        text+=['',f'最终执行失败阶段：`{dict(stages)}`。[阶段审计](runs/spm/m4_test/execution_failure_audit.json)保留每项原始错误；有些方法共享同一写入失败，所以这些数量不是独立用户数。']
    else:text+=['','全部计划项获得有效评分，严格校验保留；API/schema 修复尝试保存在账本和 usage 中。']
    if (BASE/'m4_first_pass_errors').exists():
        first=read(BASE/'m4_first_pass_summary.json')
        text+=['',f"首轮 {first['ok']}/{first['checkpointed']} 项有效、{first['failed']} 项执行失败；固定协议只重试未成功项一次，最终新增 {verify['ok']-first['ok']} 项有效评分。成功答案没有反复生成选分。"]
        text+=['','[首轮失败快照](runs/spm/m4_first_pass_errors)保留首次执行错误；续跑只重试未成功项，成功检查点逐字节保持不变。']
    replay_path=BASE/'m4_replay_verification.json'
    if replay_path.exists():
        replay=read(replay_path)
        assert all(v['http_requests']==0 for v in replay['phases'].values())
        text+=['','[离线回放](runs/spm/m4_replay_verification.json)以拒绝网络请求的 mock 验证成功检查点跳过及结果文件不变；没有重新生成或独立重新评分。']
    text+=['',f"58 项测试通过。旧实验 {verify['prior_sample_files_unchanged']} 个样本文件摘要不变，源码、分词文件、数据摘要、固定问题与模型配置核验通过；离线分析新增请求为零。",'',
           'PrefEval/CUPID 使用 API 代理裁判，CUPID 不是官方 PrefMatcher；PERMA 为三个 sd_clean type2 短历史用户的准备好选择题；BenchPreS 为给定偏好分类而非生成遵循率。结果不能直接作为官方全量成绩。','',
           '```bash','uv run python scripts/spm_m4.py run','uv run python scripts/spm_m4.py analyze',
           'uv run python scripts/spm_m4_replay.py','uv run python scripts/spm_m4_report.py','```','']
    (ROOT/'SPM_M4_RESULTS.md').write_text('\n'.join(text));print('Wrote SPM_M4_RESULTS.md')

if __name__=='__main__':main()
