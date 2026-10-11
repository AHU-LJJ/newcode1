"""Render M5 evidence and limitations from verified offline artifacts."""
from pathlib import Path
import json
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/spm'
def read(p):return json.loads(p.read_text())
def f(x,rate=False):return '缺失' if x is None else (f'{x:.1%}' if rate else f'{x:.3f}')
KEYS=[('prefeval','follows_preference','PrefEval 遵循率',True),('cupid','checklist_f1','CUPID checklist F1',False),
      ('cupid','response_score','CUPID 回答质量（1–10）',False),('perma','mcq_accuracy','PERMA MCQ',True),
      ('benchpres','gate_AAR','BenchPreS AAR',True),('benchpres','gate_MR','BenchPreS MR',True)]
METHODS=['v6_compact_no_raw','v6_compact']
def main():
    plan=read(BASE/'m5_plan.json');v=read(BASE/'m5_verification.json');pending=read(BASE/'m5_pending.json')
    diag=v['phases']['diagnostic'];test=v['phases']['test']
    test_metrics=read(BASE/'m5_test/common_success_metrics.json')['metrics']
    headline=[]
    for b,k,label,rate in KEYS[:2]:
        a=test_metrics.get(b+'/'+METHODS[0]);z=test_metrics.get(b+'/'+METHODS[1])
        if a and z:
            headline.append(f"{label}（共同有效 {a['ok']} 题）：{f(a['complete_run_metrics'][k],rate)} → {f(z['complete_run_metrics'][k],rate)}")
    lines=['# SPM M5：Qwen 小模型的原文绑定接口与 v6 消融','',
           '2026-10-11（JST）。','',
           f"诊断阶段 {diag['ok']}/{diag['planned']} 项有效，新问题组测试 {test['ok']}/{test['planned']} 项有效。原文绑定和短编号接口提升了执行覆盖率；是否保留原文补充以新测试和配对区间为依据。",'',
           '新问题组的无原文 → v6 原文对照：'+'；'.join(headline)+'。这些指标只比较共同有效样本，缺失可能不随机。','',
           '后续系统模型统一为 `Qwen/Qwen2.5-7B-Instruct`，包含抽取、规范化、关系判断、查询、门控、推断与生成；评审继续使用 `Pro/deepseek-ai/DeepSeek-V3.2`，嵌入使用 `BAAI/bge-m3` API。`.env` 已仅更新模型字段，凭证保留。实验完成后，常规 CLI / SPMConfig / SPM 默认提升为 `v6_compact`；旧 `v6` 保留原接口定义作对照。','',
           '这轮沿 v6 推进，先处理 M4 暴露的小模型工具输出问题，再对原文补充单独消融。两个阶段、全部问题、提示与代码在首次预测前同时冻结，诊断结果不用于修改本轮测试实现。没有引入强化学习或强模型写入。','',
           '[冻结方案](runs/spm/m5_plan.json)、[核验](runs/spm/m5_verification.json)、[缺失队列](runs/spm/m5_pending.json)。','',
           '## 接口改变','',
           'USER 消息按句号/换行分为原文不改写的片段，长片段分段到最多 600 字符。模型选择短 source_id，宿主只将该编号映射到已有原文和原始 turn_idx；assistant 文本没有可选证据编号。空白保留为上下文但不能作证据。未知编号、非法枚举、超长内容仍触发修复/失败，没有自动接受改写引文。完整历史仍在 SQLite 中。','',
           '显式给定的 BenchPreS 偏好逐条规范化，宿主绑定该条完整原文和 explicit 表达类型；已知输入与隐藏适用标签严格分离。门控和关系返回本次调用的整数序号，必须每项恰好一次；代码映射到稳定 ID，仍保留实体不匹配与冲突保护。','',
           '这种方式改变了证据粒度和提示：宿主选出的是模型指定的现有片段，不是在核实语义推断本身。结构校验通过仍可能意味着抽取遗漏、选错支持句或作用域判断错误。M4 对照衡量的是整套接口变化，不能归因于某一个字段修改。','',
           '`v6_compact_no_raw` 与 `v6_compact` 共享完全相同的写入库、查询和门控，仅后者给回答/推断器补充最多三个检索会话窗口、累计 6,000 个片段字符。BenchPreS given 模式继续关闭原文补充；它只检验接口与门控，不能评估原文的额外价值。','',
           '## 覆盖率与分组','',
           '诊断：M4 相同 59 题（PrefEval 24、CUPID 12、PERMA 3、BenchPreS 20），已看过，只用于可靠性与同题比较。新测试：预先排除所有旧实验中已经出现过的问题组，选择 PrefEval 24 题和 CUPID 12 题。当前 24 历史 episode 上限内，没有未见 PERMA sd_clean type2 用户；BenchPreS 没有未见画像，因此新测试不包含它们。','',
           '| 阶段 / 基准 | 无原文：有效 / 计划 | v6 原文：有效 / 计划 |','| --- | ---: | ---: |']
    for phase,p in plan['phases'].items():
        metrics=read(BASE/f'm5_{phase}/metrics.json')
        for b in sorted({c['benchmark'] for c in p['cases']}):
            n=sum(c['benchmark']==b for c in p['cases'])
            lines.append(f"| {phase} / {b} | "+' | '.join(f"{metrics.get(b+'/'+m,{}).get('ok',0)} / {n}" for m in METHODS)+' |')
    for phase in plan['phases']:
        out=BASE/f'm5_{phase}';full=read(out/'metrics.json');common=read(out/'common_success_metrics.json');pairs=read(out/'comparisons_raw.json')
        info=v['phases'][phase]
        lines+=['',f'## {"新问题组测试" if phase=="test" else "固定问题诊断"}','',
                f"{info['ok']}/{info['planned']} 项有效评分；两版本共同成功 {info['common_cases']} 题。以下表格只比较共同成功题；若该基准仍有缺失，不能当作全量基准成绩。完整指标保留在 JSON，未完成基准的 complete_run_metrics 为 null。",'',
                '| 指标 | 共同题数 | 无原文 | v6 原文 | 差值 | 成对 95% 区间 |','| --- | ---: | ---: | ---: | ---: | --- |']
        for b,k,label,rate in KEYS:
            entry=common['metrics'].get(b+'/'+METHODS[0])
            comparison=pairs.get(f'{b}/{METHODS[1]}_vs_{METHODS[0]}')
            if not entry or not comparison or k not in comparison['metrics']:continue
            a=entry['complete_run_metrics'];z=common['metrics'][b+'/'+METHODS[1]]['complete_run_metrics'];d=comparison['metrics'][k];ci=d['paired_cluster_95ci']
            lines.append(f"| {label} | {entry['ok']} | {f(a[k],rate)} | {f(z[k],rate)} | {d['delta']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
        lines+=['',f'[原始指标](runs/spm/m5_{phase}/metrics.json)、[配对区间](runs/spm/m5_{phase}/comparisons_raw.json)、[共享写入与门控审计](runs/spm/m5_{phase}/gate_audit.json)。','']
        writes=read(out/'write_quality_audit.json')
        lines+= [f"该阶段有效案例对应 {len(writes)} 份记忆检查，空库 {sum(w['empty_bank'] for w in writes)} 份；检查了 {sum(w['grounded_user_evidence'] for w in writes)} 条持久化证据的 USER 角色、turn_idx 和原文子串。这个检查核实来源对应，不核实偏好推断的语义正确性。",'']
    legacy=read(BASE/'m5_diagnostic/comparisons_legacy.json');old=read(BASE/'m4_test/metrics.json');new=read(BASE/'m5_diagnostic/metrics.json')
    lines+=['## 与旧 v6 的诊断对照','',
            '旧 v6 是 M4 的既有记录，保留全部失败，不重试或重新抽样。新接口从原始历史建库；完全相同的已验证 embedding 或裁判输入可以命中隔离缓存。下面覆盖率使用同一批计划题，评分差值只用两个接口共同有效的问题；恢复的缺失样本不加入旧接口成绩分母。','',
            '| 基准 | 旧 v6 有效 | 新 v6 有效 |','| --- | ---: | ---: |']
    for b in ('prefeval','cupid','perma','benchpres'):
        lines.append(f"| {b} | {old[b+'/v6']['ok']} / {old[b+'/v6']['attempted']} | {new[b+'/v6_compact']['ok']} / {new[b+'/v6_compact']['attempted']} |")
    lines+=['','| 指标 | 共同题 / 组 | 旧 v6 | 新 v6 | 差值 | 95% 区间 |','| --- | ---: | ---: | ---: | ---: | --- |']
    for b,k,label,rate in KEYS:
        c=legacy.get(f'{b}/compact_v6_vs_legacy_v6')
        if not c or k not in c['metrics']:continue
        d=c['metrics'][k];ci=d['paired_cluster_95ci']
        lines.append(f"| {label} | {c['paired_cases']} / {c['groups']} | {f(d['baseline_score'],rate)} | {f(d['method_score'],rate)} | {d['delta']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    lines+=['','[旧接口配对](runs/spm/m5_diagnostic/comparisons_legacy.json)、[共同有效子集](runs/spm/m5_diagnostic/legacy_common_metrics.json)、[M4 报告](SPM_M4_RESULTS.md)。','',
            '## API 用量与验证','',
            'HTTP 并发 8、任务 worker 16；chat/cheap 共用 Qwen 80,000 TPM 配额，客户端按 72,000 TPM 预留并用本地官方 tokenizer 计数。最大输出预留、重试、未知用量与缓存命中处理保持 M4 规则。所有 invocation 串行执行，限速状态存在同一隔离缓存中。外部同账户请求不在此协调范围。','',
            '| 阶段 | HTTP 尝试 | 已知 provider token | 未知用量 | HTTP 状态 |','| --- | ---: | ---: | ---: | --- |']
    for phase in plan['phases']:
        u=read(BASE/f'm5_{phase}/ledger_totals.json')
        lines.append(f"| {phase} | {u['http_attempts']} | {u['known_tokens']:,} | {u['unknown_usage']} | `{u['statuses']}` |")
    for phase in plan['phases']:
        u=read(BASE/f'm5_{phase}/ledger_totals.json')
        lines+=['',f"{phase} 的 schema/语义修复失败尝试（包含已修复成功的请求）：`{u['schema_failures']}`。",'']
    lines+=['[新问题 PrefEval 错误审计](runs/spm/m5_test/prefeval_error_audit.json)保存候选、门控、上下文、回答与既有裁判结果，供后续区分抽取、门控与生成失败；没有独立核实裁判涉及的外部事实。','']
    if pending:
        counts=Counter(p['error'].get('message','').split(':')[0] if isinstance(p['error'],dict) else str(p['error']) for p in pending)
        lines+=['',f'最终未完成 {len(pending)} 项，阶段：`{dict(counts)}`。失败按缺失保留，没有计为答错或改为人工分数。']
    else:lines+=['','全部计划项获得有效评分；schema 修复次数与最终样本失败分开统计。']
    first=read(BASE/'m5_first_pass_summary.json')
    lines+=['',f"首轮诊断 {first['diagnostic']['ok']}/{first['diagnostic']['checkpointed']} 项有效，新测试 {first['test']['ok']}/{first['test']['checkpointed']} 项有效。随后按同一冻结协议，只将执行失败项续跑一次；成功答案不重生成、不按评分挑选。首轮错误快照分别保存在诊断和测试的 first_pass_errors 目录。",'',
            f"续跑前成功的 {v['successful_checkpoints_before_resume_unchanged']} 个检查点逐字节不变。诊断新增 {diag['ok']-first['diagnostic']['ok']} 项有效，新测试新增 {test['ok']-first['test']['ok']} 项有效。",'']
    lines+=['',f"实验冻结前 63 项测试通过，默认提升与 SDK 自动 TPM 保护完成后 64 项通过。旧实验 {v['prior_samples_unchanged']} 个样本文件摘要不变；模型、冻结源码、数据摘要、原文预算和共享门控核验通过，离线分析新增 API 请求为 0。",'',
            '根据接口完成率、新问题测试和用户选择，当前默认采用 Qwen + v6_compact。[默认提升记录](runs/spm/m5_default_promotion.json)记录源码变化；M5 的预测仍使用原冻结包。新测试的 CUPID 成功配对子集出现正向收益，F1 差值区间 [0.042, 0.338]；但仅 9/12 题有效。PrefEval 差值区间 [-0.083, 0.208] 跨零，尚不能确认稳定收益，系统绝对成绩仍偏低。后续优先为漏返候选增加定向补齐，保持最终完全覆盖校验。','',
            '本轮未包含 Qwen b1/b2，图记忆与完整上下文或纯原文 RAG 的同模型对照仍需补齐。PrefEval 为遵循率代理裁判；CUPID 是 checklist 代理评分与回答质量评分，不是官方 PrefMatcher；PERMA 是短历史用户的选择题小样本；BenchPreS 是给定偏好的门控适用性。小规模、探索性多重指标、代理裁判和单次模型采样限制结论，区间未进行多重比较校正。','',
            '```bash','uv run python scripts/spm_m5.py run --phase diagnostic','uv run python scripts/spm_m5.py run --phase test',
            'uv run python scripts/spm_m5.py analyze','uv run python scripts/spm_m5_report.py','```','']
    replay=read(BASE/'m5_replay_verification.json')
    assert all(p['http_requests']==0 for p in replay['phases'].values())
    lines+=['[离线回放](runs/spm/m5_replay_verification.json)禁用请求发送，验证成功检查点跳过，并仅凭已验证缓存重建成功案例的完整检索上下文及 trace，逐字段对照保存结果，同时检查记忆逻辑快照不变；没有重新生成或独立重新打分。','']
    (ROOT/'SPM_M5_RESULTS.md').write_text('\n'.join(lines))
    print(ROOT/'SPM_M5_RESULTS.md')
if __name__=='__main__':main()
