"""Render M3's audited results; all scores come from frozen checkpoints."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/spm'
def read(p):return json.loads(p.read_text())
def fmt(x):return '未完整评分' if x is None else f'{x:.3f}'
def context_fmt(x):return '未记录' if x is None else f'{x:,.0f}'
def ci(v):return '不可估计' if v is None else f'[{v[0]:+.3f}, {v[1]:+.3f}]'

def interpretation(rows,v4,v5):
    if any(r['status']!='ok' for r in rows):return []
    metrics=read(BASE/'m3_test/metrics.json')
    def metric(b,m,k):return metrics[b+'/'+m]['complete_run_metrics'][k]
    pref=[metric('prefeval',m,'follows_preference') for m in ('v4','v5','v6')]
    cupid=[metric('cupid',m,'checklist_f1') for m in ('v4','v5','v6')]
    aar=[metric('benchpres',m,'gate_AAR') for m in ('v4','v5')]
    raw_pref=v5['prefeval/v6_vs_v5']['metrics']['follows_preference']
    raw_cupid=v5['cupid/v6_vs_v5']['metrics']['checklist_f1']
    lines=[f'本轮新测试中，旧门控 → 新门控 → 原文证据的 PrefEval 遵循率为 {pref[0]:.1%} → {pref[1]:.1%} → {pref[2]:.1%}，CUPID F1 为 {cupid[0]:.3f} → {cupid[1]:.3f} → {cupid[2]:.3f}。新门控的 BenchPreS 适用召回由 {aar[0]:.1%} 提升至 {aar[1]:.1%}，两者误用均为零。M2 的 CUPID 门控差值为负、本轮为正，跨轮方向并不一致，不能宣称稳定全面改善。','',
           f'原文证据相对 v5 的 PrefEval 差值 95% 区间为 {ci(raw_pref["paired_cluster_95ci"])}，CUPID F1 为 {ci(raw_cupid["paired_cluster_95ci"])}，均跨零。本轮 v6 的 PrefEval 点估计与完整历史相同，但不等于二者已证明等效或 v6 稳定占优。','']
    counts={m:{k:sum(r['evaluation']['metrics'][k] for r in rows if r['benchmark']=='prefeval' and r['method']==m)
               for k in ('preference_unaware','inconsistent','hallucinated_preference')} for m in ('v5','v6')}
    lines += [f'错误审计中，v5 → v6 的未注意偏好次数为 {counts["v5"]["preference_unaware"]:.0f} → {counts["v6"]["preference_unaware"]:.0f}，偏好不一致为 {counts["v5"]["inconsistent"]:.0f} → {counts["v6"]["inconsistent"]:.0f}，偏好幻觉为 {counts["v5"]["hallucinated_preference"]:.0f} → {counts["v6"]["hallucinated_preference"]:.0f}。原文帮助恢复信息，但恢复后的生成约束仍需改进；这些是裁判标签，不是独立事实核验。','',
              '继续使用用户选择的 v5 默认；v6 保留作实验分支。下一步优先检查原文恢复后发生冲突的案例，再做规则消融与更大规模的新用户验证。','']
    return lines

def table(phase,plan):
    data=read(BASE/f'm3_{phase}/metrics.json');methods=plan['phases'][phase]['methods']
    lines=['| 指标 | '+' | '.join(methods)+' |','| --- | '+' | '.join('---:' for _ in methods)+' |']
    for bench,key,label in [('prefeval','follows_preference','PrefEval 遵循率'),('cupid','checklist_f1','CUPID checklist F1'),
                            ('cupid','response_score','CUPID 回答质量（1–10）'),('perma','mcq_accuracy','PERMA MCQ')]:
        vals=[]
        for m in methods:
            value=data[bench+'/'+m]['complete_run_metrics'];x=None if value is None else value.get(key)
            vals.append(f'{x:.1%}' if x is not None and bench in ('prefeval','perma') else fmt(x))
        lines.append('| '+label+' | '+' | '.join(vals)+' |')
    vals=[]
    for m in methods:
        v=data['benchpres/'+m]['complete_run_metrics']
        vals.append('未完整评分' if v is None else f"{v['gate_AAR']:.1%} / {v['gate_MR']:.1%}")
    lines.append('| BenchPreS gate_AAR / gate_MR | '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def main():
    plan=read(BASE/'m3_plan.json');verify=read(BASE/'m3_verification.json');pending=read(BASE/'m3_pending.json')
    assert all(v['planned']==v['checkpointed'] for v in verify['phases'].values())
    rows={phase:[read(f) for f in (BASE/f'm3_{phase}/samples').glob('*.json')] for phase in ('dev','test')}
    v5=read(BASE/'m3_test/comparisons_vs_v5.json');v4=read(BASE/'m3_test/comparisons_vs_v4.json')
    text=['# SPM M3：新门控复验与原文证据消融','',
          '2026-10-10。按用户选择，CLI 和直接构造 SPM 的默认版本已切换为 **v5 答案过滤门控**，v4 仍可指定作为旧版对照。v6 是原文证据实验变体，未自动设为默认。模型、阈值和样本在本轮新预测前冻结；没有强化学习。','',
          '执行覆盖：'+ '；'.join(f"{p} {v['ok']}/{v['planned']} 有效评分" for p,v in verify['phases'].items())+('。未评分项不按答错处理。' if pending else '。无未完成项。'),'',
          '## 方案与数据划分','',
          '开发组为旧开发组 39 个问题加先前失败的 cupid/506 诊断题，v5/v6 共 80 项。测试 PrefEval 24 个问题（explicit/choice/persona 各八条）、CUPID 12 个问题、PERMA 三个用户，身份/问题组均排除所有先前观察组。BenchPreS 没有剩余未观察画像，采用十个已见画像各两个未使用问题，共 20 条；这个部分验证新问题上的行为，不能声称新用户泛化。测试共 59 条 × 五方法 = 295 项。','',
          'PERMA 是 sd_clean type2、按公共历史不超过 24 会话筛选的子集，本轮实际历史长度为 '+ '/'.join(str(c['episodes']) for c in plan['phases']['test']['cases'] if c['benchmark']=='perma')+'；没有截断历史，不代表全部 PERMA。所有新测试问题 ID 都未在之前运行过。','',
          'v4 是旧门控；v5 是用户选择的新门控；v6 为 v5 加最多三个检索原文片段、累计最多 6,000 个片段字符，保留角色、会话顺序/时间及截断标记。原文为证据，不自动认作偏好，也不应覆盖已解决的变更或跨实体条件。v5/v6 共享写入库，候选和门控决策相同，v6 不增加 LLM 读取调用。','',
          '给定偏好的 BenchPreS 关闭原文回退，故 v5/v6 相同是实现约束，不能用它证明原文证据不会导致误用。b1 在此应用全部给定偏好，b2 的 top-k=5 也覆盖全部五条，均为弱门控参照。','',
          '此次原文校验修复反馈提供所引用 USER 原句的短片段或确切匹配的 USER 索引，供模型自己纠正；没有自动改写证据或接受近似引用。所有进行结构化写入的方法共享该改动，可能影响原本失败的新写入；旧开发库和有效缓存复用，因此不能将两个阶段的耗时作为独立冷启动比较。','',
          '[预先固定方案](runs/spm/m3_plan.json)、[核验](runs/spm/m3_verification.json)、[系统文档](SPM.md)。','',
          '## 测试结果','',table('test',plan),'',*interpretation(rows['test'],v4,v5),
          '以下主要差值配对相同成功问题，按身份/问题组进行 bootstrap；BenchPreS 按偏好标签 micro 汇总。'+('失败缺失未必随机。' if pending else '')+'小样本及多项探索性比较限制解释；零差值区间仅反映已观察的配对差异，不能说明真实效果没有不确定性。','',
          '| 对照 | 指标 | 问题 / 组 | 差值 | 95% 区间 |','| --- | --- | --- | ---: | --- |']
    for b in ('prefeval','cupid','perma','benchpres'):
        for comparison,data in [('v5_vs_v4',v4),('v6_vs_v5',v5)]:
            result=data[b+'/'+comparison]
            for key,v in result['metrics'].items():
                text.append(f"| {b}/{comparison} | {key} | {result['paired_cases']} / {result['groups']} | {v['delta']:+.3f} | {ci(v['paired_cluster_95ci'])} |")
    text+=['','[对照 v4](runs/spm/m3_test/comparisons_vs_v4.json)、[对照 v5](runs/spm/m3_test/comparisons_vs_v5.json)、[对照完整历史](runs/spm/m3_test/comparisons_vs_b1.json)、[对照 RAG](runs/spm/m3_test/comparisons_vs_b2.json)。','',
           '## 开发诊断','',table('dev',plan),'',
           '开发分数用于诊断，包含已观察问题，不代替新测试表现。cupid/506 的执行情况：','']
    text+=['- '+r['method']+'：'+r['status']+(('，'+str(r['error'])) if r['status']!='ok' else '，严格证据校验通过。') for r in rows['dev'] if r['case_id']=='cupid/506']
    text+=['','[开发指标](runs/spm/m3_dev/metrics.json)、[证据消融审计](runs/spm/m3_test/gate_audit.json)、[PrefEval 错误审计](runs/spm/m3_test/error_audit.json)。审计中的裁判理由是评分记录，关于现实作品或品牌的说法未经独立核实。','',
           '## 上下文和用量','',
           '| 测试方法 | 有效样本 | 平均上下文字符 | 补充原文平均字符 |','| --- | ---: | ---: | ---: |']
    for m in plan['phases']['test']['methods']:
        good=[r for r in rows['test'] if r['method']==m and r['status']=='ok']
        known=[r for r in good if 'context' in r.get('trace',{})]
        avg=sum(len(r['trace']['context']) for r in known)/len(known) if known else None
        raw=sum(sum(len(x['dialogue_excerpt']) for x in r.get('trace',{}).get('raw_evidence',[])) for r in good)/len(good) if good else None
        text.append(f'| {m} | {len(good)} | {context_fmt(avg)} | {context_fmt(raw)} |')
    text+=['','结构化方法的上下文字符来自 trace，b1/b2 当前 trace 不保存完整文本，因此标为未记录。平均值包含门控基准；它不等于 provider token，也不能单独代表推理成本。','',
           '| 阶段 | HTTP 尝试 | 已知 provider token | 未知用量请求 |','| --- | ---: | ---: | ---: |']
    for phase in ('dev','test'):
        u=read(BASE/f'm3_{phase}/ledger_totals.json')
        text.append(f"| {phase} | {u['http_attempts']} | {u['known_tokens']:,} | {u['unknown_usage']} |")
    text+=['','只统计 M3 invocation，复制缓存中的历史账本没有重复计费。失败/重试保留；token 不是账单金额，未知用量不能当成免费。两阶段输出分开，共享本轮隔离的 API 缓存副本。','',
           '## 执行可靠性与边界','']
    if pending:
        text+=['以下计划项未获得有效结果，完整指标保持 null；查看成对文件中的相同成功样本成绩。','']
        text+=['未完成 '+str(len(pending))+' 项，逐项原因见 [待处理清单](runs/spm/m3_pending.json)。']
    else:text+=['全部计划项获得有效结果；保持严格 schema、USER 原文证据和候选 ID 覆盖校验。']
    if 'successful_checkpoints_before_resume_unchanged' in verify:
        text+=['',f"首轮因 HTTP 402 中断，在用户恢复额度后续跑；续跑前已有的 {verify['successful_checkpoints_before_resume_unchanged']} 项有效结果逐字节保持不变。"]
    replay_path=BASE/'m3_replay_verification.json'
    if replay_path.exists():
        replay=read(replay_path)
        assert all(v['http_requests']==0 for v in replay['phases'].values())
        text+=['','[离线检查点回放](runs/spm/m3_replay_verification.json)用拒绝任何网络请求的 mock 重放成功检查点，新增请求为零、结果文件保持不变。这验证断点跳过行为，不是独立重生成或重新评分。']
    text+=['','53 项测试通过。旧样本文件摘要、源码冻结、数据摘要和分组检查通过；离线分析新增请求为零，graphmem 保持原进度。','',
           '模型和裁判同为 DeepSeek-V3.2，嵌入为 BAAI/bge-m3 API。PrefEval 无干扰历史且使用 API 代理裁判；CUPID 为 API checklist 匹配、不是官方 PrefMatcher；PERMA 为准备好的单步 MCQ；BenchPreS 为给定偏好门控，不是生成消息遵循率。结果不能直接对标论文全量成绩。','',
           '```bash','uv run python scripts/spm_m3.py run --phase dev','uv run python scripts/spm_m3.py run --phase test','uv run python scripts/spm_m3.py analyze','uv run python scripts/spm_m3_replay.py','uv run python scripts/spm_m3_report.py','```','']
    (ROOT/'SPM_M3_RESULTS.md').write_text('\n'.join(text))
    print('Wrote SPM_M3_RESULTS.md')

if __name__=='__main__':main()
