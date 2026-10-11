"""Source-backed M7 results, missingness, paired intervals and update evidence."""
from collections import Counter
import json
from statistics import mean
import spm_m7 as exp


def number(x,percent=False):
    return '—' if x is None else f'{x*100:.1f}%' if percent else f'{x:.3f}'
def interval(ci):return '—' if ci is None else '['+', '.join(f'{x:+.3f}' for x in ci)+']'


def main():
    lines=['# SPM M7：保守图更新对照','',
        '系统各角色 Qwen2.5-7B-Instruct；评审 DeepSeek-V3.2；嵌入 BGE-M3 API。', '',
        '主问题：在相同抽取、原文检索、读入与生成条件下，核对对象/维度并要求明确修订证据的保守图更新，是否改善原图版的偏好遵循与推断？'
        ' [预先冻结方案](SPM_M7_PROTOCOL.md)。', '',
        '开发 12 PrefEval + 6 CUPID；测试 48 PrefEval + 24 CUPID。每组一题，均未用于旧实验，开发与测试不重叠。两阶段源码、模型、配置及成员在首次 M7 预测前冻结。'
        ' 原图和平铺保留；本轮不根据开发分数调参或改默认策略。','']
    if (exp.BASE/'m7_test/verification.json').exists():
        lines += ['主测试已完成执行和一次有界失败续跑，194/216 项有效。共同有效的 48 个 PrefEval / 16 个 CUPID 问题上，原图 / 平铺 / 保守图分别为 50.0% / 45.8% / 45.8% 与 F1 0.267 / 0.293 / 0.329。'
            ' 保守减原图的主比较区间包含零，未确认稳定收益；保守版接近平铺的保留量与上下文长度，没有实际时间替换，暂不提升为默认。'
            ' [轨迹诊断](SPM_M7_DIAGNOSTICS.md)。','']
    totals=dict(planned=0,ok=0,http_attempts=0,known_tokens=0,unknown_usage=0)
    for phase in ('dev','test'):
        out=exp.BASE/f'm7_{phase}'
        if not (out/'verification.json').exists():continue
        v=exp.read(out/'verification.json');ledger=exp.read(out/'ledger_totals.json')
        for key in ('planned','ok'):totals[key]+=v[key]
        for key in ('http_attempts','known_tokens','unknown_usage'):totals[key]+=ledger[key]
        lines += [f'## {phase}', '',f"有效 {v['ok']}/{v['planned']} 项；三方法共同有效 {v['common_cases']} 个问题。",'']
        marker=out/'resume_authorization.json'
        if marker.exists():
            initial=len(exp.read(marker)['success_hashes'])
            lines += [f"首轮有效 {initial} 项；一次有界失败续跑增加 {v['ok']-initial} 项。首轮成功样本字节不变。",'']
        full=exp.read(out/'metrics.json');metrics=exp.read(out/'common_success_metrics.json')['metrics']
        lines += ['下表限于三方法共同有效子集，不替代官方全量基准；执行缺失单列。', '',
            '| 基准 / 方法 | 有效 / 计划 | 共同题数 | 遵循率或 F1 | CUPID 回答质量 |',
            '| --- | --- | ---: | ---: | ---: |']
        for benchmark in ('prefeval','cupid'):
            for method in exp.METHODS:
                f=full[f'{benchmark}/{method}'];r=metrics.get(f'{benchmark}/{method}',{});m=r.get('means_on_successes',{})
                key='follows_preference' if benchmark=='prefeval' else 'checklist_f1'
                lines.append(f"| {benchmark} / {method} | {f['ok']} / {f['attempted']} | {r.get('ok',0)} | {number(m.get(key),benchmark=='prefeval')} | {number(m.get('response_score'))} |")
        lines += ['', '保守图版的配对差值（每行两方法共同有效题；正值表示保守版更高）：', '',
            '| 基准 / 对照 | 配对题数 | 指标 | 保守减对照 | 配对 95% 区间 |', '| --- | ---: | --- | ---: | --- |']
        for baseline in ('v6_m6','flat_m6'):
            comparisons=exp.read(out/f'comparisons_vs_{baseline}.json')
            for benchmark in ('prefeval','cupid'):
                r=comparisons[f'{benchmark}/v6_m7_vs_{baseline}']
                for key,m in r['metrics'].items():
                    lines.append(f"| {benchmark} / {baseline} | {r['paired_cases']} | {key} | {m['delta']:+.3f} | {interval(m['paired_cluster_95ci'])} |")
        lines += ['', 'CUPID 原生分层（探索性）：', '',
            '| 分层 | 配对题数 | 保守减原图 F1 | 95% 区间 |', '| --- | ---: | ---: | --- |']
        for label,comparisons in exp.read(out/'cupid_stratum_comparisons.json').items():
            r=comparisons['cupid/v6_m7_vs_v6_m6'];m=r['metrics'].get('checklist_f1',{})
            lines.append(f"| {label} | {r['paired_cases']} | {number(m.get('delta'))} | {interval(m.get('paired_cluster_95ci'))} |")
        if phase=='test':
            planned_strata=Counter(c['stratum'] for c in exp.read(exp.BASE/f'm7_{phase}_plan.json')['cases'] if c['benchmark']=='cupid')
            strata=exp.read(out/'cupid_stratum_comparisons.json')
            coverage='、'.join(f"{label} {strata[label]['cupid/v6_m7_vs_v6_m6']['paired_cases']}/{planned_strata[label]}" for label in ('changing','consistent','contrastive'))
            lines += ['', f'CUPID 各层共同成功 / 计划题数为 {coverage}。consistent 大量缺失，均值的分层权重已改变，不能将当前子集成绩当作原计划测试的全量效果。','']
        lines += ['', '同题效率与候选保留（共同有效子集）：', '',
            '| 基准 / 方法 | 平均上下文字符 | 平均候选数 | 平均 apply 数 | 中位案例秒数 |',
            '| --- | ---: | ---: | ---: | ---: |']
        for b,methods in exp.read(out/'matched_efficiency.json').items():
            for method,e in methods.items():
                lines.append(f"| {b} / {method} | {number(e['mean_context_chars'])} | {number(e['mean_candidates'])} | {number(e['mean_applied'])} | {number(e['median_case_seconds'])} |")
        audit=exp.read(out/'manager_control_audit.json')
        case_benchmark={c['id']:c['benchmark'] for c in exp.read(exp.BASE/f'm7_{phase}_plan.json')['cases']}
        lines += ['', '同题记忆节点保留：数量只说明保留程度，不能证明保留了正确或任务相关的信息。','',
            '| 基准 / 方法 | 平均未失效节点 | 平均 pending | 平均已失效节点 |', '| --- | ---: | ---: | ---: |']
        for benchmark in ('prefeval','cupid'):
            subset=[a for a in audit if case_benchmark[a['case_id']]==benchmark]
            for method in exp.METHODS:
                counts=[a['units'][method] for a in subset]
                live=mean(sum(c.get(k,0) for k in ('active','candidate','pending')) for c in counts) if counts else None
                pending_mean=mean(c.get('pending',0) for c in counts) if counts else None
                superseded=mean(c.get('superseded',0) for c in counts) if counts else None
                lines.append(f'| {benchmark} / {method} | {number(live)} | {number(pending_mean)} | {number(superseded)} |')
        ops=exp.read(out/'conservative_operations.json');actions=Counter()
        for o in ops:actions.update(o['actions'])
        rejected=sum(o['rejected_relations'] for o in ops)
        lines += ['', f"保守版成功记忆快照的事件总计 `{dict(actions)}`；核对后拒绝 {rejected} 次潜在破坏性关系判断。每次实际 supersede 都检查了明确修订、对象/维度及原始 USER 引文。"
            ' 这些是算法与来源审计，不等于独立确认关系语义正确。','']
        if not actions.get('supersede'):
            lines += ['该阶段保守版没有实际时间替换，因此这些原生样本尚不能检验其真实更新召回。局部测试覆盖了明确修订触发，但不能替代基准中的时间更新效果证据。','']
        pending=exp.read(out/'pending.json');failures=Counter(r['error']['message'].split(':')[0] for r in pending)
        check=ledger['by_tag'].get('update_check',{})
        lines += [f"最终缺失 {len(pending)} 项，前缀 `{dict(failures)}`。缺失未计为答错，未人工代替评分。", '',
            f"HTTP {ledger['http_attempts']} 次；已知 provider tokens {ledger['known_tokens']:,}；未知用量 {ledger['unknown_usage']} 次；状态 `{ledger['statuses']}`。", '',
            f"更新核对 {check.get('http_attempts',0)} 次 HTTP / {check.get('known_tokens',0):,} 已知 tokens，包含有限 schema 修复。共享缓存下的方法调用归属不能视为独立冷启动成本。", '',
            f"[核验](runs/spm/m7_{phase}/verification.json)、[主比较](runs/spm/m7_{phase}/comparisons_vs_v6_m6.json)、"
            f"[控制审计](runs/spm/m7_{phase}/manager_control_audit.json)、[更新核对轨迹](runs/spm/m7_{phase}/conservative_operations.json)、"
            f"[同题效率](runs/spm/m7_{phase}/matched_efficiency.json)、[缺失](runs/spm/m7_{phase}/pending.json)、[用量](runs/spm/m7_{phase}/ledger_totals.json)。",'']
        if phase=='test':
            lines += ['22 项缺失来自六个问题的共享抽取错误（18 项）、cupid/337 的三方法分析错误（3 项），以及保守版 cupid/622 的修订引文校验错误（1 项）。'
                ' 保守版完成 16/24 CUPID，其他两方法 17/24；其额外缺失可能影响质量比较。失败续跑未恢复任何新增结果，后续修复需另立实验，不能改写本轮冻结成绩。','']
        replay=out/'replay_verification.json'
        if replay.exists():
            r=exp.read(replay)
            lines += [f"离线完整重建 {r['contexts_reconstructed']} 个上下文与 trace，回放 {r['evaluations_replayed']} 个原生成和评分，HTTP {r['http_requests']}；样本/记忆快照字节不变。"
                f" [回放核验](runs/spm/m7_{phase}/replay_verification.json)。这是可复现性验证，不是独立重评。",'']
        write_replay=out/'write_replay_verification.json'
        if write_replay.exists():
            r=exp.read(write_replay)
            lines += [f"补充离线写入回放从空临时库重建 {r['successful_conservative_banks_rebuilt']} 个保守记忆库，最终节点与事件快照完全一致，HTTP 为零，原记忆文件不变。"
                f" 同时保存 {r['cached_update_pairs']} 个核对时的真实输入语义/引文（包含预计算后未使用的比较），用于后续诊断。"
                f" [写入回放核验](runs/spm/m7_{phase}/write_replay_verification.json)。这是额外诊断，未改动预测。",'']
    lines += ['## 结论与边界','',
        f"已核验阶段共 {totals['ok']}/{totals['planned']} 项有效；{totals['http_attempts']} 次 HTTP，{totals['known_tokens']:,} 已知 tokens，{totals['unknown_usage']} 次未知用量。", '',
        '82 项实现测试通过。开发/测试使用相同冻结源码；旧 2208 个样本摘要逐文件核验，默认仍为 v6_compact。', '',
        '新增策略同时约束语义合并和时间替换，并增加同模型证据核对，测量的是整套策略，而非单独某条规则或纯计算量相等的效果。'
        ' 核对的语义判断仍来自 Qwen 7B；字面引文验证只保证来源，不保证推断正确。弱冲突保留为 pending，不因重复两次自动淘汰旧观察，可能降低真实更新的召回。', '',
        '保守版若接近平铺保留数量，收益可能来自减少删除/合并，不能据此声称图结构本身贡献了增益；仍需查看 vs 平铺的对照与成本。','',
        '主比较为保守 vs 原图；vs 平铺、回答质量、分层为次要或探索性，区间未作多重比较校正。'
        ' 裁判是已有代理协议，非官方 PrefMatcher；缺失可能非随机，小样本子集不代表全量。并发时延包含 TPM 等待、共享写入与缓存。','']
    if (exp.BASE/'m7_test/verification.json').exists():
        comparisons=exp.read(exp.BASE/'m7_test/comparisons_vs_v6_m6.json')
        for benchmark,key in [('prefeval','follows_preference'),('cupid','checklist_f1')]:
            r=comparisons[f'{benchmark}/v6_m7_vs_v6_m6'];m=r['metrics'].get(key)
            if not m:lines += [f'{benchmark} 无有效配对，不能估计收益。',''];continue
            ci=m['paired_cluster_95ci'];statement='区间跨零或接触零，尚未确认稳定收益。'
            if ci and ci[0]>0:statement='该有效子集配对区间为正，仍须结合缺失与绝对成绩判断。'
            elif ci and ci[1]<0:statement='该有效子集保守版低于原图，需要诊断过度保留或抑制。'
            lines += [f"{benchmark}：{r['paired_cases']} 配对，保守减原图 {m['delta']:+.3f}，95% 区间 {interval(ci)}。{statement}",'']
    else:lines += ['主测试尚未完成，暂不作测试效果结论。','']
    if (exp.BASE/'m7_test/verification.json').exists():
        lines += ['下一轮优先简化小模型输出协议，修复枚举、过长内容与实体绑定失败，再验证显式对象/维度编码与更新证据绑定。'
            ' 当前额外核对既没有建立稳定质量增益，又使 CUPID 运行时延和成本增加；不继续堆叠核对模型或直接引入强化学习。'
            ' 新方案应另选问题组，区分真实更新、互补观察和生成器忽视规则，并同时保留平铺基线。','']
    (exp.ROOT/'SPM_M7_RESULTS.md').write_text('\n'.join(lines));print(json.dumps(totals))


if __name__=='__main__':main()
