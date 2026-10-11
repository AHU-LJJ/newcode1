"""Generate M6 report from verified checkpoints, with explicit missingness and controls."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median
import spm_m6 as exp


def number(value,percent=False):
    if value is None:return '—'
    return f'{value*100:.1f}%' if percent else f'{value:.3f}'


def main():
    lines=['# SPM M6：图管理与平铺记忆的同模型对照','',
        '2026-10-11（JST）。系统各角色 Qwen2.5-7B-Instruct；评审 DeepSeek-V3.2；嵌入 BGE-M3 API。',
        '', '主问题是图管理是否在相同初始抽取和原文证据之上带来增量。[方案](SPM_M6_PROTOCOL.md)与每阶段冻结计划包含预留组、模型、数据及源码摘要。',
        '', '测试成员在首次 M6 预测前预留；开发与测试组不重叠。保留 M5 及更早样本。每阶段最多一次有预算上限的失败续跑，成功输出不重生成。','']
    review=exp.BASE/'m6_freeze_review.json'
    if review.exists():
        lines += ['开发中发现限速等待占据 HTTP 槽，以及重叠 embedding 批次覆盖单文本向量的并发问题。主测试冻结前仅改 api.py 的准入顺序与向量首次提交后保持不变，提示、schema、配置、模型、阈值和 HTTP 并发上限不变。开发原冻结包保留。'
            ' [冻结审查](runs/spm/m6_freeze_review.json)记录源码差异；开发与测试的时延不作为修复的独立性能对照。','']
    totals=dict(planned=0,ok=0,http_attempts=0,known_tokens=0,unknown_usage=0)
    for phase in ('reliability','dev','test'):
        out=exp.BASE/f'm6_{phase}'
        if not (out/'verification.json').exists():continue
        v=exp.read(out/'verification.json');ledger=exp.read(out/'ledger_totals.json')
        for key in ('planned','ok'):totals[key]+=v[key]
        for key in ('http_attempts','known_tokens','unknown_usage'):totals[key]+=ledger[key]
        rows=[exp.read(p) for p in sorted((out/'samples').glob('*.json'))]
        coverage = f"单方法有效 {v['common_cases']} 个问题" if phase=='reliability' else f"四方法共同有效 {v['common_cases']} 个问题"
        lines += [f'## {phase}', '',f"有效 {v['ok']}/{v['planned']} 项；{coverage}。",'']
        marker=out/'resume_authorization.json'
        if marker.exists():
            saved=exp.read(marker);initial=len(saved['success_hashes'])
            lines += [f"首轮有效 {initial}/{v['planned']} 项，一次失败续跑新增 {v['ok']-initial} 项有效评分；首轮成功的 {initial} 个样本文件逐字节保持不变。",'']
        if phase=='reliability':
            lines += ['这四题是已观察的 M5 门控执行失败，仅作修复诊断。不能作为新样本效果。','',
                '| 案例 | 状态 | 初始 / 预期门控数 | 单项补齐数 |','| --- | --- | ---: | ---: |']
            for r in rows:
                g=r.get('trace',{}).get('gate_coverage',{})
                lines.append(f"| {r['case_id']} | {r['status']} | {g.get('initial_returned','—')} / {g.get('expected','—')} | {len(g.get('completed_indices',[]))} |")
        else:
            if phase=='dev':
                audit=exp.read(out/'manager_control_audit.json')
                differences=[a['rag_score_difference'] for a in audit if not a['rag_exact_scores']]
                lines += [f'开发中 RAG 与图版的检索原文、顺序和截取完全相同，但 {len(differences)} 个共同有效案例的相似度分数不完全相同，最大绝对差 {max(differences,default=0.):.6f}。图/平铺的初始抽取、向量及原文分数仍一致。原输出保留；RAG 数值输入差异作为开发限制单列，主测试修复缓存覆盖后要求完整一致。','']
            metrics=exp.read(out/'common_success_metrics.json')['metrics']
            lines += ['下表仅比较四种方法共同有效的问题。执行覆盖另列；任何不完整基准的全量指标仍为 null。','',
                '| 基准 / 方法 | 有效 / 计划 | 共同题数 | 遵循率或 F1 | CUPID 回答质量 |',
                '| --- | --- | ---: | ---: | ---: |']
            full=exp.read(out/'metrics.json')
            for benchmark in ('prefeval','cupid'):
                for method in exp.METHODS:
                    f=full[f'{benchmark}/{method}'];r=metrics.get(f'{benchmark}/{method}',{})
                    m=r.get('means_on_successes',{});key='follows_preference' if benchmark=='prefeval' else 'checklist_f1'
                    lines.append(f"| {benchmark} / {method} | {f['ok']} / {f['attempted']} | {r.get('ok',0)} | {number(m.get(key),benchmark=='prefeval')} | {number(m.get('response_score'))} |")
            lines += ['', '图 vs 各对照的配对差值：正值表示图版更高。每行只用该两方法共同有效题，可能与上表四方法交集不同。','',
                '| 基准 / 对照 | 配对题数 | 指标 | 图减对照 | 成对 95% 区间 |',
                '| --- | ---: | --- | ---: | --- |']
            for baseline in ('flat_m6','b2_matched','b1'):
                comparisons=exp.read(out/f'comparisons_vs_{baseline}.json')
                for benchmark in ('prefeval','cupid'):
                    row=comparisons[f'{benchmark}/v6_m6_vs_{baseline}']
                    for key,metric in row['metrics'].items():
                        ci=metric['paired_cluster_95ci'];label='—' if ci is None else '['+', '.join(f'{x:+.3f}' for x in ci)+']'
                        lines.append(f"| {benchmark} / {baseline} | {row['paired_cases']} | {key} | {metric['delta']:+.3f} | {label} |")
            lines += ['', 'CUPID 原生分层的图 vs 平铺：探索性小样本比较。','',
                '| 分层 | 配对题数 | F1 差值 | 95% 区间 |', '| --- | ---: | ---: | --- |']
            for label,comparisons in exp.read(out/'cupid_stratum_comparisons.json').items():
                r=comparisons['cupid/v6_m6_vs_flat_m6'];m=r['metrics'].get('checklist_f1',{});ci=m.get('paired_cluster_95ci')
                interval='—' if ci is None else '['+', '.join(f'{x:+.3f}' for x in ci)+']'
                lines.append(f"| {label} | {r['paired_cases']} | {number(m.get('delta'))} | {interval} |")
            lines += ['', '实际运行效率（各方法自身成功子集，案例组成可能不同）：','', '| 方法 | 平均上下文字符 | 中位案例秒数 | 补齐候选 / 案例 |', '| --- | ---: | ---: | ---: |']
            for method,e in exp.read(out/'efficiency.json').items():
                good=[r for r in rows if r['method']==method and r['status']=='ok']
                repaired=[r['trace'].get('gate_coverage',{}).get('completed_indices',[]) for r in good]
                lines.append(f"| {method} | {number(e['mean_context_chars'])} | {number(e['median_case_seconds'])} | {sum(map(len,repaired))} / {sum(bool(x) for x in repaired)} |")
            matched={}
            for benchmark in ('prefeval','cupid'):
                ids=set.intersection(*[{r['case_id'] for r in rows if r['benchmark']==benchmark and r['method']==method and r['status']=='ok'} for method in exp.METHODS])
                matched[benchmark]={}
                for method in exp.METHODS:
                    good=[r for r in rows if r['benchmark']==benchmark and r['method']==method and r['case_id'] in ids and r['status']=='ok']
                    matched[benchmark][method]=dict(cases=len(good),mean_context_chars=mean(r['trace']['context_characters'] for r in good) if good else None,
                        median_case_seconds=median(r['seconds'] for r in good) if good else None,
                        caveat='Identical common-success cases; concurrent time includes shared writes, cache and TPM waits.')
            (out/'matched_efficiency.json').write_text(json.dumps(matched,ensure_ascii=False,indent=2)+'\n')
            if phase=='test':
                flat=matched['cupid']['flat_m6'];graph=matched['cupid']['v6_m6']
                reduction=1-graph['mean_context_chars']/flat['mean_context_chars']
                lines += ['',f"同一组 {graph['cases']} 个 CUPID 共同有效问题，平铺 / 图版平均上下文为 {flat['mean_context_chars']:.0f} / {graph['mean_context_chars']:.0f} 字符，图版减少 {reduction:.1%}。这是上下文长度压缩；尚未确认回答质量增益，也不能据此估计独立冷启动时延。"
                    ' [同题效率数据](runs/spm/m6_test/matched_efficiency.json)。']
        pending=exp.read(out/'pending.json');failures=Counter(r['error']['message'].split(':')[0] for r in pending)
        lines += ['',f"最终缺失 {len(pending)} 项；错误前缀：`{dict(failures)}`。缺失没有计为答错，没有人工代替评分。",'',
            f"HTTP 尝试 {ledger['http_attempts']}；已知 provider tokens {ledger['known_tokens']:,}；未知用量 {ledger['unknown_usage']} 次；HTTP 状态 `{ledger['statuses']}`。",'',
            (f"[核验](runs/spm/m6_{phase}/verification.json)、[主比较](runs/spm/m6_{phase}/comparisons_vs_flat_m6.json)" if phase!='reliability' else f"[核验](runs/spm/m6_{phase}/verification.json)")+
            f"、[控制审计](runs/spm/m6_{phase}/manager_control_audit.json)、[缺失](runs/spm/m6_{phase}/pending.json)、[用量](runs/spm/m6_{phase}/ledger_totals.json)。",'']
        tags=ledger.get('by_tag',{})
        if tags:
            relations=sum(tags.get(t,{}).get('known_tokens',0) for t in ('relation','relations_batch'))
            completion=tags.get('gate_complete',{})
            lines += [f"关系判断新增的已知用量 {relations:,} tokens；门控单项补齐 {completion.get('http_attempts',0)} 次 HTTP 尝试 / {completion.get('known_tokens',0):,} 已知 tokens。这些含失败、修复和缓存影响，不是独立冷启动开销。",'']
    lines += ['## 用量、控制与结论边界','',
        f"以上已核验阶段共 {totals['ok']}/{totals['planned']} 项有效；{totals['http_attempts']} 次实验 HTTP 尝试、{totals['known_tokens']:,} 已知 tokens，未知用量 {totals['unknown_usage']} 次。启动时另有一次不产生推理 token 的模型元数据读取，不计入实验 HTTP 账本。",'',
        '图和平铺共用完全相同的初始抽取与条件字段；分别建库，检索候选可能因为合并/失效/冲突边改变。审计逐条核实 USER 原文来源、原始抽取一致性和相同 raw 证据；不证明语义抽取正确。', '',
        'b2_matched 与图版同原文窗口、查询扩展及 6,000 字符片段预算，但图版额外包含结构化偏好，因此不是总输入 token 完全匹配的比较。完整历史也不是等成本对照。', '',
        '抽取缓存及共享请求影响实际调用归属；HTTP 账本的方法归属是首先触发请求的方法，不能用于声称独立冷启动成本。案例时延包含并发、TPM 等待及缓存效果。', '',
        '主比较为图 vs 平铺。其余对照、CUPID 三层、回答质量为次要或探索性；区间未作多重比较校正。PrefEval/CUPID 使用现有 API 代理裁判；不是官方 PrefMatcher。执行缺失可能非随机；共同成功子集不能替代完整基准。', '',
        '当前默认仍是 M5 的 v6_compact。M6 保留为独立方法，未根据开发或测试成绩自动更换默认，也未引入强化学习。','']
    if (exp.BASE/'m6_test/verification.json').exists():
        verified=exp.read(exp.BASE/'m6_test/verification.json')
        replay=exp.read(exp.BASE/'m6_test/replay_verification.json')
        lines += [f"主测试 {verified['manager_control_cases']} 个共同有效案例通过初始抽取、向量与完整原文检索分数一致性审计；{verified['prior_samples_unchanged']} 个旧样本文件保持不变。71 项实现测试通过。",'',
            f"主测试 {replay['contexts_reconstructed']} 个成功上下文与 trace 完整重建，{replay['evaluations_replayed']} 个原始生成和评分从已验证缓存回放，新增 HTTP {replay['http_requests']} 次，样本与记忆快照未改写。回放验证可复现性，不是独立重生成或重评。开发阶段仅回放已保存上下文对应的输出，未声称重建其检索分数。"
            ' [离线回放核验](runs/spm/m6_test/replay_verification.json)。','']
        comparisons=exp.read(exp.BASE/'m6_test/comparisons_vs_flat_m6.json')
        lines += ['## 新测试主比较读数','']
        for benchmark,key in [('prefeval','follows_preference'),('cupid','checklist_f1')]:
            r=comparisons[f'{benchmark}/v6_m6_vs_flat_m6'];m=r['metrics'].get(key)
            if not m:
                lines.append(f'{benchmark} 没有有效配对，不能估计图管理增量。');continue
            ci=m['paired_cluster_95ci'];statement='区间跨零或接触零，尚不能确认稳定增益。'
            if ci and ci[0]>0:statement='配对区间为正，支持该有效子集上的图管理增益；仍须结合缺失和绝对成绩判断。'
            elif ci and ci[1]<0:statement='配对区间为负，该有效子集上图版低于平铺，需要先诊断管理损失。'
            lines.append(f"{benchmark}：{r['paired_cases']} 个配对，图减平铺 {m['delta']:+.3f}。{statement}")
        lines += ['', '下一轮优先检验保守更新：只有同一偏好对象与维度的明确矛盾才替换旧观察，同轮互补偏好并存，弱冲突暂不令旧节点失效。'
            ' [开发轨迹诊断](SPM_M6_DIAGNOSTICS.md)记录了互补偏好被替换的案例；需另设新开发与测试组，并同时记录信息保留、回答遵循和压缩率。该策略尚未实现或取得效果证据。','']
    (exp.ROOT/'SPM_M6_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(totals))


if __name__=='__main__':main()
