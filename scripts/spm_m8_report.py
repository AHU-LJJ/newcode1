"""M8 CUPID-only source-backed report; no API calls or independent regrading."""
from collections import Counter
import json
import spm_m8 as exp


def num(x):return '—' if x is None else f'{x:.3f}'
def ci(x):return '—' if x is None else '['+', '.join(f'{a:+.3f}' for a in x)+']'


def main():
    lines=['# M8：CUPID 对象与维度图更新','',
        'PrefEval 已退出后续实验。系统模型 Qwen2.5-7B-Instruct；嵌入 BGE-M3 API；裁判 DeepSeek-V3.2，沿用 API 代理评测。',
        '', '[预先冻结方案](SPM_M8_PROTOCOL.md)。开发 6 题、测试 24 题，各原生层分别 2 / 8 题；每 persona 一题，排除全部旧样本及预留组。 [开发来源诊断](SPM_M8_DIAGNOSTICS.md)。', '',
        '三种管理方法共用整数标签抽取、原文编号、对象/维度字段及查询提及编号。新图只对同轴对象分类一次，允许源自直接陈述/反馈/事件的隐含偏好转变；不使用 M7 第二轮核对。', '',
        '新增键可能受同义词或编码误判影响；字面来源正确不证明变更判断正确。原图与新图均使用新编码，本轮不单独估计协议改善，也不与不同题组 M7 均值直接比较。','']
    if (exp.BASE/'m8_test/verification.json').exists():
        lines += ['主测试完成 93/120 项，五方法共同有效 15/24 题。共同子集原图 / 平铺 / 对象维度图的 F1 为 0.525 / 0.377 / 0.456。新图减平铺的主比较为 +0.078，95% 区间 [-0.033, +0.194]，尚未确认稳定收益。', '',
            '原图减平铺的探索性比较为 +0.148，95% 区间 [+0.031, +0.275]；这是未作多重比较校正的次要结果，且缺失可能非随机，需要独立复验。默认保持 v6_compact。', '',
            '新图实际 3 次替换全部发生在同一个 changing 问题，存在将助手草稿修订误认为用户偏好转变的迹象。M9 修复接口与关系提示已通过本地验证，尚未预留或调用 API。', '']
    totals=Counter()
    for phase in ('dev','test'):
        out=exp.BASE/f'm8_{phase}'
        if not (out/'verification.json').exists():
            lines += [f'## {phase}', '', '尚未完成核验；不报告临时均值。',''];continue
        v=exp.read(out/'verification.json');ledger=exp.read(out/'ledger_totals.json')
        for k in ('planned','ok','failed'):totals[k]+=v[k]
        for k in ('http_attempts','known_tokens','unknown_usage'):totals[k]+=ledger[k]
        lines += [f'## {phase}', '',f"有效 {v['ok']}/{v['planned']} 方法项，五方法共同有效 {v['common_cases']} 题。",'']
        marker=out/'resume_authorization.json'
        if marker.exists():
            n=len(exp.read(marker)['success_hashes']);lines += [f"首轮 {n} 项成功；一次有界失败续跑增加 {v['ok']-n} 项。首轮成功样本字节未变。",'']
        else:lines += ['首轮已完成，无失败续跑。','']
        full=exp.read(out/'metrics.json');common=exp.read(out/'common_success_metrics.json')
        lines += ['下面均值限于五方法共同有效子集，不代表官方全量基准。', '',
            '| 方法 | 有效 / 计划 | 共同题数 | Checklist F1 | 回答质量 |', '| --- | --- | ---: | ---: | ---: |']
        for m in exp.METHODS:
            f=full[f'cupid/{m}'];r=common['metrics'].get(f'cupid/{m}',{});means=r.get('means_on_successes',{})
            lines.append(f"| {m} | {f['ok']} / {f['attempted']} | {r.get('ok',0)} | {num(means.get('checklist_f1'))} | {num(means.get('response_score'))} |")
        lines += ['', '主比较为 typed_m8 减 flat_m8 的 F1；以下其余指标与对照为次要或探索性。每行使用两方法共同有效题，题数可能不同于上表。','',
            '| 对照 | 配对题数 | 指标 | 新图减对照 | 配对 95% 区间 |', '| --- | ---: | --- | ---: | --- |']
        primary=None
        for baseline in ('flat_m8','v6_m8','raw_m8','b1'):
            comparisons=exp.read(out/f'comparisons_vs_{baseline}.json');r=comparisons[f'cupid/typed_m8_vs_{baseline}']
            for key,m in r['metrics'].items():
                lines.append(f"| {baseline} | {r['paired_cases']} | {key} | {m['delta']:+.3f} | {ci(m['paired_cluster_95ci'])} |")
            if baseline=='flat_m8':primary=r
        if primary and 'checklist_f1' in primary['metrics']:
            m=primary['metrics']['checklist_f1'];bounds=m['paired_cluster_95ci']
            verdict=('区间包含或接触零，尚未确认稳定质量增益。' if bounds and bounds[0]<=0<=bounds[1] else
                '区间在零以上，当前配对子集支持正向差值；仍需核对缺失、成本与后续独立复验。' if bounds and bounds[0]>0 else
                '区间在零以下，当前配对子集支持新图更低；不提升为默认。' if bounds else '没有可计算区间。')
            lines += ['',f"主比较 {primary['paired_cases']} 题，F1 差值 {m['delta']:+.3f}，95% 区间 {ci(bounds)}。{verdict}",'']
        if phase=='test':
            r=exp.read(out/'comparisons_vs_flat_m8.json')['cupid/v6_m8_vs_flat_m8']
            m=r['metrics']['checklist_f1']
            lines += [f"原图减平铺的探索性比较：{r['paired_cases']} 题，F1 差值 {m['delta']:+.3f}，95% 区间 {ci(m['paired_cluster_95ci'])}。未作多重比较校正；不据此推广到全部问题或其他数据集。", '']
        strata=exp.read(out/'cupid_stratum_comparisons.json');plan=exp.read(exp.BASE/f'm8_{phase}_plan.json')
        planned=Counter(c['stratum'] for c in plan['cases']);case_stratum={c['id']:c['stratum'] for c in plan['cases']}
        shared=Counter(case_stratum[c] for c in common['case_ids'])
        lines += ['| 原生层 | 五方法共同 / 计划 | 新图与平铺配对 | F1 差值 | 95% 区间 |', '| --- | --- | ---: | ---: | --- |']
        for s in ('changing','consistent','contrastive'):
            r=strata[s]['cupid/typed_m8_vs_flat_m8'];m=r['metrics'].get('checklist_f1',{})
            lines.append(f"| {s} | {shared[s]} / {planned[s]} | {r['paired_cases']} | {num(m.get('delta'))} | {ci(m.get('paired_cluster_95ci'))} |")
        lines += ['', '同题上下文、保留量和并发案例时间：时间包含共享缓存/写入与 TPM 等待，不是独立冷启动速度。','',
            '| 方法 | 平均上下文字符 | 平均未失效节点 | 平均候选 | 中位案例秒 |', '| --- | ---: | ---: | ---: | ---: |']
        for m,e in exp.read(out/'matched_efficiency.json').items():
            lines.append(f"| {m} | {num(e.get('mean_context_chars'))} | {num(e.get('mean_live_units'))} | {num(e.get('mean_candidates'))} | {num(e.get('median_case_seconds'))} |")
        ops=exp.read(out/'typed_operations.json');actions=Counter();replacements=Counter()
        for o in ops:actions.update(o['actions']);replacements[o['stratum']]+=len(o['replacements'])
        lines += ['', f"成功新图库的事件总计 `{dict(actions)}`。实际替换按层 `{dict(replacements)}`。", '',
            ('成功样本中没有实际替换，无法声称已验证更新召回。' if not sum(replacements.values()) else
             '实际替换均通过对象/维度/条件、当前 USER 来源与因果顺序审计；这不等于独立语义标签验证或真实更新召回率。'), '']
        if phase=='test':
            lines += ['这 3 次 supersede 全在 cupid/53：日期优先整理照片、坚定指令语气两个轴中，用户认可更符合原偏好的修订被标成变更。时间与来源校验通过并不能证明用户想要的值改变了。该题平铺与新图 F1 均 0.857，原图 1.000；没有单独更新带来收益的证据。这是轨迹诊断，未修改原评分。详见 [诊断](SPM_M8_DIAGNOSTICS.md)。', '']
        pending=exp.read(out/'pending.json');prefix=Counter(p['error']['message'].split(':')[0] for p in pending)
        lines += [f"缺失 {len(pending)} 项，错误前缀 `{dict(prefix)}`；不计为答错，不手工补评。", '',
            f"HTTP {ledger['http_attempts']} 次，provider 已知 tokens {ledger['known_tokens']:,}，未知用量 {ledger['unknown_usage']} 次。状态 `{ledger['statuses']}`。", '',
            '| API 标签 | HTTP 次数 | 已知 tokens | 未知用量 |', '| --- | ---: | ---: | ---: |']
        for tag,e in sorted(ledger['by_tag'].items()):lines.append(f"| {tag} | {e['http_attempts']} | {e['known_tokens']:,} | {e['unknown_usage']} |")
        if phase=='test':
            g=ledger['by_tag']['gate_complete']
            lines += ['', '27 项缺失中，21 项为抽取失败（15 项 nature 越界、6 项 USER 来源越界），4 项为查询实体编号类型错误，1 项为裁判覆盖不足，1 项为关系变更缺少有效来源。这些是执行缺失，不计为答错。', '',
                f"逐项门控补齐 gate_complete 共 {g['http_attempts']} 次 HTTP，已知 tokens {g['known_tokens']:,}，占主测试已知 tokens {g['known_tokens']/ledger['known_tokens']:.1%}。M9 将三管理方法的门控改为宿主按位置绑定的完整概率数组，以另轮实验核查效率与质量。", '']
        lines += ['',f'[核验](runs/spm/m8_{phase}/verification.json)、[主比较](runs/spm/m8_{phase}/comparisons_vs_flat_m8.json)、[控制审计](runs/spm/m8_{phase}/manager_control_audit.json)、[管理事件](runs/spm/m8_{phase}/typed_operations.json)、[缺失](runs/spm/m8_{phase}/pending.json)、[用量](runs/spm/m8_{phase}/ledger_totals.json)。','']
        literal=out/'literal_copy_diagnostics.json'
        if literal.exists():
            d=exp.read(literal)
            lines += [f"核对输入的字面检查另发现 {d['flagged']} 条去重后的跨 USER turn 复制错绑迹象。合法原文编号不能证明其支持观察语义或时间顺序；没有独立语义重评。 [来源诊断](SPM_M8_DIAGNOSTICS.md)。",'']
        replay=out/'replay_verification.json'
        if replay.exists():
            r=exp.read(replay);assert r['http_requests']==0
            lines += [f"离线精确重建 {r['contexts_reconstructed']} 个成功上下文/trace，回放 {r['evaluations_replayed']} 个原生成与评分，0 HTTP；样本与记忆 JSON 不变。不是独立重评。 [回放核验](runs/spm/m8_{phase}/replay_verification.json)。",'']
    lines += ['## 边界与默认策略','',
        f"已核验阶段共 {totals['ok']}/{totals['planned']} 有效；HTTP {totals['http_attempts']} 次，已知 tokens {totals['known_tokens']:,}，未知用量 {totals['unknown_usage']} 次。", '',
        'M8 冻结前 94 项实现测试通过；后续 M9 接口实现使本地测试增至 120 项并通过。旧 2478 个样本保持不变；M8 使用自己的冻结源码、配置和评分。默认仍为 Qwen2.5-7B + v6_compact。缺失可能非随机；分层与多个次要比较未经多重比较校正。代理裁判沿用既有协议，非官方 PrefMatcher，不把当前小样本当作完整基准。', '',
        '对象/维度键和关系语义均来自小模型，可能出现同义键分裂或错误归类；隐含变更可能误替换。需要结合真实事件轨迹和简单基线判断图的价值。历史 PrefEval 仅保留，不再用来选择后续方向。','']
    (exp.ROOT/'SPM_M8_RESULTS.md').write_text('\n'.join(lines));print(json.dumps(dict(totals)))


if __name__=='__main__':main()
