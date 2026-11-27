from collections import Counter
from pathlib import Path
from hashlib import sha256
import json
R=Path(__file__).resolve().parent
read=lambda x:json.loads(x.read_bytes())
def pin(p):return dict(path=str(p.relative_to(R)),sha256=sha256(p.read_bytes()).hexdigest())
cfg=read(R/'handoff_registration.json'); results=read(R/'results.json'); assert len(results)<=len(cfg['cases'])
for v in results:
    c=R/f"case{v['case']:02}";t=[json.loads(l) for l in (c/'trace.jsonl').open()];assert len(t)==v['total_intervals'];assert [a['step'] for a in t]==list(range(1,len(t)+1)); assert all(abs((b['sim_s']-a['sim_s'])-.02)<.000045 for a,b in zip(t,t[1:]))
    h=v['handoff']; e=v['first_actual']; n=v['nav_ready']
    if h:assert h['owner_unchanged'] and h['world_preserved'] and h['history_unchanged'] and n['dwell_s']>=1. and min(n['ground_load_n'])>4.
    if e:assert h and e['step']==h['nav_end_k']+1 and e['previous_target_leg_rad']==n['target_leg_rad'] and t[h['nav_end_k']]['stage']=='STAIRS'
    if v['video']:assert v['video']['frames']==len(t)+1 and v['video']['sha256']==pin(c/'full.mp4')['sha256'] and 'decode-OK' in (R/'resources.txt').read_text()
    v['trace']=pin(c/'trace.jsonl');v['video_file']=str((c/'full.mp4').relative_to(R)) if v['video'] else None
acts=[x for x in results if x['first_actual']]; data=dict(registered=len(cfg['cases']),observed=len(results),nav_finished=sum(x['nav_ready'] is not None for x in results),stairs_first_applied=len(acts),ascent_old_rule_completed=sum(bool((x['stairs_final'] or {}).get('original_goal')) for x in acts),strict_no_strike_complete=sum(x['terminal']=='STRICT_NO_STRIKE' for x in acts),stairs_terminal_counts=dict(Counter(x['terminal'] for x in acts)),case0=results[0],cases=results,no_top_nav_dispatched=True,scope=cfg['scope'],weights={k:cfg[k] for k in ('nav_sha256','stairs_sha256')},source=[pin(R/q) for q in ('handoff.py','handoff_registration.json','stairs.py','nav_motor_bridge.py','fbi_nav.xml','network.py')])
(R/'delivery.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({k:data[k] for k in ('registered','observed','nav_finished','stairs_first_applied','ascent_old_rule_completed','strict_no_strike_complete','stairs_terminal_counts')},ensure_ascii=False))
