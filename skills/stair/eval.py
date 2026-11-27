import os,json,random,sys,subprocess,traceback
from pathlib import Path
from hashlib import sha256
import numpy as np
import torch,mujoco as mj
from stairs import Stairs
from train import Net,REFERENCE,LEG_SHA,digest,atomic
R=Path(__file__).resolve().parent

@torch.no_grad()
def evaluate(checkpoint,out):
    cfg=json.loads((R/'registration.json').read_bytes());randomizer=random.Random(int(os.environ.get('FBI_STAIR_EVAL_SEED',cfg['eval_seed']))); cases=[[-randomizer.uniform(.45,.85),randomizer.uniform(-.18,.18),randomizer.uniform(-.15,.15),randomizer.uniform(.60,.94),randomizer.uniform(2.65,2.81)] for _ in range(cfg['heldout'])];out.mkdir(); model=Net();
    if checkpoint is None: assert digest(REFERENCE)==LEG_SHA;model.init_base()
    else:
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)['actor'];n=weights['0.weight'].shape[-1]
        assert n in (326,334); expanded=torch.zeros_like(model.actor[0].weight);expanded[:,:n]=weights['0.weight'];weights['0.weight']=expanded
        model.actor.load_state_dict(weights)
    model=model.cuda().eval(); torch.set_num_threads(1)
    n=len(cases); env=Stairs(R,n,cfg['eval_seed'],cases); result={}; active=set(range(n));
    for _ in range(1105):
        _,_,d,z=env.step(model.actor(env.obs)); done=d.nonzero().flatten().tolist()
        for j in active.intersection(done):result[j]=row(z,j)
        active.difference_update(done)
        if not active:break
    for j in active:result[j]={'terminal':'UNOBSERVED'}
    summary=dict(weights=digest(checkpoint) if checkpoint else LEG_SHA,cases=cases,results=result,strict=sum(x['terminal']=='SUCCESS' for x in result.values()),n=n)
    atomic(out/'evaluation.json',summary); print('EVAL',out.name,summary['strict'],n,flush=True)
    env=Stairs(R,1,cfg['eval_seed'],[cases[cfg['video_case_index']]]); m=env.m; d=mj.MjData(m); r=mj.Renderer(m,448,832); cam=mj.MjvCamera(); cam.azimuth=107;cam.elevation=-17;cam.distance=5.1;cam.lookat[:]=[.85,0,.48]
    vid=out/'full.mp4'; proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','832x448','-r','50','-i','-','-c:v','libx264','-threads','2','-preset','veryfast','-pix_fmt','yuv420p','-movflags','+faststart',str(vid)],stdin=subprocess.PIPE); frames=0; last={'terminal':'START'}
    trace=(out/'case0_trace.jsonl').open('x',buffering=1)
    def paint():
        nonlocal frames
        d.qpos[:]=env.q[0].cpu().numpy(); d.qvel[:]=env.v[0].cpu().numpy();mj.mj_kinematics(m,d);mj.mj_comPos(m,d);r.update_scene(d,cam);proc.stdin.write(r.render().tobytes());frames+=1
    try:
        paint()
        for _ in range(1105):
            _,_,finished,z=env.step(model.actor(env.obs),auto_reset=False); last=row(z,0);trace.write(json.dumps(last)+'\n');paint()
            if finished[0]:break
    except Exception as ex: traceback.print_exc();last.update(terminal='ENGINEERING_STOP',error=repr(ex))
    finally:trace.close();proc.stdin.close(); code=proc.wait(timeout=60);r.close();assert code==0
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(vid),'-f','null','-'],check=True)
    atomic(out/'video.json',dict(file='full.mp4',index=0,frames=frames,fps=50,playback_speed=1.,decode_pass=True,sha256=digest(vid),checkpoint_sha256=summary['weights'],rollout=last))


def row(z,i):
    terminal='INVALID_PHYSICS' if int(z['overflow'][i]) else 'PROTECTION' if z['failed'][i] else 'SUCCESS' if z['success'][i] else 'ASCENT_WITH_COLLISION' if z['hit_completion'][i] else 'TIMEOUT' if z['timeout'][i] else 'RUNNING'
    return dict(terminal=terminal,sim_s=float(z['sim_s'][i]),foot_max_levels=z['highwater'][i].tolist(),last_levels=z['levels'][i].tolist(),visited_treads=z['seen'][i,1:].tolist(),top_four=bool(z['landing'][i]),hold_s=float(z['dwell'][i]),bad_force=float(z['bad_force'][i]),arm_error_rad=float(z['arm'][i]),overflow=int(z['overflow'][i]), original_goal=bool(z['original_goal'][i]), clean=bool(z['clean'][i]),
        first_hit_tick=int(z['first_hit_tick'][i]), peak_n_by_foot_and_nonfoot=z['hit_peak_n'][i].tolist(),impulse_ns_by_foot_and_nonfoot=z['hit_impulse_ns'][i].tolist(),hit_contact_substep_samples=z['hit_substep_contact_count'][i].tolist(),interval_hit_peak_n=z['interval_hits'][i].tolist())

if __name__=='__main__':evaluate(None if sys.argv[1]=='baseline' else Path(sys.argv[1]),Path(sys.argv[2]))
