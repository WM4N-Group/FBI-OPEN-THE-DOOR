"""Diagnose controller custody and task results on one continuous Warp world per case."""
from pathlib import Path
from hashlib import sha256
import json,math,subprocess,traceback,os

import torch, warp as wp, mujoco as mj, numpy as np
import cv2
from stairs import Stairs,HIT_N
from nav_motor_bridge import NavEnv
from network import mlp

R=Path(__file__).resolve().parent
NAV=Path(os.environ['FBI_STAIR_NAV_CHECKPOINT'])
sha=lambda p:sha256(p.read_bytes()).hexdigest()

def digest(x):return sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

@wp.kernel
def copy_ground_load(load:wp.array3d[float],foot:wp.array2d[float]):
    w,k=wp.tid()
    foot[w,k]=load[w,k,0]

class Chain(Stairs):
    stage='NAV'
    def _physics(self):
        super()._physics()
        # Parent NAV scores before observe(), so provide the same substep's
        # upward ground loads here. Both owners use the same graph and actuators.
        wp.launch(copy_ground_load,dim=(self.n,4),inputs=(self.load_wp,self.w_foot))

    def geometry(self):
        f=NavEnv.geometry if self.stage=='NAV' else Stairs.geometry
        return f(self)

    def observe(self):
        if self.stage=='NAV':
            self.foot.copy_(self.load[:,:,0]);return NavEnv.observe(self)
        if self.stage=='INIT_STAIR_OBS':raise AssertionError('unexpected stage')
        age=self.age
        try:
            self.age=age+self.elapsed_before_stairs # global phase, local skill timeout
            return Stairs.observe(self)
        finally:self.age=age

    def fingerprint(self):
        d=self.wd
        values={k:getattr(self,k) for k in ('q','v','target','old','last','history','load','time','lag')}
        values.update(ctrl=wp.to_torch(d.ctrl),warmstart=wp.to_torch(d.qacc_warmstart))
        return {k:digest(v) for k,v in values.items()}

    @torch.no_grad()
    def handoff(self, goal):
        old=self.fingerprint();history=self.history;cmd=self.cmd;prev_age=self.age
        assert self.stage=='NAV' and (self.load[:,:,0]>4.).all() and (self.lag==0).all()
        assert (self.episode_hit==0).all() and not self.tread_seen.any()
        self.stage='STAIRS'; self.goal[:,0]=goal
        self.elapsed_before_stairs=prev_age.clone();self.age=torch.zeros_like(prev_age);self.dwell=torch.zeros_like(self.dwell)
        _,dist,*_=self.geometry(); self.previous_distance.copy_(dist)
        # Only task goal/clock bookkeeping changes. Build current task features
        # in a read-only preview; do not append a duplicate observation frame.
        preview=self.observe();self.history=history;self.cmd=cmd
        preview[:,:300].copy_(history.flatten(1));self.obs=preview
        after=self.fingerprint(); assert old==after,sorted(k for k in old if old[k]!=after[k])
        return dict(owner_unchanged=old==after,hashes=after,history_unchanged=True,initialized_new_task_features=34,nav_goal_x=-.60,stairs_goal_x=goal)

class Film:
    def __init__(self,m,path):
        self.m=m;self.d=mj.MjData(m);self.r=mj.Renderer(m,360,640);self.cams=[];self.frames=0;self.path=path
        for a in (96,143):
            c=mj.MjvCamera();c.azimuth=a;c.elevation=-20;c.distance=4.5;c.lookat[:]=[.60,0,.42];self.cams.append(c)
        self.p=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x432','-r','50','-i','-','-c:v','libx264','-threads','2','-preset','veryfast','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    def paint(self,e,line):
        self.d.qpos[:]=e.q[0].cpu().numpy();self.d.qvel[:]=e.v[0].cpu().numpy();mj.mj_kinematics(self.m,self.d);mj.mj_comPos(self.m,self.d)
        f=np.full((432,1280,3),(17,24,33),dtype=np.uint8)
        for i,c in enumerate(self.cams):self.r.update_scene(self.d,c);f[64:424,i*640:(i+1)*640]=self.r.render()
        cv2.putText(f,'FBI NAV50100 -> STAIRS  |  6 x 10 cm  |  SAME SIMULATION  |  1x',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.64,(223,238,252),1,cv2.LINE_AA)
        cv2.putText(f,line,(12,51),cv2.FONT_HERSHEY_SIMPLEX,.57,(166,220,251),1,cv2.LINE_AA);self.p.stdin.write(f.tobytes());self.frames+=1
    def close(self):
        self.p.stdin.close();c=self.p.wait(timeout=60);self.r.close();assert c==0
        subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(self.path),'-f','null','-'],check=True)


def tick_check(env,before):
    aft=float(env.time[0]);assert abs(aft-before-.02)<.000045,(aft,before);return aft

def arr(x):return x[0].cpu().tolist()

@torch.no_grad()
def trial(i,v,nav,stair,video):
    out=R/f'case{i:02}';out.mkdir(); init=v.copy();init[-1]=-.60
    e=Chain(R,1,71198+i,[init]); assert e.obs.shape==(1,310); assert not e.tread_seen.any();
    ident=(id(e.m),id(e.wm),id(e.wd),e.q.data_ptr(),e.v.data_ptr(),id(e.graph)); identify=lambda:(id(e.m),id(e.wm),id(e.wd),e.q.data_ptr(),e.v.data_ptr(),id(e.graph));
    f=Film(e.m,out/'full.mp4') if video else None; k=0; term=None; ready=None; first=None; event=None; final=None;first_hit=None
    log=(out/'trace.jsonl').open('x',buffering=1)
    def emit(x):log.write(json.dumps(x)+'\n')
    def frame(x):
        if f:f.paint(e,x)
    try:
        frame('0.00 s | start / NAV')
        for n in range(1,705):
            assert e.stage=='NAV' and identify()==ident
            t=float(e.time[0]);action=nav(e.obs); _,_,done,z=NavEnv.step(e,action,auto_reset=False); now=tick_check(e,t);k+=1
            name='NAV_INVALID_PHYSICS' if z['overflow'][0] else 'NAV_PROTECTION' if z['failed'][0] else 'NAV_TIMEOUT' if z['timeout'][0] else 'NAV_READY' if z['success'][0] else 'NAV'
            row=dict(stage='NAV',step=k,sim_s=now,terminal=name,root=arr(e.q)[:7],speed_m_s=float(e.v[0,:3].norm()),angular_rad_s=float(e.v[0,3:6].norm()),heading_error_rad=float(z['heading_error'][0]),goal_error_m=float(z['dist'][0]),dwell_s=float(z['dwell'][0]),ground_load_n=arr(e.load[:,:,0]),target_leg_rad=arr(e.target[:,:12]))
            emit(row);frame(f'{k*.02:05.2f}s  |  {name}  |  stop dwell {row["dwell_s"]:.2f}s')
            if name=='NAV_READY':ready=row;break
            if done[0] or z['overflow'][0]:term=name;break
        if ready:
            assert ready['dwell_s']>=1 and min(ready['ground_load_n'])>4.
            event=e.handoff(v[-1]);event.update(nav_end_k=k,nav_end_sim_s=ready['sim_s'],epoch_only_reset=True,world_preserved=identify()==ident);assert event['world_preserved']; save(out/'handoff.json',event);before=arr(e.target[:,:12])
            for n in range(1,1105):
                assert identify()==ident and e.obs.shape==(1,334)
                t=float(e.time[0]); _,_,done,z=e.step(stair(e.obs),auto_reset=False); now=tick_check(e,t);k+=1
                collision=max(arr(z['interval_hits']))>HIT_N
                name='INVALID_PHYSICS' if z['overflow'][0] else 'PROTECTION' if z['failed'][0] else 'STRICT_NO_STRIKE' if z['success'][0] else 'ASCENT_WITH_STRIKE' if z['hit_completion'][0] else 'TIMEOUT' if z['timeout'][0] else 'STAIRS'
                final=dict(stage='STAIRS',step=k,sim_s=now,offset_s=n*.02,terminal=name,root=arr(e.q)[:7],speed_m_s=float(e.v[0,:2].norm()),first_hit_step=int(z['first_hit_tick'][0]),
                    foot_max_levels=arr(z['highwater']),levels=arr(z['levels']),goal_error_m=float(z['dist'][0]),top_four=bool(z['landing'][0]),original_goal=bool(z['original_goal'][0]),clean=bool(z['clean'][0]),hold_s=float(z['dwell'][0]),
                    hit_peak_n=arr(z['hit_peak_n']),hit_impulse_ns=arr(z['hit_impulse_ns']),this_interval_hit_n=arr(z['interval_hits']),overflow=int(z['overflow'][0]),target_leg_rad=arr(e.target[:,:12]))
                if n==1:
                    first=dict(**final,actual_physics_time_delta_s=now-t,previous_target_leg_rad=before,first_target_max_delta_rad=max(abs(a-b) for a,b in zip(before,final['target_leg_rad'])))
                    save(out/'first_actual.json',first)
                if collision and first_hit is None:first_hit=final
                emit(final);frame(f'{k*.02:05.2f}s  |  {name}  |  highest {max(final["foot_max_levels"])}/6  |  strike peak {max(final["hit_peak_n"]):.1f} N')
                if done[0]:term=name;break
        term=term or 'UNEXPECTED_LOOP_LIMIT'
    except Exception as ex:traceback.print_exc();term='ENGINEERING_STOP:'+repr(ex)
    finally:
        log.close()
        if f:f.close();assert f.frames==k+1
    vout=dict(file=str(f.path.name),sha256=sha(f.path),fps=50,frames=f.frames,full_decode=True,playback_speed=1) if f else None
    result=dict(case=i,terminal=term,nav_ready=ready,handoff=event,first_actual=first,first_hit=first_hit,stairs_final=final,total_intervals=k,video=vout); save(out/'result.json',result)
    print('CASE',i,term,'NAV',ready is not None,'STAIRS',first is not None,flush=True);return result

if __name__=='__main__':
    torch.set_num_threads(1);cv2.setNumThreads(1);cfg=json.loads((R/'handoff_registration.json').read_bytes());assert sha(NAV)==cfg['nav_sha256'];s=Path(os.environ['FBI_STAIR_CHECKPOINT']);assert sha(s)==cfg['stairs_sha256'];
    nav=mlp(310,12);nav.load_state_dict(torch.load(NAV,map_location='cpu',weights_only=True)['actor']);nav.cuda().eval()
    stair=mlp(334,12);stair.load_state_dict(torch.load(s,map_location='cpu',weights_only=True)['actor']);stair.cuda().eval()
    result=[]
    for i,v in enumerate(cfg['cases']):
        x=trial(i,v,nav,stair,i==cfg['video_index']);result.append(x);save(R/'results.json',result)
        if x['terminal'].startswith('ENGINEERING_STOP'):raise SystemExit(1)
