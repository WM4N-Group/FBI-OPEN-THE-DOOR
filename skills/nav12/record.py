import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import mujoco as mj
from env import NavEnv
from train import Net, atomic, digest, LEG_SHA, REFERENCE, model_root


def scene_film(env,net,case,file):
    m=env.m;d=mj.MjData(m); renderer=mj.Renderer(m,height=540,width=960);cam=mj.MjvCamera();cam.elevation=-24;cam.azimuth=78;cam.distance=5.0
    cam.lookat[:]=[(case['x']+case['gx'])/2,(case['y']+case['gy'])/2,.30]
    args=['ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','960x540','-framerate','50','-i','-','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(file)]
    enc=subprocess.Popen(args,stdin=subprocess.PIPE); frames=0
    def frame():
        nonlocal frames
        d.qpos[:]=env.q[0].cpu().numpy();d.qvel[:]=env.v[0].cpu().numpy();d.time=float(env.time[0]);mj.mj_forward(m,d)
        renderer.update_scene(d,cam);enc.stdin.write(renderer.render().tobytes());frames+=1
    result=None
    try:
        frame()
        for i in range(705):
            _,_,done,z=env.step(net.actor(env.obs),auto_reset=False)
            frame()
            if bool(done[0]) or bool(z['overflow'][0]):
                result=receipt(z,0);break
        if result is None:result=dict(terminal='capture_limit',sim_s=float(env.time[0]))
    finally:
        enc.stdin.close(); rc=enc.wait(timeout=90);renderer.close()
        if rc:raise RuntimeError('encoder failed')
    v=subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,r_frame_rate,width,height,nb_read_frames,duration','-of','json',str(file)],text=True)
    v=json.loads(v)['streams'][0];assert int(v['nb_read_frames'])==frames
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(file),'-f','null','-'],check=True)
    return dict(relative_file=file.name,sha256=digest(file),frames=frames,format=v,simulated_outcome=result,case=case,full_decode=True,rate=1,live_render_of_original_rollout=True)


def receipt(z,i):
    out={k:(bool(z[k][i]) if k in ('success','failed','timeout','finite') else float(z[k][i])) for k in ('success','failed','timeout','finite','dist','sim_s','dwell','v','world_max_arm')};out['overflow']=int(z['overflow'][i]);return out


@torch.no_grad()
def main():
    root=model_root();out=Path(sys.argv[1]); gpu=int(sys.argv[2]); ck=out/'policy_latest.pt'; start_sha=digest(ck)
    train_state=json.loads((out/'status.json').read_text()); assert train_state['final_checkpoint_sha256']==start_sha
    out2=out/'final';out2.mkdir();cfg=json.loads(Path(__file__).with_name('heldout.json').read_text()); variants=cfg['variants']; n=len(variants)
    model=Net(); model.actor.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['actor']);model.cuda().eval();torch.set_num_threads(1)
    evalcase=dict(variants[0],variants=variants); results={}; videos=[]; errors=[]
    try:
        e=NavEnv(root,n,881122,eval_cases=evalcase)
        for _ in range(705):
            _,_,done,z=e.step(model.actor(e.obs),auto_reset=True)
            for i in torch.where(done|(z['overflow']!=0))[0].tolist():
                if i not in results:results[i]=dict(case=variants[i],**receipt(z,i))
            if len(results)==n:break
        for i in range(n):results.setdefault(i,dict(case=variants[i],success=False,terminal='unobserved'))
    except Exception as e:errors.append('heldout: '+str(e))
    for j in cfg['video_indices']:
        case=variants[j]
        try:
            env=NavEnv(root,1,881122,eval_cases=case);videos.append(scene_film(env,model,case,out2/f'case_{j:02d}_full.mp4'))
        except Exception as e:errors.append('video '+str(j)+': '+str(e))
    assert digest(ck)==start_sha and (REFERENCE is None or digest(REFERENCE)==LEG_SHA)
    score=[dict(mode=j,assigned=32,observed=sum(int(v['case']['mode']==j and 'sim_s' in v) for v in results.values()),passed=sum(int(v['case']['mode']==j and v.get('success',False)) for v in results.values())) for j in range(3)]
    report=dict(scope='independent FBI NAV evaluation; no chain; one initial placement per independent elevated episode',train_state=train_state,final_checkpoint_sha256=start_sha,
            used_ainav=False,heldout_registration_sha256=digest(Path(__file__).with_name('heldout.json')),numeric=score,cases=results,videos=videos,errors=errors,video_pending=bool(len(videos)!=len(cfg['video_indices'])),gpu_index=gpu)
    atomic(out2/'delivery.json',report)
    print(json.dumps({k:report[k] for k in ('numeric','final_checkpoint_sha256','video_pending','errors')}),flush=True)


if __name__=='__main__':main()
