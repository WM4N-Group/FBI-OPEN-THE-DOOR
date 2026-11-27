import argparse, hashlib, json, os, signal, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from stairs import Stairs as NavEnv
from network import mlp

REFERENCE=Path(os.environ.get('FBI_STAIR_NAV_CHECKPOINT','nav12-seed50100.pt'))
LEG_SHA='f4b874027bea26f801876936d0dbac22491dd7b0de66ab0dd4b968b580a6ad14'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,d):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,path)


class Net(nn.Module):
    def __init__(self):
        super().__init__();self.actor=mlp(334,12);self.critic=mlp(334,1);self.std=nn.Parameter(torch.full((12,),-1.85))
    def init_base(self):
        predecessor=os.environ.get('FBI_STAIR_APPROVED_INIT'); expected=os.environ.get('FBI_STAIR_APPROVED_SHA')
        if predecessor:
            assert digest(predecessor)==expected
            weights=torch.load(predecessor,map_location='cpu',weights_only=True)['actor']
        else:
            assert digest(REFERENCE)==LEG_SHA
            weights=torch.load(REFERENCE,map_location='cpu',weights_only=True)['actor']
        n=weights['0.weight'].shape[-1];assert n in (310,326,334)
        expanded=torch.zeros_like(self.actor[0].weight); expanded[:,:n]=weights['0.weight'];weights['0.weight']=expanded
        self.actor.load_state_dict(weights)
    def dist(self,o):return Normal(self.actor(o),self.std.clamp(-3.5,-.8).exp())
    def value(self,o):return self.critic(o).squeeze(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--seed',type=int,required=True);p.add_argument('--n',type=int,default=2048);p.add_argument('--steps',type=int,default=12000000);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False); torch.set_num_threads(1);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    root=Path(__file__).resolve().parent; h=32; batch=a.n*h
    config=dict(seed=a.seed,n=a.n,budget=a.steps,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),actor='334->12 from verified 10cm FBI stair actor; add per-foot edge gap and clearance',
                trainable='all Actor parameters, critic, exploration std; no arm actions',warm_start_sha256=LEG_SHA,repo='WM4N-Group/FBI-OPEN-THE-DOOR',
                physics_model_sha256=digest(root/'fbi_nav.xml'),robot_source='frozen FBI climb asset',curriculum_predecessor_sha256=os.environ.get('FBI_STAIR_APPROVED_SHA'),uses_ainav=False,source={p.name:digest(p) for p in root.glob('*.py')},method='PPO six-step ascent with real per-tread support',formal_chain_validation=False)
    atomic(out/'config.json',config);model=Net();model.init_base();model.cuda();initial={k:v.clone() for k,v in model.actor.state_dict().items()}
    optim=torch.optim.Adam(model.parameters(), lr=2.e-4); start=time.monotonic(); total=updates=0; status=dict(state='constructing',steps=0)
    atomic(out/'status.json',status)
    interrupted=[]
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,lambda n,f:interrupted.append(n))
    def checkpoint():
        t=out/'policy_tmp.pt';torch.save(dict(actor=model.actor.state_dict(),critic=model.critic.state_dict(),seed=a.seed,steps=total,config=config),t);os.replace(t,out/'policy_latest.pt')
    def change():return max((v-initial[k]).abs().max().item() for k,v in model.actor.state_dict().items())
    checkpoint(); reason='budget_complete'; error=None
    reached_first=False
    try:
        env=NavEnv(root,a.n,a.seed); obs=env.obs; assert obs.shape==(a.n,334)
        shape=(h,a.n); O=torch.empty((*shape,334),device='cuda');A=torch.empty((*shape,12),device='cuda'); L=torch.empty(shape,device='cuda');R=L.clone();D=L.clone(); V=L.clone(); G=L.clone()
        while total<a.steps and not interrupted:
            window_start=time.monotonic()
            pop=torch.zeros((3,4),device='cuda'); proof=torch.zeros(2,dtype=torch.int64,device='cuda');
            with torch.no_grad():
                masks={}
                clean_term=0; hit_term=0
                for i in range(h):
                    O[i]=obs;d=model.dist(obs);ac=d.sample();A[i]=ac;L[i]=d.log_prob(ac).sum(1);V[i]=model.value(obs);obs,r,done,info=env.step(ac)
                    R[i]=r;D[i]=done
                    for mode in range(3):
                        actual=info['highwater'].amin(1).clamp(max=2)==mode; pop[mode]+=torch.stack(tuple((actual&x).sum() for x in (done,info['success'],info['failed'],info['timeout'])))
                    proof[0]+=(info['overflow']!=0).sum();proof[1]+=(~info['finite']).sum()
                    clean_term+=info['success'].sum();hit_term+=info['hit_completion'].sum()
                    if bool((info['overflow']!=0).any()):
                        k,c=torch.unique(info['overflow'][info['overflow']!=0],return_counts=True)
                        masks[i]=dict(flags={str(int(bit)):int(count) for bit,count in zip(k,c)},world_ids=(info['overflow']!=0).nonzero().flatten().cpu().tolist()[:32])
                        atomic(out/'rejected_batch.json',dict(start_step=total,updates=updates,rollout_intervals=masks))
                        break
                    reached_first = reached_first or bool((info['highwater'].amin(1)>=1).any())
                if proof.any().item():raise RuntimeError(f'overflow or nonfinite physical worlds; refuse optimizer update {proof.tolist()}')
                expected=model.value(obs);gae=torch.zeros_like(expected)
                for i in reversed(range(h)):
                    live=1-D[i];gae=R[i]+.99*expected*live-V[i]+.99*.95*live*gae;G[i]=gae;expected=V[i]
                targets=(G+V).flatten();adv=G.flatten();adv=(adv-adv.mean())/(adv.std()+1.e-8)
            flat=O.flatten(0,1); act=A.flatten(0,1); lp=L.flatten(); last_loss=None
            for _ in range(3):
                for ids in torch.randperm(batch,device='cuda').split(8192):
                    pred=model.dist(flat[ids]); ratio=(pred.log_prob(act[ids]).sum(1)-lp[ids]).exp(); gain=adv[ids]
                    loss=-torch.minimum(ratio*gain,ratio.clamp(.8,1.2)*gain).mean()+.45*(model.value(flat[ids])-targets[ids]).square().mean()-.002*pred.entropy().sum(1).mean()
                    if not torch.isfinite(loss).item():raise FloatingPointError('nonfinite PPO update')
                    optim.zero_grad(set_to_none=True);loss.backward();g=nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    if not torch.isfinite(g).item():raise FloatingPointError('nonfinite PPO gradient')
                    optim.step();last_loss=float(loss.detach())
            total+=batch;updates+=1;ch=change()
            status=dict(state='training',last_batch_clean_done=int(clean_term),last_batch_collision_goal=int(hit_term),last_batch_all_feet_level_hist=torch.bincount(info['highwater'].amin(1),minlength=7).tolist(),steps=total,updates=updates,budget=a.steps,seed=a.seed,actor_max_change=ch,loss=last_loss,
                       rollout_plus_update_sps=batch/(time.monotonic()-window_start),minimum_foot_level_buckets_done_success_failure_timeout=pop.to(torch.int64).tolist(),gpu=config['gpu'],
                       arm_target_max_change_rad=float((env.target[:,12:]-env.ref[12:]).abs().max()))
            atomic(out/'status.json',status);print(json.dumps(status),flush=True)
            if updates==1 or updates%10==0:checkpoint()
            if total>=12_000_000 and not reached_first:
                reason='early_gate_no_all_feet_first_step';break
        if interrupted:reason='operator_stop'
    except Exception as e:
        error=type(e).__name__+': '+str(e);reason='failed';print(error,flush=True)
        import traceback;traceback.print_exc()
    finally:
        if torch.isfinite(next(model.actor.parameters())).all().item():checkpoint()
        atomic(out/'status.json',dict(status,state=reason,error=error,steps=total,updates=updates,actor_max_change=change(),final_checkpoint_sha256=digest(out/'policy_latest.pt')))
    if error:raise SystemExit(1)


if __name__=='__main__':main()
