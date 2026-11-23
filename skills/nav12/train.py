import argparse, hashlib, json, os, signal, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from env import NavEnv
def mlp(nin,nout):
    return nn.Sequential(nn.Linear(nin,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,nout))

REFERENCE=Path(os.environ['FBI_NAV_WARMSTART']) if 'FBI_NAV_WARMSTART' in os.environ else None

def model_root():
    if 'FBI_NAV_MODEL_DIR' not in os.environ:raise SystemExit('Set FBI_NAV_MODEL_DIR to the compiled NAV model directory')
    return Path(os.environ['FBI_NAV_MODEL_DIR']).resolve()
LEG_SHA='408f96e58a3ea544cdf9a2a27ba0e913c7d02c6a4346cc6202e5c1f3351b0556'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,d):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,path)


class Net(nn.Module):
    def __init__(self):
        super().__init__();self.actor=mlp(310,12);self.critic=mlp(310,1);self.std=nn.Parameter(torch.full((12,),-1.85))
    def init_base(self):
        if REFERENCE is None or digest(REFERENCE)!=LEG_SHA:raise ValueError('frozen FBI reference SHA mismatch')
        weights={k[6:]:v for k,v in torch.load(REFERENCE,map_location='cpu',weights_only=False)['model_state_dict'].items() if k.startswith('actor.')}
        expanded=torch.zeros_like(self.actor[0].weight);expanded[:,:300]=weights['0.weight'];weights['0.weight']=expanded
        self.actor.load_state_dict(weights)
    def dist(self,o):return Normal(self.actor(o),self.std.clamp(-3.5,-.7).exp())
    def value(self,o):return self.critic(o).squeeze(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--seed',type=int,required=True);p.add_argument('--n',type=int,default=2048);p.add_argument('--steps',type=int,default=12000000);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False); torch.set_num_threads(1);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    root=model_root(); h=24; batch=a.n*h
    config=dict(seed=a.seed,n=a.n,budget=a.steps,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),actor='310->12 full leg actor; 300 original history + 10 new target/height/load terms',
                trainable='all Actor parameters, critic, exploration std; no arm actions',warm_start_sha256=LEG_SHA,repo='WM4N-Group/FBI-OPEN-THE-DOOR',
                physics_model_sha256=digest(root/'fbi_nav.xml'),robot_source='frozen FBI climb asset',uses_ainav=False,source={p.name:digest(p) for p in Path(__file__).resolve().parent.glob('*.py')},method='PPO goal NAV',formal_chain_validation=False)
    atomic(out/'config.json',config);model=Net();model.init_base();model.cuda();initial={k:v.clone() for k,v in model.actor.state_dict().items()}
    optim=torch.optim.Adam(model.parameters(), lr=1.e-4); start=time.monotonic(); total=updates=0; status=dict(state='constructing',steps=0)
    atomic(out/'status.json',status)
    interrupted=[]
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,lambda n,f:interrupted.append(n))
    def checkpoint():
        t=out/'policy_tmp.pt';torch.save(dict(actor=model.actor.state_dict(),critic=model.critic.state_dict(),seed=a.seed,steps=total,config=config),t);os.replace(t,out/'policy_latest.pt')
    def change():return max((v-initial[k]).abs().max().item() for k,v in model.actor.state_dict().items())
    checkpoint(); reason='budget_complete'; error=None
    try:
        env=NavEnv(root,a.n,a.seed); obs=env.obs; assert obs.shape==(a.n,310)
        shape=(h,a.n); O=torch.empty((*shape,310),device='cuda');A=torch.empty((*shape,12),device='cuda'); L=torch.empty(shape,device='cuda');R=L.clone();D=L.clone(); V=L.clone(); G=L.clone()
        while total<a.steps and not interrupted:
            window_start=time.monotonic()
            pop=torch.zeros((3,4),device='cuda'); proof=torch.zeros(2,dtype=torch.int64,device='cuda');
            with torch.no_grad():
                for i in range(h):
                    O[i]=obs;d=model.dist(obs);ac=d.sample();A[i]=ac;L[i]=d.log_prob(ac).sum(1);V[i]=model.value(obs);obs,r,done,info=env.step(ac)
                    R[i]=r;D[i]=done
                    for mode in range(3):
                        actual=info['mode']==mode; pop[mode]+=torch.stack(tuple((actual&x).sum() for x in (done,info['success'],info['failed'],info['timeout'])))
                    proof[0]+=(info['overflow']!=0).sum();proof[1]+=(~info['finite']).sum()
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
            status=dict(state='training',steps=total,updates=updates,budget=a.steps,seed=a.seed,actor_max_change=ch,loss=last_loss,
                       rollout_plus_update_sps=batch/(time.monotonic()-window_start),modes_ground_box_deck_done_success_failure_timeout=pop.to(torch.int64).tolist(),gpu=config['gpu'],
                       arm_target_max_change_rad=float((env.target[:,12:]-env.ref[12:]).abs().max()))
            atomic(out/'status.json',status);print(json.dumps(status),flush=True)
            if updates==1 or updates%10==0:checkpoint()
        if interrupted:reason='operator_stop'
    except Exception as e:
        error=type(e).__name__+': '+str(e);reason='failed';print(error,flush=True)
        import traceback;traceback.print_exc()
    finally:
        if torch.isfinite(next(model.actor.parameters())).all().item():checkpoint()
        atomic(out/'status.json',dict(status,state=reason,error=error,steps=total,updates=updates,actor_max_change=change(),final_checkpoint_sha256=digest(out/'policy_latest.pt')))
    if error:raise SystemExit(1)


if __name__=='__main__':main()
