"""Separate simulation-only FBI stair skill, using the original motor graph timing."""
import numpy as np
import os
import torch, warp as wp
from nav_motor_bridge import NavEnv, flags, motor, invrot, mw, contact_force_fn, vec5
from build import H,T,W,END

HIT_N=.10
DENSE_CLEARANCE=float(os.environ.get("FBI_CLEARANCE_COST","1.0"))
assert DENSE_CLEARANCE in (0.,1.)

@wp.kernel
def loads(cone:int,nacon:wp.array[int],world:wp.array[int],geom:wp.array[wp.vec2i],
    frame:wp.array[wp.mat33],friction:wp.array[vec5],dim:wp.array[int],addr:wp.array2d[int],force:wp.array2d[float],njmax:int,
    bodies:wp.array[int],feet:wp.array[int],bad_pair:wp.array2d[int],level:wp.array[int],s:wp.array3d[float],bad:wp.array[float], hits:wp.array2d[float], impulses:wp.array2d[float], contact_counts:wp.array2d[int]):
    i=wp.tid()
    if i<nacon[0] and addr[i,0]>=0:
        w=world[i]; a,b=geom[i][0],geom[i][1]
        if a>=0 and b>=0:
            x,y=bodies[a],bodies[b]
            if x>0 or y>0:
                f=contact_force_fn(cone,frame,friction,dim,addr,force,njmax,nacon,w,i,False)
                n=wp.max(0.0,f[0]); z=frame[i][0,2]; la,lb=level[a],level[b]
                if la>=0 and feet[b]>=0 and z>0.65: wp.atomic_add(s,w,feet[b],la,n*z)
                if lb>=0 and feet[a]>=0 and z< -0.65: wp.atomic_add(s,w,feet[a],lb,-n*z)
                # Feet can load upward on a tread, but not strike the riser or
                # side. Any resolved non-foot load against a stair is unwanted.
                k=-1
                if la>0 and y>0:
                    if feet[b]>=0 and z<0.85: k=feet[b]
                    elif feet[b]<0: k=4
                if lb>0 and x>0:
                    if feet[a]>=0 and z> -0.85: k=feet[a]
                    elif feet[a]<0: k=4
                if k>=0:
                    wp.atomic_max(hits,w,k,n)
                    wp.atomic_add(impulses,w,k,.002*n)
                    if n>0.10:wp.atomic_add(contact_counts,w,k,1)
                if (x==2 and y==0) or (y==2 and x==0) or bad_pair[a,b]>0:wp.atomic_max(bad,w,n)


class Stairs(NavEnv):
    def __init__(self,root,n,seed,cases=None):
        self.stair_cases=cases
        super().__init__(root,n,seed)

    def height_at(self,x,y):
        ix=torch.where(x>=0,torch.floor(x/T)+1.,0.).clamp(0,6)
        return torch.where((y.abs()<W)&(x<END),ix*H,0.)

    def _setup(self):
        m=self.m; self.load=torch.zeros((self.n,4,7),device=self.dev)
        self.foot_ids=[m.geom(k).id for k in ('FL','FR','RL','RR')]
        self.foot_radius=self.t(m.geom_size[self.foot_ids,0])
        self.pos=wp.to_torch(self.wd.geom_xpos)
        level=np.full(m.ngeom,-1,np.int32)
        geo=[m.geom(k).id for k in ('ground',*(f'stair_{i}' for i in range(1,6)),'nav_deck')]
        self.terrain_ids=geo
        for i,g in enumerate(geo):level[g]=i
        self.load_wp=wp.from_torch(self.load)
        self.hit=torch.zeros((self.n,5),device=self.dev);self.impulse=torch.zeros_like(self.hit)
        self.hit_counts=torch.zeros((self.n,5),device=self.dev,dtype=torch.int32)
        self.episode_hit=torch.zeros_like(self.hit); self.episode_impulse=torch.zeros_like(self.hit); self.episode_hit_counts=torch.zeros_like(self.hit_counts)
        self.first_hit_tick=torch.zeros(self.n,device=self.dev,dtype=torch.int32)
        self.load_args=(*self.audit_args[:13],self.wa(level,torch.int32),self.load_wp,wp.from_torch(self.bad),wp.from_torch(self.hit),wp.from_torch(self.impulse),wp.from_torch(self.hit_counts))
        self.highwater=torch.zeros((self.n,4),device=self.dev,dtype=torch.long)
        self.tread_seen=torch.zeros((self.n,7),device=self.dev,dtype=torch.bool)
        self.level_ids=torch.arange(7,device=self.dev)

    def reset(self,mask):
        if not hasattr(self,'load'):self._setup()
        a=self.rand(self.n,5); sx=-.48-.32*a[:,0]; sy=.26*(a[:,1]-.5); yaw=.24*(a[:,2]-.5); mu=.64+.30*a[:,3]; goal=2.65+.16*a[:,4]
        if self.stair_cases is not None:
            sx,sy,yaw,mu,goal=self.t(self.stair_cases).T
        q=self.home.expand(self.n,-1).clone();q[:,0]=sx;q[:,1]=sy;q[:,2]=.30;q[:,3]=torch.cos(yaw/2);q[:,4:6]=0;q[:,6]=torch.sin(yaw/2);q[:,self.qids]=self.ref;q[:,25]=.006;q[:,26]=-.006
        self.q.copy_(torch.where(mask[:,None],q,self.q));self.v.masked_fill_(mask[:,None],0.);self.time.masked_fill_(mask,0.);wp.to_torch(self.wd.qacc_warmstart).masked_fill_(mask[:,None],0.)
        self.goal[:,0]=torch.where(mask,goal,self.goal[:,0]);self.target.copy_(torch.where(mask[:,None],self.ref,self.target));self.old.copy_(torch.where(mask[:,None],self.ref,self.old));self.lag.masked_fill_(mask,0)
        for t in (self.age,self.dwell,self.max_arm):t.masked_fill_(mask,0.)
        for t in (self.history,self.last,self.foot,self.load,self.highwater,self.tread_seen,self.hit,self.impulse,self.hit_counts,self.episode_hit,self.episode_impulse,self.episode_hit_counts,self.first_hit_tick):t.masked_fill_(mask.reshape((-1,)+(1,)*(t.ndim-1)),0)
        self.fric[:,self.terrain_ids,0]=torch.where(mask[:,None],mu[:,None],self.fric[:,self.terrain_ids,0])
        self.previous_distance.copy_(torch.where(mask,(self.goal-self.q[:,:2]).norm(dim=-1),self.previous_distance))
        mw.fwd_kinematics(self.wm,self.wd)

    def _physics(self):
        for _ in range(10):
            wp.launch(motor,dim=(self.n,18),inputs=self.motor_args)
            mw.step(self.wm,self.wd)
            self.load_wp.zero_();wp.launch(loads,dim=self.wd.naconmax,inputs=self.load_args)
            wp.launch(flags,dim=self.n,inputs=self.of_args)
        mw.fwd_kinematics(self.wm,self.wd)

    def geometry(self):
        local,dist,error,cmd,_=super().geometry()
        # Match original flat-ground guidance on approach and landing; cap
        # forward demand on stairs instead of asking for a running gait.
        cmd[:,0].clamp_(max=.36)
        return local,dist,error,cmd,self.q[:,2]-self.height_at(self.q[:,0],self.q[:,1])

    def next_riser(self):
        fp=self.pos[:,self.foot_ids]; sole=fp[:,:,2]-self.foot_radius
        i=torch.ceil((fp[:,:,0]-.025)/T).clamp(0,5)
        gap=T*i-fp[:,:,0]-self.foot_radius
        present=(fp[:,:,0]<5*T+.035)&(fp[:,:,1].abs()<W)
        dz=sole-H*(i+1.)
        return torch.where(present,gap,.6),torch.where(present,dz,.20)

    def observe(self):
        self.foot.copy_(self.load.sum(-1))
        base=super().observe();q=self.q
        x=q[:,0:1]+self.t([-.32,-.18,0.,.16,.30,.46,.62,.78]);y=q[:,1:2]
        fp=self.pos[:,self.foot_ids]; sole=fp[:,:,2]-self.foot_radius
        extra=torch.cat((self.height_at(x,y)-q[:,2:3],sole-self.height_at(fp[:,:,0],fp[:,:,1]),(self.load[:,:,1:].sum(-1)/50).clamp(0,3)),1)
        gap,dz=self.next_riser(); return torch.cat((base,extra,gap.clamp(-.10,.60),dz.clamp(-.45,.30)),1)

    def step(self,action,auto_reset=True):
        self.bad.zero_();self.overflow.zero_();self.peak.zero_();self.hit.zero_();self.impulse.zero_();self.hit_counts.zero_();u=action.clamp(-5,5)
        previous=self.target[:,:12].clone(); t=(self.leg_center+.5*(.8*u+.2*self.last)[:,self.m_order]).clamp(self.low,self.high); self.target[:,:12].copy_(t)
        wp.capture_launch(self.graph) if self.graph else self._physics()
        self.last.copy_(((t-self.leg_center)/.5)[:,self.p_order]);self.old.copy_(self.target);self.age+=.02
        _,dist,e,_,h=self.geometry();q=self.q;v=self.v;up=-invrot(q[:,3:7],self.bias_gravity)[:,2]; lin=v[:,:3].norm(dim=-1);ang=v[:,3:6].norm(dim=-1)
        arm=(q[:,self.qids[12:]]-self.ref[12:]).abs().amax(1);self.max_arm=torch.maximum(self.max_arm,arm)
        fp=self.pos[:,self.foot_ids]; sole=fp[:,:,2]-self.foot_radius
        support=(self.load>4.) & ((sole[:,:,None]-self.level_ids*H).abs()<.035)
        current=torch.where(support,self.level_ids,0).amax(-1); new=(current-self.highwater).clamp(min=0); self.highwater=torch.maximum(self.highwater,current); self.tread_seen|=support.any(1)
        finite=torch.isfinite(q).all(1)&torch.isfinite(v).all(1)
        # Same NAV 1 N non-foot/self contact, arm bounds and body tilt guards.
        bad=(self.bad>1.)|(up<.78)|(h<.085)|(arm>.12)|(q[:,1].abs()>=W-.06)|(self.peak[:,12:]/self.t(self.cfg['effort'][12:])).amax(1).ge(.98)|~finite
        landing=(support[:,:,6] & (fp[:,:,0]>5*T+.025) & (fp[:,:,0]<END-.025) & (fp[:,:,1].abs()<W-.025)).all(1); passed_treads=self.tread_seen[:,1:].all(1); hold=(dist<.08)&(e.abs()<.12)&(lin<.055)&(ang<.24)&(up>.97)&landing&passed_treads&~bad
        fresh_hit=self.hit.amax(1)>HIT_N
        first_hit=fresh_hit & (self.first_hit_tick==0)
        self.first_hit_tick.copy_(torch.where(first_hit,(self.age/.02).round().int(),self.first_hit_tick))
        self.episode_hit.copy_(torch.maximum(self.episode_hit,self.hit));self.episode_impulse.add_(self.impulse);self.episode_hit_counts.add_(self.hit_counts)
        clean=self.episode_hit.amax(1)<=HIT_N; invalid=self.overflow!=0
        self.dwell=torch.where(hold,self.dwell+.02,0.); original_goal=(self.dwell>=1.)&~bad
        success=original_goal&clean&~invalid; hit_completion=original_goal&~clean&~invalid
        timeout=(self.age>=22.)&~original_goal&~bad&~invalid;done=original_goal|bad|timeout|invalid
        prog=(self.previous_distance-dist).clamp(-.045,.045);self.previous_distance.copy_(dist)
        rew=11*prog+.68*new.float().sum(1)+.25*(up-1.)-.045*(t-previous).square().mean(1)-.018-.15*(h-.29).square()/.01
        # Reward swing clearance before the next riser without using artificial contact.
        fp=self.pos[:,self.foot_ids];sole=fp[:,:,2]-self.foot_radius; rise=self.height_at(fp[:,:,0]+.10,fp[:,:,1])-self.height_at(fp[:,:,0]-.025,fp[:,:,1]); airborne=self.load.sum(-1)<3.
        clr=(sole-self.height_at(fp[:,:,0]+.10,fp[:,:,1])+.035).clamp(0,.07)
        rew+=.55*(clr*(rise>.05)*airborne).sum(1)+.40*hold.float()+12*success.float()-8*bad.float()
        # Impulses cover all original substeps, not the 50 Hz endpoint only.
        gap,dz=self.next_riser()
        close=(gap>-.035)&(gap<.075)&airborne
        clearance_cost=(.028-dz).clamp(0,.14)*close*(1.-gap.clamp(0,.075)/.075)
        rew-=1.2*first_hit.float()+.24*fresh_hit.float()+.8*torch.log1p(self.impulse[:,:4].sum(1)*4.)+8.*DENSE_CLEARANCE*clearance_cost.sum(1)
        rew-=1.5*hit_completion.float()
        info=dict(success=success,hit_completion=hit_completion,original_goal=original_goal,clean=clean,failed=bad,timeout=timeout,finite=finite,overflow=self.overflow.clone(),sim_s=self.age.clone(),dist=dist,dwell=self.dwell.clone(),landing=landing,seen=self.tread_seen.clone(),levels=current,highwater=self.highwater.clone(),bad_force=self.bad.clone(),arm=arm,hit_peak_n=self.episode_hit.clone(),hit_impulse_ns=self.episode_impulse.clone(),hit_substep_contact_count=self.episode_hit_counts.clone(),first_hit_tick=self.first_hit_tick.clone(),interval_hits=self.hit.clone())
        if auto_reset:self.reset(done)
        self.obs=self.observe();return self.obs,torch.nan_to_num(rew),done,info
