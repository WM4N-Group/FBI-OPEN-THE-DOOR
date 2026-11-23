import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import warp as wp
import mujoco as mj
import mujoco_warp as mw
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5


@wp.kernel
def motor(q:wp.array2d[float], v:wp.array2d[float], bias:wp.array2d[float], ctrl:wp.array2d[float],
          qids:wp.array[int],dids:wp.array[int],aids:wp.array[int], kp:wp.array[float],kd:wp.array[float],
          limit:wp.array[float], target:wp.array2d[float], old:wp.array2d[float], latency:wp.array[int],
          peak:wp.array2d[float]):
    w,j=wp.tid()
    pos=target[w,j]
    if latency[w] != 0:
        pos=old[w,j]
    torque=kp[j]*(pos-q[w,qids[j]])-kd[j]*v[w,dids[j]]
    if j>=12:
        torque+=bias[w,dids[j]]
    ctrl[w,aids[j]]=wp.clamp(torque,-limit[j],limit[j])
    wp.atomic_max(peak,w,j,wp.abs(torque))
    if j==0:
        ctrl[w,18]=wp.clamp(800.0*(0.006-q[w,25])-5.0*v[w,24],-20.0,20.0)


@wp.kernel
def contact_audit(cone:int,nacon:wp.array[int],world:wp.array[int],geom:wp.array[wp.vec2i],
    frame:wp.array[wp.mat33],friction:wp.array[vec5],dim:wp.array[int],adres:wp.array2d[int],force:wp.array2d[float],njmax:int,
    body_class:wp.array[int],feet:wp.array[int],bad_pair:wp.array2d[int],expected:wp.array[int],
    load:wp.array2d[float],bad:wp.array[float]):
    i=wp.tid()
    if i<nacon[0]:
        w=world[i]; a,b=geom[i][0],geom[i][1]
        if a>=0 and b>=0:
            x,y=body_class[a],body_class[b]
            if (x>0 or y>0):
                f=contact_force_fn(cone,frame,friction,dim,adres,force,njmax,nacon,w,i,False)
                f0=wp.max(0.0,f[0]); fa=feet[a]; fb=feet[b]
                if a==expected[w] and fb>=0:
                    wp.atomic_add(load,w,fb,wp.abs(frame[i][0,2])*f0)
                if b==expected[w] and fa>=0:
                    wp.atomic_add(load,w,fa,wp.abs(frame[i][0,2])*f0)
                if (x==2 and y==0) or (y==2 and x==0) or bad_pair[a,b]>0:
                    wp.atomic_max(bad,w,f0)


@wp.kernel
def flags(current:wp.array[int],agg:wp.array[int]):
    w=wp.tid(); wp.atomic_or(agg,w,current[w])


def invrot(q,v):
    h=q[:,1:]; c=torch.cross(h,v,dim=1)
    return v-2*q[:,:1]*c+2*torch.cross(h,c,dim=1)


class NavEnv:
    dt=.02
    def __init__(self,root,n,seed,capture=True,eval_cases=None):
        wp.init(); self.n=n; self.dev='cuda:0'; self.rng=torch.Generator(device=self.dev).manual_seed(seed)
        self.case=eval_cases
        root=Path(root); cfg=json.loads((root/'model.json').read_text()); mesh=root/'fbi_nav.xml'; self.cfg=cfg
        assert hashlib.sha256(mesh.read_bytes()).hexdigest()==cfg['scene_sha256']
        self.m=m=mj.MjModel.from_xml_path(str(mesh)); assert m.opt.timestep==.002
        assert [m.nq,m.nv,m.nu,m.nbody,m.ngeom]==cfg['shape']
        np.testing.assert_allclose(m.body_mass,cfg['robot_mass'],rtol=6.e-6,atol=1.e-8)
        np.testing.assert_allclose(m.body_inertia,cfg['robot_inertia'],rtol=1.e-5,atol=1.e-8)
        cpu=mj.MjData(m); mj.mj_resetDataKeyframe(m,cpu,m.key('home').id); mj.mj_forward(m,cpu)
        joints=cfg['joints']; j=np.array([m.joint(name).id for name in joints]); aids=np.array([next(i for i in range(m.nu) if m.actuator_trnid[i,0]==x) for x in j]);
        self.aids=aids; self.qids=m.jnt_qposadr[j]; self.dids=m.jnt_dofadr[j]; self.box_qid=int(m.joint('push_box_free').qposadr[0]);
        effort=np.array(cfg['effort']); np.testing.assert_allclose(m.actuator_ctrlrange[aids],np.c_[-effort,effort],atol=1e-7,rtol=0)
        self.p_order=[joints.index(f'{leg}_{typ}_joint') for typ in ('hip','thigh','calf') for leg in ('FL','FR','RL','RR')]; self.m_order=np.argsort(self.p_order).tolist()
        self.ref=self.t(cfg['reference']); self.leg_center=self.ref[:12]; self.low=self.t(m.jnt_range[j[:12],0]+.01); self.high=self.t(m.jnt_range[j[:12],1]-.01)
        self.home=self.t(cpu.qpos.copy()); self.home[self.box_qid:self.box_qid+7]=self.t(m.qpos0[self.box_qid:self.box_qid+7]); self.support_geoms=self.t([m.geom(g).id for g in ('ground','push_box_geom','nav_deck')],torch.int32)
        self.wm=mw.put_model(m,batch_sizes={'geom_friction':n}); self.wm.opt.warn_overflow=False
        self.wd=mw.put_data(m,cpu,nworld=n,nconmax=96,njmax=384)
        self.q=wp.to_torch(self.wd.qpos); self.v=wp.to_torch(self.wd.qvel); self.time=wp.to_torch(self.wd.time)
        self.target=self.ref.expand(n,-1).clone(); self.old=self.target.clone(); self.lag=self.t(np.zeros(n),torch.int32)
        self.foot=self.t(np.zeros((n,4))); self.bad=self.t(np.zeros(n)); self.overflow=self.t(np.zeros(n),torch.int32); self.peak=self.t(np.zeros((n,18)))
        self.expected=self.t(np.zeros(n),torch.int32); self.body_class=np.zeros(m.ngeom,np.int32); self.feet=np.full(m.ngeom,-1,np.int32)
        base=m.body('base').id
        def ancestry(b):
            out=[]
            while b:
                out.append(int(b));b=m.body_parentid[b]
            return out
        bases=[ancestry(b) for b in range(m.nbody)]
        self.style_legs=[m.body(leg+'_foot').id for leg in ('FL','FR','RL','RR')]
        for g in range(m.ngeom):
            body=m.geom_bodyid[g]
            if base in bases[body] and (m.geom_contype[g] or m.geom_conaffinity[g]):
                if body in self.style_legs:
                    self.feet[g]=self.style_legs.index(body);self.body_class[g]=1
                else:self.body_class[g]=2
        forbid=np.zeros((m.ngeom,m.ngeom),np.int32)
        robot=np.flatnonzero(self.body_class)
        for a in robot:
            ca=bases[m.geom_bodyid[a]]
            for b in robot:
                cb=bases[m.geom_bodyid[b]]
                if min(ca.index(c)+cb.index(c) for c in set(ca)&set(cb))>2:
                    forbid[a,b]=1
        w=self.wa; c=self.wd.contact
        self.motor_args=(self.wd.qpos,self.wd.qvel,self.wd.qfrc_bias,self.wd.ctrl,w(self.qids,torch.int32),w(self.dids,torch.int32),w(aids,torch.int32),
                          w(cfg['kp']),w(cfg['kd']),w(effort),wp.from_torch(self.target),wp.from_torch(self.old),wp.from_torch(self.lag),wp.from_torch(self.peak))
        self.audit_args=(self.wm.opt.cone,self.wd.nacon,c.worldid,c.geom,c.frame,c.friction,c.dim,c.efc_address,self.wd.efc.force,self.wd.njmax,
                w(self.body_class,torch.int32),w(self.feet,torch.int32),w(forbid,torch.int32),wp.from_torch(self.expected),wp.from_torch(self.foot),wp.from_torch(self.bad))
        self.of_args=(self.wd.overflow,wp.from_torch(self.overflow)); self.fric=wp.to_torch(self.wm.geom_friction)
        self.history=torch.zeros((n,5,60),device=self.dev); self.last=torch.zeros((n,12),device=self.dev)
        self.age=torch.zeros(n,device=self.dev); self.dwell=torch.zeros_like(self.age); self.mode=torch.zeros(n,device=self.dev,dtype=torch.int64)
        self.goal=torch.zeros((n,2),device=self.dev); self.heading=torch.zeros(n,device=self.dev); self.travel=torch.zeros(n,device=self.dev)
        self.init_q=torch.zeros_like(self.q); self.init_box=torch.zeros((n,2),device=self.dev); self.max_arm=torch.zeros_like(self.age)
        self.previous_distance=torch.zeros(n,device=self.dev); self.active= self.age>=0
        self.bias_gravity=self.t([0,0,-1.]).expand(n,-1); self.phase_offsets=self.t([0.,.5,.5,0.]); self.z0=self.t([0.,.2,.4]); self.all=self.age>=0
        self.w_foot=wp.from_torch(self.foot); self.graph=None
        self.reset(self.all); self._physics(); self.reset(self.all)
        if capture:
            with wp.ScopedCapture() as cap:self._physics()
            self.graph=cap.graph
        self.reset(self.all); self.obs=self.observe()

    def t(self,x,dtype=torch.float32):return torch.as_tensor(x,dtype=dtype,device=self.dev)
    def wa(self,x,dtype=torch.float32):return wp.from_torch(self.t(x,dtype).contiguous())
    def rand(self,*s):return torch.rand(s,generator=self.rng,device=self.dev)
    def reset(self,mask):
        a=self.rand(self.n,8); mode=torch.where(a[:,0]<.55,0,torch.where(a[:,0]<.78,1,2))
        length=torch.where(mode==0,.45+a[:,1]*2.35,torch.where(mode==1,.36+a[:,1]*.55,.40+a[:,1]*1.10))
        y=torch.where(mode==1,-8.,torch.where(mode==2,8.,0.)); z=self.z0[mode]; phi=(a[:,2]-.5)*(torch.where(mode==0,2.8,1.15)); yaw=phi+(a[:,3]-.5)*1.05
        sx=torch.where(mode==0,-1.4+(a[:,4]-.5)*.8,torch.where(mode==1,-.53,-.85)); sy=y+(a[:,5]-.5)*.26
        gx=sx+torch.cos(phi)*length;gy=sy+torch.sin(phi)*length
        face=phi+(a[:,7]-.5)*.60;mu=.42+.48*a[:,6]; lag=torch.where(a[:,5]<.18,0,4)
        if self.case is not None:
            k=self.case; mode[:]=int(k['mode']); sx[:]=float(k['x']); sy[:]=float(k['y']); z=self.z0[mode]; yaw[:]=float(k['yaw']);gx[:]=float(k['gx']);gy[:]=float(k['gy']);face[:]=float(k['heading']);mu[:]=float(k['friction']);lag[:]=int(k['delay'])
            for i in range(self.n):
                if 'variants' in k:
                    v=k['variants'][i];mode[i]=v['mode'];sx[i]=v['x'];sy[i]=v['y'];yaw[i]=v['yaw'];gx[i]=v['gx'];gy[i]=v['gy'];face[i]=v['heading'];mu[i]=v['friction'];lag[i]=v['delay']
            z=self.z0[mode]
        new=self.home.expand(self.n,-1).clone();new[:,0]=sx;new[:,1]=sy;new[:,2]=.31+z
        new[:,3]=torch.cos(yaw/2);new[:,4:6]=0;new[:,6]=torch.sin(yaw/2);new[:,self.qids]=self.ref;new[:,25]=.006;new[:,26]=-.006
        self.q.copy_(torch.where(mask[:,None],new,self.q));self.v.masked_fill_(mask[:,None],0.);self.time.masked_fill_(mask,0.)
        wp.to_torch(self.wd.qacc_warmstart).masked_fill_(mask[:,None],0.)
        self.goal.copy_(torch.where(mask[:,None],torch.stack((gx,gy),-1),self.goal));self.heading.copy_(torch.where(mask,face,self.heading)); self.mode.copy_(torch.where(mask,mode,self.mode));
        self.expected.copy_(torch.where(mask,self.support_geoms[mode],self.expected));self.lag.copy_(torch.where(mask,lag.int(),self.lag));
        self.fric[:,int(self.m.geom('ground').id),0].copy_(torch.where(mask,mu,self.fric[:,int(self.m.geom('ground').id),0]));self.fric[:,int(self.m.geom('nav_deck').id),0].copy_(torch.where(mask,mu,self.fric[:,int(self.m.geom('nav_deck').id),0]));
        self.target.copy_(torch.where(mask[:,None],self.ref,self.target));self.old.copy_(torch.where(mask[:,None],self.ref,self.old));self.init_q.copy_(torch.where(mask[:,None],new,self.init_q));
        for a in (self.age,self.dwell,self.max_arm,self.travel):a.masked_fill_(mask,0.)
        for a in (self.history,self.last):a.masked_fill_(mask.reshape((-1,)+(1,)*(a.ndim-1)),0.)
        self.previous_distance.copy_(torch.where(mask,(self.goal-self.q[:,:2]).norm(dim=-1),self.previous_distance))

    def _physics(self):
        for _ in range(10):
            wp.launch(motor,dim=(self.n,18),inputs=self.motor_args)
            mw.step(self.wm,self.wd)
            self.w_foot.zero_()
            wp.launch(contact_audit,dim=self.wd.naconmax,inputs=self.audit_args)
            wp.launch(flags,dim=self.n,inputs=self.of_args)
        mw.fwd_kinematics(self.wm,self.wd)

    def geometry(self):
        q=self.q[:,3:7];x,y,z=q[:,1],q[:,2],q[:,3];w=q[:,0];yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z));
        d=self.goal-self.q[:,:2];dist=d.norm(dim=-1); c,s=torch.cos(yaw),torch.sin(yaw)
        local=torch.stack((c*d[:,0]+s*d[:,1],-s*d[:,0]+c*d[:,1]),-1)
        error=(self.heading-yaw+torch.pi)%(2*torch.pi)-torch.pi
        goal_heading=torch.atan2(d[:,1],d[:,0]); direction=(goal_heading-yaw+torch.pi)%(2*torch.pi)-torch.pi
        e=torch.where(dist>.20,direction,error); speed=(1.1*dist).clamp(max=.43); speed=torch.where(dist<.060,0.,speed)
        cmd=torch.stack((speed*torch.cos(direction).clamp(min=0),(.6*local[:,1]).clamp(-.12,.12)*(dist>.08), (1.2*e).clamp(-.75,.75)*(e.abs()>.055)),-1)
        z0=torch.where(self.mode==1,self.q[:,self.box_qid+2]+.1,self.z0[self.mode]);height=self.q[:,2]-z0
        return local,dist,error,cmd,height

    def observe(self):
        local,dist,e,cmd,h=self.geometry();q=self.q[:,3:7];self.cmd=cmd
        phase=((self.age[:,None]+.02)*1.4+self.phase_offsets)%1.;phase=torch.where(cmd.norm(dim=-1,keepdim=True)<.01,-1.,phase)
        legq=(self.q[:,self.qids[:12]]-self.leg_center)[:,self.p_order]; legv=self.v[:,self.dids[:12]][:,self.p_order]
        single=torch.cat((invrot(q,self.v[:,:3]),self.v[:,3:6],invrot(q,self.bias_gravity),cmd,torch.zeros_like(local),legq,legv,self.last,phase,self.q[:,self.qids[12:]]-self.ref[12:]),1)
        self.history=torch.cat((self.history[:,1:],single[:,None]),1)
        extra=torch.cat((local.clamp(-3,3),dist[:,None].clamp(max=3),e.sin()[:,None],e.cos()[:,None],h[:,None],(self.foot/50.).clamp(0,3)),1)
        return torch.cat((self.history.flatten(1),extra),1)

    def step(self,action,auto_reset=True):
        self.foot.zero_();self.bad.zero_();self.overflow.zero_();self.peak.zero_();u=action.clamp(-5,5)
        t=(self.leg_center+.5*(.8*u+.2*self.last)[:,self.m_order]).clamp(self.low,self.high)
        self.target[:,:12].copy_(t);prev_cmd=self.cmd
        if self.graph:wp.capture_launch(self.graph)
        else:self._physics()
        eff=torch.where(self.lag[:,None]>0,self.old[:,:12],t);actual=((eff-self.leg_center)/.5)[:,self.p_order]; penalty=(u-self.last).square().mean(dim=1)
        self.last.copy_(actual);self.old.copy_(self.target);self.age+=.02
        local,dist,e,cmd,height=self.geometry();q=self.q[:,3:7]; upright=-invrot(q,self.bias_gravity)[:,2];bvel=invrot(q,self.v[:,:3]); yawrate=self.v[:,5];
        armerr=(self.q[:,self.qids[12:]]-self.ref[12:]).abs().amax(1);self.max_arm=torch.maximum(self.max_arm,armerr)
        v=self.v[:,:3].norm(dim=-1);o=self.v[:,3:6].norm(dim=-1); four=(self.foot>4.).all(1); target_vel=torch.cat((bvel[:,:2],yawrate[:,None]),1)
        risk=(self.bad>1.)|(height<.15)|(upright<.78)|(armerr>.12)|(self.peak[:,12:]/self.t(self.cfg['effort'][12:])).amax(1).ge(.98)
        finite=torch.isfinite(self.q).all(1)&torch.isfinite(self.v).all(1)
        bad=risk|~finite
        hold=(dist<.08)&(e.abs()<.12)&(v<.055)&(o<.24)&(upright>.97)&four&~bad
        self.dwell=torch.where(hold,self.dwell+.02,0.);success=(self.dwell>=1.0)&~bad; timeout=(self.age>=14.)&~success&~bad;done=success|bad|timeout
        progress=(self.previous_distance-dist).clamp(-.04,.04);self.previous_distance.copy_(dist)
        tracking=torch.exp(-((target_vel-prev_cmd).square()*self.t([6.,8.,2.5])).sum(1));
        rew=.75*tracking+12.*progress+.25*upright-.18*(height-.28).square()/.01-.025*penalty-.003*u.square().mean(1)
        rew+=torch.where(dist<.23,.8*torch.exp(-12*v.square())*torch.exp(-4*e.square()),0.)+.7*hold.float()+8.*success.float()-12.*bad.float()
        info=dict(success=success,failed=bad,timeout=timeout,dist=dist,heading_error=e,sim_s=self.age.clone(),dwell=self.dwell.clone(),
                  v=v,armerr=armerr,mode=self.mode.clone(),overflow=self.overflow.clone(),finite=finite,world_max_arm=self.max_arm.clone())
        if auto_reset:self.reset(done)
        self.obs=self.observe()
        return self.obs,torch.nan_to_num(rew),done,info
