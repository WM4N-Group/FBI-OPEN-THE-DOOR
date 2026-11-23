"""Independent NAV assets from unchanged FBI robot. No AINav dependencies."""
import argparse, hashlib, json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco as mj

cli=argparse.ArgumentParser(description='Build NAV assets from the pinned authorized FBI base assets')
cli.add_argument('--base-assets',type=Path,required=True)
cli.add_argument('--output',type=Path,required=True)
args=cli.parse_args();root=args.base_assets.resolve();dst=args.output.resolve();dst.mkdir(parents=True,exist_ok=True)
orig=root/'scene.xml'; robot=root/'robot.xml'
if (dst/'fbi_nav.xml').exists() or (dst/'model.json').exists():raise SystemExit('Output already exists; use a fresh directory')
assert hashlib.sha256(orig.read_bytes()).hexdigest()=='8dd1716adf4e393a7a418cfa2258d266e284836a45abe1e932a18a4892af90fc'
assert hashlib.sha256(robot.read_bytes()).hexdigest()=='248d0bc1d01be735897c5e9199415457b9a9f88b688cd29ae31cebfcdc7501ce'
old=mj.MjModel.from_xml_path(str(orig)); xml=ET.fromstring(orig.read_bytes()); xml.find('include').set('file',str(robot)); world=xml.find('worldbody')
box=world.find('body'); box.set('pos','0 -8 .1'); g=box.find('geom'); g.set('size','1.4 .83 .1'); g.set('mass','16');g.set('name','push_box_geom'); box.find('freejoint').set('name','push_box_free')
ET.SubElement(world,'geom',name='nav_deck',type='box',pos='0 8 .2',size='2.2 1.3 .2',friction='.7 .01 .001',rgba='.33 .56 .68 1')
contact=ET.SubElement(xml,'contact');
for leg in ('FL','FR','RL','RR'):ET.SubElement(contact,'pair',geom1='push_box_geom',geom2=leg,condim='3',friction='.7 .7 0 0 0')
ET.SubElement(contact,'pair',geom1='push_box_geom',geom2='ground',condim='3',friction='.45 .45 0 0 0')
scene=dst/'fbi_nav.xml'; scene.write_bytes(ET.tostring(xml)); m=mj.MjModel.from_xml_path(str(scene)); assert m.opt.timestep==old.opt.timestep==.002 and m.nu==old.nu==19
for k in ('body_mass','body_inertia','body_pos','body_quat','body_ipos','dof_damping','actuator_ctrlrange','actuator_trnid','jnt_range','geom_size','geom_friction','geom_contype','geom_conaffinity','geom_pos','geom_quat'):
    a,b=getattr(old,k),getattr(m,k)
    if k.startswith('body_'): mask=[i for i in range(old.nbody) if i!=old.body('step_box').id]; np.testing.assert_array_equal(a[mask],b[mask],err_msg=k)
    elif k.startswith('geom_'):
        aid=[i for i in range(old.ngeom) if i!=old.geom('step_box_geom').id]; bid=[i for i in range(m.ngeom) if i not in (m.geom('push_box_geom').id,m.geom('nav_deck').id)]; np.testing.assert_array_equal(a[aid],b[bid],err_msg=k)
    else: np.testing.assert_array_equal(a,b,err_msg=k)
joints=[f'{leg}_{j}_joint' for leg in ('FL','FR','RL','RR') for j in ('hip','thigh','calf')]+[f'joint{i}' for i in range(1,7)]
d=mj.MjData(m); mj.mj_resetDataKeyframe(m,d,m.key('home').id); assert [int(m.joint(j).qposadr[0]) for j in joints]==list(range(7,25));
metadata=dict(source='WM4N-Group/FBI-OPEN-THE-DOOR climb freeze 441d477ed4c4d726aa281c2472cd497a712905c9',
  scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),robot_sha256=hashlib.sha256(robot.read_bytes()).hexdigest(),base_asset=str(orig),
  joints=joints,reference=d.qpos[7:25].tolist(),effort=m.actuator_ctrlrange[:18,1].tolist(),kp=[20.]*12+[50.]*6,kd=[1.5]*12+[4.]*6,
  shape=[m.nq,m.nv,m.nu,m.nbody,m.ngeom],robot_mass=m.body_mass.tolist(),robot_inertia=m.body_inertia.tolist(),physics_dt=.002,policy_dt=.02,uses_ainav=False)
(dst/'model.json').write_text(json.dumps(metadata,indent=2)+'\n')
