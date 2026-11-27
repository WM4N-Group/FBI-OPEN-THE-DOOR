from pathlib import Path
from hashlib import sha256
import os, json, xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np
R=Path(__file__).resolve().parent
B=Path(os.environ['FBI_NAV_MODEL_DIR']).resolve()
H=float(os.environ["FBI_STAIR_RISE"]); T=.28; W=.60; END=4.2
assert H in (.06,.10,.12,.14,.17)

def build():
    if (R/'fbi_nav.xml').exists() or (R/'model.json').exists():
        raise FileExistsError('Use a fresh stair run directory')
    blob=(B/'fbi_nav.xml').read_bytes(); cfg=json.loads((B/'model.json').read_bytes()); assert sha256(blob).hexdigest()==cfg['scene_sha256']
    canonical=ET.fromstring(blob); include=canonical.find('include'); robot=(B/include.get('file')).resolve()
    assert sha256(robot.read_bytes()).hexdigest()==cfg['robot_sha256']=='248d0bc1d01be735897c5e9199415457b9a9f88b688cd29ae31cebfcdc7501ce'
    include.set('file','@FBI_ROBOT_XML@')
    assert sha256(ET.tostring(canonical)).hexdigest()=='6470f48f997cd3d44859f8719cd3600b6fb698cba06f45cf669038d64cbf8de0'
    x=ET.fromstring(blob); x.find('include').set('file',str(robot)); w=x.find('worldbody'); w.remove(w.find('geom[@name="nav_deck"]'))
    # Contiguous treads, no alternate path to the top. Plate 6 includes landing.
    for i in range(6):
        lo=i*T; hi=(i+1)*T if i<5 else END; h=(i+1)*H/2
        ET.SubElement(w,'geom',name='nav_deck' if i==5 else f'stair_{i+1}',type='box',pos=f'{(lo+hi)/2} 0 {h}',size=f'{(hi-lo)/2} {W} {h}',friction='.8 .01 .001',rgba=f'{.23+.035*i} .49 .57 1')
    path=R/'fbi_nav.xml'; path.write_bytes(ET.tostring(x)); m=mj.MjModel.from_xml_path(str(path)); b=mj.MjModel.from_xml_path(str(B/'fbi_nav.xml'))
    for k in ('body_mass','body_inertia','jnt_range','actuator_trnid','actuator_ctrlrange','dof_damping'):
        np.testing.assert_array_equal(getattr(m,k),getattr(b,k))
    old=[i for i in range(b.ngeom) if b.geom(i).name!='nav_deck']
    new=[i for i in range(m.ngeom) if m.geom(i).name!='nav_deck' and not m.geom(i).name.startswith('stair_')]
    assert [m.geom(i).name for i in new]==[b.geom(i).name for i in old]
    for k in ('geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity','geom_friction'):
        np.testing.assert_array_equal(getattr(m,k)[new],getattr(b,k)[old])
    assert m.opt.timestep==b.opt.timestep==.002 and m.nu==19
    cfg.update(scene_sha256=sha256(path.read_bytes()).hexdigest(),shape=[m.nq,m.nv,m.nu,m.nbody,m.ngeom]); (R/'model.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print('ASSET_PASS unchanged robot and 6 continuous steps',H,T)

if __name__=='__main__':build()
