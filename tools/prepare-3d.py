"""Prepare measured 3D labels only; never run a model or training on the host."""
import hashlib,json,sys,os
from pathlib import Path
import numpy as np
import gdown

root=Path(os.environ.get('H3WB_CACHE','.cache/h3wb'));root.mkdir(parents=True,exist_ok=True)
out=Path(__file__).resolve().parents[1]/'public'
source=root/'h3wb.npz'
if not source.exists():
    if gdown.download(id='1LZh4Jsg3_ZKBF0iEPiexzoGHE4srLgfC',output=str(source),quiet=False,use_cookies=False) is None:
        raise RuntimeError('H3WB annotations could not be downloaded')
digest=hashlib.sha256(source.read_bytes()).hexdigest()
if digest!='d423224a9e3c47aa8320687a8fcb58e732aa8d3aabe589e01ce390030d123b1d':
    raise ValueError('Official H3WB hash mismatch; refusing object-array loading')
ids=list(range(5,17))+list(range(92,112))+list(range(113,133))+list(range(17,23))+[23+i for i in [3,13,30,36,39,37,41,42,45,44,46,48,54,62,66]]
with np.load(source,allow_pickle=True) as f: training=f['train_data'].item()
manifest={'dataset':'H3WB','source':'https://github.com/wholebody3d/wholebody3d','distribution':'https://drive.google.com/file/d/1LZh4Jsg3_ZKBF0iEPiexzoGHE4srLgfC/view','license':'MIT (3D annotations only)','sha256':digest,'stride':511,'layout':'73*(xy:2,confidence:1,vrc_xyz:3,valid3d:1) float32 little endian','coordinates':'hip-relative meters; Unity +X right +Y up +Z forward; no absolute translation','train':[],'val':[]}
manifest.update({'revision':'h3wb-all-subject-split-v2','sampling':'all usable observations; no subsampling or duplicated rows','rgb_paired_training':False,'pairing':{}})
for split,subjects,legacy_size in [('train',['S1','S5','S6'],20000),('val',['S7'],1024)]:
    rows=[];sample_ids=[];references=[]
    for subject in subjects:
        for action_name,action in sorted(training[subject].items()):
            frames=np.asarray(action['frame_id']).reshape(-1)
            for camera in ['54138969','55011271','58860488','60457274']:
                if camera not in action:continue
                view=action[camera]; all_xy=np.asarray(view['pose_2d'],np.float32).reshape(-1,133,2)
                xy=all_xy[:,ids];xyz=np.asarray(view['camera_3d'],np.float32).reshape(-1,133,3)[:,ids]/1000
                finite=np.isfinite(all_xy).all(-1);lo=np.where(finite[...,None],all_xy,np.inf).min(1);hi=np.where(finite[...,None],all_xy,-np.inf).max(1)
                extent=hi-lo;h=np.maximum(extent[:,1],extent[:,0]/.75)*1.25;sz=np.stack([h*.75,h],-1);origin=(lo+hi-sz)/2
                with np.errstate(invalid='ignore',divide='ignore'): uv=(xy-origin[:,None])/sz[:,None]
                v2=np.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=1).all(-1);v3=np.isfinite(xyz).all(-1)
                usable=(h>0)&np.isfinite(h)&v3[:,6]&v3[:,7]&(v2[:,:12].sum(-1)>=6)
                xyz-= (xyz[:,6:7]+xyz[:,7:8])*.5
                xyz[:,:,1]*=-1 # camera +Y down -> Unity +Y up
                uv=np.where(v2[...,None],uv,0);xyz=np.where(v3[...,None],xyz,0)
                packed=np.concatenate([uv,v2[...,None],xyz,v3[...,None]],-1)[usable].reshape(-1,511).astype('<f4')
                rows.append(packed)
                original_ids=np.asarray(view['sample_id']).reshape(-1)
                for i in np.flatnonzero(usable):
                    sample_ids.append(f'{subject}/{action_name}/{camera}/{frames[i]}')
                    references.append([str(original_ids[i]),subject,action_name,camera,str(frames[i]),[float(origin[i,0]),float(origin[i,1]),float(sz[i,0]),float(sz[i,1])]])
    rows=np.concatenate(rows)
    # Retain the old subset as a prefix so its complete cached chunks remain reusable.
    legacy=np.random.default_rng(20261103).choice(len(rows),min(legacy_size,len(rows)),replace=False);legacy.sort()
    remaining=np.ones(len(rows),dtype=bool);remaining[legacy]=False
    chosen=np.concatenate([legacy,np.flatnonzero(remaining)]);rows=rows[chosen]
    if len(set(sample_ids))!=len(sample_ids): raise ValueError('Duplicate source observations')
    manifest[split+'_subjects']=subjects;manifest[split+'_count']=len(rows)
    manifest[split+'_unique_moments']=len({(r[1],r[2],r[4]) for r in references})
    for i in range(0,len(rows),1024):
        name=f'h3wb-{split}-{i//1024:03}.bin';data=rows[i:i+1024].tobytes();(out/name).write_bytes(data)
        manifest[split].append({'file':name,'count':min(1024,len(rows)-i),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    (out/f'h3wb-{split}-ids.json').write_text(json.dumps([sample_ids[i] for i in chosen]),encoding='utf-8')
    reference_file=f'h3wb-{split}-frames.json'
    reference={'columns':['official_sample_id','subject','action','camera','frame_id','crop_xywh_pixels'],'row_order':'same as binary labels and ids.json','rgb_included':False,'rgb_source':'Human3.6M; source images must be acquired separately under its terms','rows':[references[i] for i in chosen]}
    (out/reference_file).write_text(json.dumps(reference,separators=(',',':')),encoding='utf-8')
    manifest['pairing'][split]=reference_file
(out/'h3wb.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k.endswith('_count') or k.endswith('_subjects')}))
