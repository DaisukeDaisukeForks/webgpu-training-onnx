"""Use all available measured ASPset body labels without a sample-count/stride cap."""
import os,csv,hashlib,json
from pathlib import Path
import numpy as np,ezc3d

root=Path(os.environ.get('ASPSET_CACHE', '.cache/aspset'));out=Path(__file__).resolve().parents[1]/'public'
manifest=json.loads((out/'h3wb.json').read_text())
manifest['train']=[x for x in manifest['train'] if x['file'].startswith('h3wb-')];manifest['val']=[x for x in manifest['val'] if x['file'].startswith('h3wb-')]
manifest['train_count']=sum(x['count'] for x in manifest['train']);manifest['val_count']=sum(x['count'] for x in manifest['val']);manifest['primary_val_count']=manifest['val_count']
names=['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
paired=json.loads((out/'paired.json').read_text());reserved={(c['subject'],c['clip']) for c in paired['val']}
source={'dataset':'ASPset-510','source':'https://github.com/anibali/aspset-510','distribution':'https://archive.org/details/aspset510','license':'CC0-1.0','frame_stride':1,'subsample_limit':None,'joints':names,'coordinates':'hip-relative meters; Unity +X right +Y up +Z forward','excluded_from_train_for_video_validation':sorted('/'.join(x) for x in reserved)}
rows={s:[] for s in ['train','val']};ids={s:[] for s in rows};subjects={s:set() for s in rows};clips={s:set() for s in rows};seen=set()
for subject,clip,split,cameras in csv.reader((root/'splits.csv').open()):
    if split not in rows:continue
    if (subject,clip) in reserved:split='val'
    path=root/'trainval/joints_3d'/subject/f'{subject}-{clip}.c3d';m=ezc3d.c3d(str(path))
    labels=[str(x).strip().lower().replace(' ','_') for x in m['parameters']['POINT']['LABELS']['value']];ix=[labels.index(n) for n in names]
    points=np.asarray(m['data']['points']);world=points[:3,ix,:].transpose(2,1,0);factor={'mm':.001,'cm':.01,'m':1}[str(m['parameters']['POINT']['UNITS']['value'][0]).strip().lower()]
    for camera_id in (['left','mid','right'] if cameras=='all' else [cameras]):
        key=(subject,clip,camera_id)
        if key in seen:continue
        seen.add(key);camera=json.loads((root/'trainval/cameras'/subject/f'{subject}-{camera_id}.json').read_text())
        K=np.asarray(camera['intrinsic_matrix']).reshape(3,4);E=np.asarray(camera['extrinsic_matrix']).reshape(4,4)
        hom=np.concatenate([world,np.ones((*world.shape[:-1],1))],-1)@E.T;xyz=hom[...,:3]/hom[...,3:];projected=hom@K.T
        with np.errstate(divide='ignore',invalid='ignore'):xy=projected[...,:2]/projected[...,2:]
        valid=np.isfinite(xyz).all(-1)&np.isfinite(xy).all(-1)&(points[3,ix,:].T>=0)
        sign=1 if np.median(xyz[np.isfinite(xyz[...,2])][:,2])>0 else -1;axes=np.array([np.sign(K[0,0]/K[2,2])*sign,-np.sign(K[1,1]/K[2,2])*sign,sign])
        xyz=xyz*axes*factor;xyz-=(xyz[:,6:7]+xyz[:,7:8])*.5
        lo=np.where(valid[...,None],xy,np.inf).min(1);hi=np.where(valid[...,None],xy,-np.inf).max(1);extent=hi-lo;h=np.maximum(extent[:,1],extent[:,0]/.75)*1.25;size=np.stack([h*.75,h],-1);origin=(lo+hi-size)*.5
        with np.errstate(divide='ignore',invalid='ignore'):uv=(xy-origin[:,None])/size[:,None]
        mask2=valid&np.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=1).all(-1);usable=valid[:,6]&valid[:,7]&(valid.sum(1)>=6)&np.isfinite(h)&(h>0)
        packed=np.concatenate([np.where(mask2[...,None],uv,0),mask2[...,None],np.where(valid[...,None],xyz,0),valid[...,None]],-1)[usable].reshape(-1,84).astype('<f4')
        if not np.isfinite(packed).all():raise ValueError('Nonfinite source labels')
        rows[split].append(packed);ids[split].extend('/'.join(key)+f'/{i}' for i in np.flatnonzero(usable));subjects[split].add(subject);clips[split].add((subject,clip))
if clips['train']&clips['val']:raise ValueError('Clip leakage')
for split in rows:
    data=np.concatenate(rows[split]);keys=ids[split]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate observations')
    for i in range(0,len(data),1024):
        name=f'aspset3d-{split}-{i//1024:03}.bin';blob=data[i:i+1024].tobytes();(out/name).write_bytes(blob)
        manifest[split].append({'file':name,'count':len(data[i:i+1024]),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'stride':84,'joints':list(range(12)),'dataset':'ASPset-510'})
    source[split+'_count']=len(data);source[split+'_subjects']=sorted(subjects[split]);source[split+'_clips']=len(clips[split]);manifest[split+'_count']+=len(data)
    (out/f'aspset3d-{split}-ids.json').write_text(json.dumps(keys,separators=(',',':')),encoding='utf-8')
manifest['additional_sources']=[source];manifest['revision']='h3wb-aspset-all-frames-v3'
(out/'h3wb.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'aspset':source,'combined_train':manifest['train_count'],'combined_val':manifest['val_count']}))
