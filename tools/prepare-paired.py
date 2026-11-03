"""Acquire a bounded prefix of official CC0 videos; pair decoded RGB with measured frames."""
import os,csv,hashlib,io,json,shutil,subprocess,tarfile
from pathlib import Path
import requests,numpy as np,ezc3d
from PIL import Image

root=Path(os.environ.get('ASPSET_CACHE', '.cache/aspset'));root.mkdir(parents=True,exist_ok=True)
out=Path(__file__).resolve().parents[1]/'public';images=out/'paired';images.mkdir(exist_ok=True)
recipe=json.loads((out.parent/'config/data-recipe.json').read_text(encoding='utf-8'))
wanted=set(recipe['paired_clips'])
base='https://archive.org/download/aspset510/'
archives={'common-splits':'ca539550f46762741d8239ed4ac16b8e','trainval-cameras':'1dca82253524fa6f379616851c8e9356','trainval-joints_3d':'f848dc688c3d8536063515fff5349b98','trainval-boxes':'9816529acd1bacf21ad527412251e21e'}
for key,md5 in archives.items():
    path=root/(key+'.tgz')
    if not path.exists():
        with requests.get(base+'aspset510_v1_'+key+'.tar.gz',stream=True,timeout=(30,120)) as r:
            r.raise_for_status()
            with path.open('wb') as f:
                for b in r.iter_content(1024*1024):f.write(b)
    if hashlib.md5(path.read_bytes()).hexdigest()!=md5:raise ValueError('Annotation hash mismatch: '+key)
    with tarfile.open(path) as t:
        for member in t:
            if not member.isfile():continue
            p=Path(member.name)
            if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
            p=root.joinpath(*p.parts[1:]);p.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(member) as a,p.open('wb') as b:shutil.copyfileobj(a,b)
split_rows=list(csv.reader((root/'splits.csv').open()))
split_by_clip={(s,c):split for s,c,split,cam in split_rows}
names=['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
manifest={'dataset':'ASPset-510','source':'https://github.com/anibali/aspset-510','distribution':base+'aspset510_v1_trainval-videos.tar.gz','license':'CC0-1.0','fps':50,'joints':names,'train':[],'val':[],'clips':[],'alignment':'Decoded RGB frame index = C3D point index = annotated box row. Three causal consecutive frames, no cross-clip history.'}
def prepare(video,subject,clip,cam):
    split='val' if clip=='0009' else 'train'
    mocap=ezc3d.c3d(str(root/'trainval/joints_3d'/subject/f'{subject}-{clip}.c3d'))
    labels=[str(x).strip().lower().replace(' ','_') for x in mocap['parameters']['POINT']['LABELS']['value']]
    ix=[labels.index(n) for n in names];points=np.asarray(mocap['data']['points']);world=points[:3,ix].transpose(2,1,0)
    unit=str(mocap['parameters']['POINT']['UNITS']['value'][0]).strip().lower();factor={'mm':.001,'cm':.01,'m':1}[unit]
    camera=json.loads((root/'trainval/cameras'/subject/f'{subject}-{cam}.json').read_text())
    K=np.asarray(camera['intrinsic_matrix']).reshape(3,4);E=np.asarray(camera['extrinsic_matrix']).reshape(4,4)
    hom=np.concatenate([world,np.ones((*world.shape[:-1],1))],-1)@E.T
    xyz=hom[...,:3]/hom[...,3:];proj=hom@K.T;xy=proj[...,:2]/proj[...,2:]
    boxes=np.loadtxt(root/'trainval/boxes'/subject/f'{subject}-{clip}-{cam}.csv',delimiter=',',skiprows=1)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(video)]))['streams'][0] if video else {'nb_read_frames':len(world),'width':3840,'height':2160,'r_frame_rate':'50/1'}
    count=int(probe['nb_read_frames']);width=int(probe['width']);height=int(probe['height'])
    if count!=len(world) or count!=len(boxes) or probe['r_frame_rate']!='50/1':raise ValueError('Video/mocap/box frame alignment mismatch')
    sign=1 if np.median(xyz[...,2])>0 else -1;axes=np.array([np.sign(K[0,0]/K[2,2])*sign,-np.sign(K[1,1]/K[2,2])*sign,sign]);xyz=xyz*axes*factor;xyz-=(xyz[:,6:7]+xyz[:,7:8])*.5
    # Decode once; bound the pipe to the person region shared by this clip.
    x0=max(0,int(np.floor(boxes[:,0].min()))-100);y0=max(0,int(np.floor(boxes[:,1].min()))-100)
    x1=min(width,int(np.ceil(boxes[:,2].max()))+100);y1=min(height,int(np.ceil(boxes[:,3].max()))+100)
    rw,rh=x1-x0,y1-y0;cmd=['ffmpeg','-v','error','-threads','2','-filter_threads','1','-i',str(video),'-map','0:v:0','-an','-vf',f'format=rgb24,crop={rw}:{rh}:{x0}:{y0}','-vsync','0','-f','rawvideo','pipe:1']
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE) if video else None;accepted=[];clipid=f'{subject}-{clip}-{cam}';dest=images/clipid;dest.mkdir(exist_ok=True)
    try:
        for i in range(count):
            b=bytearray()
            while video and len(b)<rw*rh*3:
                part=process.stdout.read(rw*rh*3-len(b))
                if not part:raise ValueError('Truncated decoded video')
                b.extend(part)
            lo=boxes[i,:2];hi=boxes[i,2:];center=(lo+hi)*.5;h=max(hi[1]-lo[1],(hi[0]-lo[0])/.75)*1.25;size=np.array([h*.75,h]);origin=center-size*.5
            uv=(xy[i]-origin)/size;valid=np.isfinite(xyz[i]).all(-1)&np.isfinite(uv).all(-1)&(points[3,ix,i]>=0)
            path=dest/f'{i:06}.jpg'
            if video:
                image=Image.frombytes('RGB',(rw,rh),bytes(b));box=[origin[0]-x0,origin[1]-y0,origin[0]+size[0]-x0,origin[1]+size[1]-y0]
                image=image.transform((192,256),Image.Transform.EXTENT,box,resample=Image.Resampling.BILINEAR);image.save(path,quality=88)
            elif not path.exists():raise ValueError('Incomplete cached clip '+clipid)
            accepted.append({'image':str(path.relative_to(out)),'frame':i,'xy':np.nan_to_num(uv).round(7).tolist(),'xyz':np.nan_to_num(xyz[i]).round(7).tolist(),'mask':valid.astype(int).tolist()})
        if process and process.wait()!=0:raise ValueError('Video decode failed')
    finally:
        if process:
            process.stdout.close()
            if process.poll() is None:process.kill();process.wait()
    manifest[split].append({'id':clipid,'subject':subject,'clip':clip,'camera':cam,'frames':accepted})
    manifest['clips'].append({'id':clipid,'split':split,'frames':count,'mocap_first_frame':int(mocap['header']['points']['first_frame'])})
    print(json.dumps({'prepared':clipid,'split':split,'frames':count,'train':sum(len(c['frames']) for c in manifest['train']),'val':sum(len(c['frames']) for c in manifest['val'])}),flush=True)

# A compressed tar prefix avoids the 23.7GB archive download. Stop once useful clips are collected.
class Reader:
    def __init__(self,raw):self.raw=raw;self.count=0
    def read(self,n=-1):
        if n<0:n=65536
        if self.count+n>900*1024**2:raise ValueError('Stop before downloading the whole video archive')
        b=self.raw.read(n);self.count+=len(b);return b
if all((images/clip).is_dir() for clip in wanted):
    for folder in sorted(images.iterdir()):
        if not folder.is_dir() or folder.name not in wanted:continue
        subject,clip,cam=folder.name.split('-');prepare(None,subject,clip,cam)
    manifest['video_bytes_transferred']=900*1024**2
    manifest['split_policy']='Clip-held-out subset of official train: 04ac/0009 and all its cameras reserved for validation. No official test frames used.'
else:
 with requests.get(manifest['distribution'],stream=True,timeout=(30,120)) as response:
    response.raise_for_status();reader=Reader(response.raw)
    with tarfile.open(fileobj=reader,mode='r|gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('.mkv'):continue
            clipid=Path(member.name).stem
            if clipid not in wanted:continue
            subject,clip,cam=clipid.split('-')
            video=root/'current.mkv'
            with archive.extractfile(member) as a,video.open('wb') as b:shutil.copyfileobj(a,b)
            prepare(video,subject,clip,cam);video.unlink()
            if {c['id'] for c in manifest['clips']}==wanted:break
    manifest['video_bytes_transferred']=reader.count
if {c['id'] for c in manifest['clips']}!=wanted:raise ValueError('Required video clips are missing')
manifest['clips'].sort(key=lambda c:c['id'])
manifest['split_policy']='Clip-held-out subset of official train: 04ac/0009 and all its cameras reserved for validation. No official test frames used.'
for split in ['train','val']:
    manifest[split].sort(key=lambda c:c['id'])
    manifest[split+'_frames']=sum(len(c['frames']) for c in manifest[split])
    manifest[split+'_windows']=sum(max(0,len(c['frames'])-2) for c in manifest[split])
    if not manifest[split]:raise ValueError('Missing paired split')
(out/'paired.json').write_text(json.dumps(manifest,separators=(',',':')),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k.endswith('_frames') or k.endswith('_windows') or k=='video_bytes_transferred'}),flush=True)
