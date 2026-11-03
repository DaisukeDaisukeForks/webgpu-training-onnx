"""Create a reproducible manifest from verified cached official AIST++ metadata."""
import hashlib,io,json,os,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];cache=Path(os.environ.get('AIST_CACHE','.cache/aist'))
assets=json.loads((root/'config/source-assets.json').read_text(encoding='utf-8'))
def read(name,file):
 data=(cache/file).read_bytes();a=assets[name]
 if len(data)!=a['bytes'] or hashlib.sha256(data).hexdigest()!=a['sha256']:raise ValueError('AIST source hash mismatch: '+name)
 return data
cameras=zipfile.ZipFile(io.BytesIO(read('aist-cameras','cameras.zip')));splits=zipfile.ZipFile(io.BytesIO(read('aist-splits','splits.zip')))
meta={'splits':{s:splits.read('splits/pose_'+s+'.txt').decode('utf-8').split() for s in ['train','val','test']},'ignore':read('aist-ignore-list','ignore_list.txt').decode('utf-8').split(),'cameras':{Path(n).stem:json.loads(cameras.read(n)) for n in cameras.namelist() if n.startswith('cameras/') and n.endswith('.json')},'mapping':dict(line.split() for line in cameras.read('cameras/mapping.txt').decode('utf-8').splitlines()),'sources':{k:v for k,v in assets.items() if k.startswith('aist-')},'revision':'aist-pose-v1-scaled','dataset':'AIST++','license':'CC-BY-4.0 annotations; application restricted to noncommercial research','source':'https://google.github.io/aistplusplus_dataset/download.html','coordinates':'Camera xyz / per-sequence SMPL scale; hip-relative Unity axes in meters. Reconstructed labels, not measured absolute distance. No RGB videos.'}
if set(meta['splits']['train'])&set(meta['splits']['val']):raise ValueError('Split overlap')
(root/'public/aist.json').write_text(json.dumps(meta,separators=(',',':'),ensure_ascii=False),encoding='utf-8')
print('AIST camera/split source hashes verified')
