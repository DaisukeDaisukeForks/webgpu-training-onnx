"""Rebuild teacher files from official sources; do not train or fetch COCO images."""
import os,subprocess,sys
from pathlib import Path
import gdown

root=Path(__file__).resolve().parents[1]
os.chdir(root)
cache=Path(os.environ.get('COCO_ANNOTATION_CACHE','.cache/coco'))
cache.mkdir(parents=True,exist_ok=True)
for split,source in [('train','1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf'),('val','1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb')]:
    file=Path(os.environ.get('COCO_'+split.upper()+'_ANNOTATIONS',str(cache/(split+'.json'))))
    if not file.exists():
        if gdown.download(id=source,output=str(file),quiet=False,use_cookies=False) is None:
            raise RuntimeError('Official COCO annotations could not be downloaded: '+split)
    with file.open('rb') as stream:
        if stream.read(2) in (b'\xff\xfe',b'\xfe\xff'):raise ValueError('UTF-16 annotations are not supported')
        stream.seek(0)
        subprocess.run(['node','tools/prepare-coco.mjs',split],stdin=stream,check=True)
for script in ['prepare-3d.py','prepare-paired.py','prepare-aspset-3d.py']:
    subprocess.run([sys.executable,'tools/'+script],check=True)
subprocess.run(['node','tools/verify-data.mjs'],check=True)
