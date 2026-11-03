// Maintainer utility: explicitly record a reviewed dataset revision.
import fs from 'node:fs';import crypto from 'node:crypto';
const root=new URL('../',import.meta.url),publicDir=new URL('public/',root),sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const json=name=>JSON.parse(fs.readFileSync(new URL(name,publicDir),'utf8'));
const paired=json('paired.json'),lock={files:{},coco:{},paired:{}};
for(const name of fs.readdirSync(publicDir,{recursive:true}).filter(n=>/\.bin$|^paired\/.*\.jpg$|^cmu-foot-.*\.json$|^aist\.json$/.test(n)))lock.files[name]=sha(fs.readFileSync(new URL(name,publicDir)));
for(const split of ['train','val']){lock.coco[split]=sha(JSON.stringify(json(`coco-${split}.json`).rows.map(r=>[r.id,r.bbox,r.points,r.width,r.height])));lock.paired[split]=sha(JSON.stringify(paired[split]));}
fs.mkdirSync(new URL('config/',root),{recursive:true});
fs.writeFileSync(new URL('config/data-lock.json',root),JSON.stringify(lock));
fs.writeFileSync(new URL('config/data-recipe.json',root),JSON.stringify({seed:20261103,paired_clips:paired.clips.map(c=>c.id).sort(),ffmpeg:'6.1.1',python:'3.12',numpy:'2.3.5',pillow:'12.3.0',aspset_prefix_limit_bytes:900*1024**2},null,2));
