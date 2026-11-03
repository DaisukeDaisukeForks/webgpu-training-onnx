import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {IDS} from '../public/schema.mjs';
const root=path.dirname(path.dirname(new URL(import.meta.url).pathname));
const assets=JSON.parse(fs.readFileSync(path.join(root,'config/source-assets.json'),'utf8'));
const cache=process.env.CMU_FOOT_CACHE||'.cache/cmu-foot';
const summary={dataset:'CMU Human Foot Keypoint Dataset',source:'https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/',license:'CC-BY-4.0',mirror:'https://github.com/Eva20150932/coco-foot-and-leg/tree/5d99ff365a82073704e6feda1a4412065a0b62b1',mirror_note:'The mirror documents wrapping categories in an array. Original annotation download returned HTTP 503. The mirrored bytes are pinned by SHA-256.',coordinates:'2D pixel x,y,visibility; six toe/heel points, no sole surface/contact or 3D labels'};
for(const split of ['train','val']){
  const file=path.join(cache,`cmu-${split}.json`);
  if(!fs.existsSync(file))throw Error(`Missing ${file}. Retrieve cmu-${split} through the Site /api/teacher-assets cache before data:prepare.`);
  const bytes=fs.readFileSync(file),asset=assets[`cmu-${split}`];
  if(bytes.length!==asset.bytes||crypto.createHash('sha256').update(bytes).digest('hex')!==asset.sha256)throw Error(`CMU source hash mismatch: ${split}`);
  const json=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes)),images=new Map(json.images.map(i=>[i.id,i])),rows=[];
  for(const a of json.annotations){
    if(a.keypoints.length!==(split==='train'?69:18))throw Error('Unexpected CMU keypoint layout');
    const foot=a.keypoints.slice(-18);
    if(!foot.some((v,i)=>i%3===2&&v>0))continue;
    const image=images.get(a.image_id);
    if(!image||!a.bbox||a.bbox.some(v=>!Number.isFinite(v))||a.bbox[2]<=0||a.bbox[3]<=0)throw Error('Invalid CMU image/bbox');
    const points=Array.from({length:73},()=>[0,0,0]);
    if(split==='train')for(let j=0;j<12;j++)points[j]=a.keypoints.slice(IDS[j]*3,IDS[j]*3+3);
    for(let j=0;j<6;j++)points[52+j]=foot.slice(j*3,j*3+3);
    if(points.flat().some(v=>!Number.isFinite(v)))throw Error('Nonfinite CMU labels');
    rows.push({id:a.image_id,annotationId:a.id,split,dataset:'CMU Human Foot',bbox:a.bbox,points,width:image.width,height:image.height,file_name:image.file_name,license:image.license});
  }
  rows.sort((a,b)=>a.id-b.id||a.annotationId-b.annotationId);
  if(new Set(rows.map(r=>`${r.id}/${r.annotationId}`)).size!==rows.length)throw Error('Duplicate CMU annotations');
  const result={...summary,split,source_annotations:json.annotations.length,count:rows.length,unique_images:new Set(rows.map(r=>r.id)).size,source_sha256:asset.sha256,licenses:json.licenses,rows};
  fs.writeFileSync(path.join(root,'public',`cmu-foot-${split}.json`),JSON.stringify(result));
  console.log(JSON.stringify({split,source_annotations:result.source_annotations,accepted:result.count,images:result.unique_images}));
}
