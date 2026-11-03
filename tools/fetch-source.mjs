// Reproduction downloads use a deployed Sites proxy, never direct origin fetches.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {prefetchAll} from '../public/prefetch.mjs';
const root=path.dirname(path.dirname(new URL(import.meta.url).pathname)),assets=JSON.parse(fs.readFileSync(path.join(root,'config/source-assets.json'),'utf8'));
const names=process.argv.slice(2).length?process.argv.slice(2):['cmu-train','cmu-val','aist-cameras','aist-splits','aist-ignore-list'];
const filename={'cmu-train':'cmu-train.json','cmu-val':'cmu-val.json','aist-cameras':'cameras.zip','aist-splits':'splits.zip','aist-ignore-list':'ignore_list.txt','aist-motions':'motions.zip','aist-keypoints3d':'keypoints3d.zip'};
const digest=async file=>{const hash=crypto.createHash('sha256');for await(const b of fs.createReadStream(file))hash.update(b);return hash.digest('hex');};
for(const name of names){
 const item=assets[name];if(!item||!filename[name])throw Error('Unknown source asset');
 const folder=name.startsWith('cmu-')?(process.env.CMU_FOOT_CACHE||'.cache/cmu-foot'):(process.env.AIST_CACHE||'.cache/aist'),dest=path.resolve(folder,filename[name]);fs.mkdirSync(path.dirname(dest),{recursive:true});
 if(fs.existsSync(dest)&&fs.statSync(dest).size===item.bytes&&await digest(dest)===item.sha256){console.log(name+': verified local cache');continue;}
 if(!process.env.TEACHER_SITE)throw Error('Set TEACHER_SITE to your deployed Sites URL. No direct origin fallback is used.');
 const base=new URL(process.env.TEACHER_SITE);if(base.protocol!=='https:'||base.username||base.password)throw Error('Invalid Sites URL');
 const temporary=dest+'.partial',file=await fs.promises.open(temporary,'w');let done=0;
 try{
  await prefetchAll(Array.from({length:Math.ceil(item.bytes/item.chunkBytes)},(_,i)=>i),async(i,signal)=>{
   const response=await fetch(new URL(`/api/teacher-assets/${name}/${i}`,base),{signal,redirect:'manual',headers:process.env.SITES_BYPASS_TOKEN?{'OAI-Sites-Authorization':'Bearer '+process.env.SITES_BYPASS_TOKEN}:{}});
   if(!response.ok||response.headers.get('X-Teacher-SHA256')!==item.sha256)throw Error(`${name}: Sites HTTP ${response.status} or source version mismatch`);
   const bytes=new Uint8Array(await response.arrayBuffer()),expected=Math.min(item.chunkBytes,item.bytes-i*item.chunkBytes);if(bytes.length!==expected)throw Error('Source chunk length mismatch');
   let written=0;while(written<bytes.length){const result=await file.write(bytes,written,bytes.length-written,i*item.chunkBytes+written);if(!result.bytesWritten)throw Error('Source cache write failed');written+=result.bytesWritten;}
   done+=bytes.length;console.log(`${name}: ${done}/${item.bytes} · Cloudflare ${response.headers.get('X-Teacher-Cache')}`);
  },{concurrency:4});
 }finally{await file.close();}
 if(await digest(temporary)!==item.sha256){await fs.promises.unlink(temporary);throw Error(name+': source SHA-256 mismatch');}
 await fs.promises.rename(temporary,dest);console.log(name+': saved and SHA-256 verified');
}
