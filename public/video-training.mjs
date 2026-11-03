import {teachers} from './teacher-cache.mjs?revision=17';
import {rng} from './cache.mjs?revision=8';
import {occlusionPlan,drawOcclusion} from './augmentation.mjs?revision=11';
export async function pairedData(){const m=await teachers.json('paired.json');const windows={};for(const split of ['train','val'])windows[split]=m[split].flatMap(c=>c.frames.slice(2).map((_,i)=>({clip:c.id,frames:c.frames.slice(i,i+3)})));return{manifest:m,...windows};}
export function videoFiles(data){return [...new Set(['train','val'].flatMap(split=>data.manifest[split].flatMap(c=>c.frames.map(f=>f.image))))];}
export async function prefetchVideo(data,signal,concurrency,onProgress){return teachers.prefetch(videoFiles(data),signal,concurrency,onProgress);}
export async function videoBatch(tf,windows,signal,{augment=false,seed=20261103,epoch=0}={}){
 const pixels=new Float32Array(windows.length*3*256*192*3),xyz=new Float32Array(windows.length*219),xy=new Float32Array(windows.length*146),mask=new Float32Array(windows.length*73),valid=new Float32Array(windows.length*73),sequence=new Float32Array(windows.length*3*219),sequenceMask=new Float32Array(windows.length*3*73);
 for(let n=0;n<windows.length;n++){
  let key=(seed+Math.imul(epoch,0x9e3779b1))>>>0;for(const c of windows[n].clip+'/'+windows[n].frames[2].image)key=Math.imul(key^c.charCodeAt(0),16777619)>>>0;
  const random=rng(key),zoom=augment?.8+random()*.6:1,dx=augment?(random()-.5)*.12:0,dy=augment?(random()-.5)*.12:0,plan=augment?occlusionPlan(random):[],firstCovered=1+Math.floor(random()*2),gain=augment?.8+random()*.4:1;
  for(let t=0;t<3;t++){
   const f=windows[n].frames[t],bytes=await teachers.bytes(f.image,signal,{cacheOnly:true}),im=await createImageBitmap(new Blob([bytes])),canvas=new OffscreenCanvas(192,256),c=canvas.getContext('2d');
   c.fillStyle='black';c.fillRect(0,0,192,256);c.setTransform(zoom,0,0,zoom,192*((1-zoom)*.5+dx),256*((1-zoom)*.5+dy));c.drawImage(im,0,0);im.close();c.resetTransform();if(t>=firstCovered)drawOcclusion(c,plan,192,256);
   const rgba=c.getImageData(0,0,192,256).data,offset=(n*3+t)*256*192*3;
   for(let p=0;p<256*192;p++)for(let k=0;k<3;k++)pixels[offset+p*3+k]=Math.min(1,rgba[p*4+k]/255*gain);
   for(let j=0;j<12;j++){sequence.set(f.xyz[j],(n*3+t)*219+j*3);sequenceMask[(n*3+t)*73+j]=f.mask[j];}
  }
  const f=windows[n].frames[2];for(let j=0;j<12;j++){
   const uv=[(f.xy[j][0]-.5)*zoom+.5+dx,(f.xy[j][1]-.5)*zoom+.5+dy];xyz.set(f.xyz[j],n*219+j*3);xy.set(uv,n*146+j*2);mask[n*73+j]=f.mask[j];valid[n*73+j]=f.mask[j]&&uv.every(x=>x>=0&&x<=1)?1:0;
  }
 }
 return{frames:tf.tensor(pixels,[windows.length,3,256,192,3]),target:tf.tensor(xyz,[windows.length,73,3]),xy:tf.tensor(xy,[windows.length,73,2]),mask:tf.tensor(mask,[windows.length,73]),valid:tf.tensor(valid,[windows.length,73]),sequence:tf.tensor(sequence,[windows.length,3,73,3]),sequenceMask:tf.tensor(sequenceMask,[windows.length,3,73])};
}
