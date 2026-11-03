export const SELECTED=[0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,92,93,95,96,97,99,100,101,103,104,105,107,108,109,111,113,114,116,117,118,120,121,122,124,125,126,128,129,130,132,60,64,67,69,71,77,85,89,25,37];
export function rng(seed){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
export function shuffle(a,r){for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
export const points=a=>[...a.keypoints,...(a.foot_kpts||Array(18).fill(0)),...(a.face_kpts||Array(204).fill(0)),...(a.lefthand_kpts||Array(63).fill(0)),...(a.righthand_kpts||Array(63).fill(0))];
const visible=(a,lo,hi)=>{const p=points(a);return SELECTED.slice(lo,hi).filter(k=>p[k*3+2]>0).length;};
export function records(json,split,count,seed,exclude=new Set()){const images=new Map(json.images.map(i=>[i.id,i])),perImage=new Map();for(const a of json.annotations){if(a.iscrowd||exclude.has(a.image_id))continue;const b=a.bbox;if(!b||Math.min(b[2],b[3])<48||visible(a,0,59)<6)continue;const old=perImage.get(a.image_id);if(!old||visible(a,0,59)>visible(old,0,59))perImage.set(a.image_id,a);}const r=rng(seed),buckets=[[],[],[],[]];for(const[id,a]of perImage){const k=visible(a,19,49)>=8?0:visible(a,13,19)>=3?1:visible(a,49,59)>=5?2:3;buckets[k].push({id,split,image:images.get(id),annotation:a});}buckets.forEach(b=>shuffle(b,r));const out=[];let k=0;while(out.length<count&&buckets.some(b=>b.length)){const b=buckets[k++%4];if(b.length)out.push(b.pop());}if(out.length<count)throw Error(`${split}: 対象画像が不足 ${out.length}/${count}`);return shuffle(out,r);}
export async function fetchBytes(url,progress=()=>{},signal){const r=await fetch(url,{signal});if(!r.ok)throw Error(`HTTP ${r.status}: ${url}`);const reader=r.body.getReader(),total=+(r.headers.get('Content-Length')||0),a=[];let n=0;while(true){const{value,done}=await reader.read();if(done)break;a.push(value);n+=value.length;progress(n,total);}return new Blob(a);}
export class Dataset{
 constructor(onProgress){this.onProgress=onProgress;this.memory=new Map();this.pending=new Map();this.saved=new Set();this.warnings=new Set();this.bytes=0;this.stats={hits:0,network:0,stored:0,edgeHits:0};}
 warn(key,message){if(!this.warnings.has(key)){this.warnings.add(key);this.onProgress(message);}}
 async init(){
  try{this.cache=await caches.open('compact59-public-v2');for(const r of await this.cache.keys())this.saved.add(r.url);}catch(e){this.cache=null;this.warn('cache-open',`画像キャッシュを開けません (${e.name})。IndexedDBを使用します。`);}
  try{this.db=await new Promise((resolve,reject)=>{const r=indexedDB.open('compact59-image-cache',1);r.onupgradeneeded=()=>r.result.createObjectStore('images');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});for(const key of await this.disk('keys'))this.saved.add(key);}catch(e){this.db=null;this.warn('db-open',`画像のIndexedDBを開けません (${e.name})。利用可能な保存先を使用します。`);}
  this.stats.stored=this.saved.size;this.onProgress(`画像キャッシュ: 保存済み${this.saved.size}枚。取得済みの画像は再ダウンロードしません。`);
 }
 disk(mode,key,value){return new Promise((resolve,reject)=>{const tx=this.db.transaction('images',mode==='put'?'readwrite':'readonly'),store=tx.objectStore('images');const r=mode==='put'?store.put(value,key):mode==='keys'?store.getAllKeys():store.get(key);let result;r.onsuccess=()=>{result=r.result;};tx.oncomplete=()=>resolve(result);tx.onabort=()=>reject(tx.error||Error('画像キャッシュ保存が中断されました'));tx.onerror=()=>reject(tx.error);});}
 url(r){return new URL(`./api/images/${r.split}/${String(r.id).padStart(12,'0')}.jpg`,location.href).href;}
 async image(r,signal){
  if(signal?.aborted)throw new DOMException('Stopped','AbortError');const url=this.url(r);
  // A second request for the same image shares its ongoing download.
  if(this.pending.has(url))return this.pending.get(url);
  const work=this.load(url,signal);this.pending.set(url,work);try{return await work;}finally{this.pending.delete(url);}
 }
 async load(url,signal){
  if(this.memory.has(url)){this.stats.hits++;return this.memory.get(url);}
  if(this.cache){try{const response=await this.cache.match(url);if(response?.ok&&response.headers.get('Content-Type')?.startsWith('image/')){const blob=await response.blob();if(blob.size){this.stats.hits++;return blob;}}}catch(e){this.warn('cache-read',`画像キャッシュ読込失敗 (${e.name})。別の保存先を確認します。`);}}
  if(this.db){try{const blob=await this.disk('get',url);if(blob?.size){this.stats.hits++;return blob;}}catch(e){this.warn('db-read',`画像のIndexedDB読込失敗 (${e.name})。必要な画像を取得します。`);}}
  let blob,lastError;
  for(let attempt=0;attempt<3;attempt++){
   try{const response=await fetch(url,{signal});if(!response.ok){const reason=(await response.text()).slice(0,400);throw Error(`画像取得 HTTP ${response.status}: ${reason}`);}blob=await response.blob();this.bytes+=blob.size;
    if(!blob.size||!blob.type.startsWith('image/'))throw Error('画像ではない応答を受信しました');
    // Only complete, decodable images enter either persistent cache.
    const bitmap=await createImageBitmap(blob);bitmap.close();if(response.headers.get('X-Image-Cache')==='HIT')this.stats.edgeHits++;break;
   }catch(e){lastError=e;blob=null;if(signal?.aborted||attempt===2)throw e;}
  }
  if(!blob)throw lastError;this.stats.network++;let stored=false;
  if(this.cache){try{await this.cache.put(url,new Response(blob,{headers:{'Content-Type':blob.type}}));stored=true;}catch(e){this.warn('cache-write',`画像キャッシュ保存失敗 (${e.name})。IndexedDBへ保存します。`);}}
  if(!stored&&this.db){try{await this.disk('put',url,blob);stored=true;}catch(e){this.warn('db-write',`画像の永続保存失敗 (${e.name})。このセッションはメモリを使います。再読込後は再取得が必要です。`);}}
  if(stored){this.saved.add(url);this.stats.stored=this.saved.size;}else this.memory.set(url,blob);
  return blob;
 }
 async download(list,signal){let next=0,done=0;const controller=new AbortController(),combined=AbortSignal.any([signal,controller.signal]);const jobs=await Promise.allSettled(Array.from({length:6},async()=>{try{while(next<list.length){const r=list[next++];await this.image(r,combined);this.onProgress(null,++done,list.length,this.bytes);}}catch(e){controller.abort();throw e;}}));const failure=jobs.find(j=>j.status==='rejected');if(failure)throw failure.reason;}
}
const FACE_FLIP=[16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,26,25,24,23,22,21,20,19,18,17,27,28,29,30,35,34,33,32,31,45,44,43,42,47,46,39,38,37,36,41,40,54,53,52,51,50,49,48,59,58,57,56,55,64,63,62,61,60,67,66,65];
const FLIP=Array.from({length:133},(_,i)=>i<17?[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15][i]:i<23?i<20?i+3:i-3:i<91?23+FACE_FLIP[i-23]:i<112?i+21:i-21);
export async function sample(dataset,record,tf,augment,random,signal){const bitmap=await createImageBitmap(await dataset.image(record,signal)),canvas=new OffscreenCanvas(192,256),ctx=canvas.getContext('2d',{willReadFrequently:true}),[x,y,w,h]=record.annotation.bbox;let ew=w*1.25,eh=h*1.25;if(ew>eh*.75)eh=ew/.75;else ew=eh*.75;let cx=x+w/2,cy=y+h/2;const flip=augment&&random()<.5,angle=augment?(random()-.5)*Math.PI/6:0;if(augment){cx+=(random()-.5)*.15*ew;cy+=(random()-.5)*.15*eh;const scale=.8+random()*.4;ew*=scale;eh*=scale;}const sx=192/ew,sy=256/eh,c=Math.cos(angle),s=Math.sin(angle),f=flip?-1:1,a=f*sx*c,b=sy*s,d=-f*sx*s,e=sy*c,tx=96-a*cx-d*cy,ty=128-b*cx-e*cy;
 ctx.fillStyle='black';ctx.fillRect(0,0,192,256);ctx.setTransform(a,b,d,e,tx,ty);ctx.drawImage(bitmap,0,0);bitmap.close();ctx.resetTransform();const pixels=ctx.getImageData(0,0,192,256).data,input=new Float32Array(3*192*256),mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375],gain=augment?.8+random()*.4:1;for(let i=0;i<192*256;i++)for(let k=0;k<3;k++)input[k*192*256+i]=(Math.min(255,pixels[i*4+k]*gain)-mean[k])/std[k];
 const p=points(record.annotation),xy=[],mask=new Float32Array(59),gx=new Float32Array(59*384),gy=new Float32Array(59*512);for(let j=0;j<59;j++){const k=flip?FLIP[SELECTED[j]]:SELECTED[j],px=p[k*3],py=p[k*3+1],u=a*px+d*py+tx,v=b*px+e*py+ty;xy.push([u,v]);const valid=k<17?true:k<23?record.annotation.foot_valid!==false:k<91?record.annotation.face_valid!==false:k<112?record.annotation.lefthand_valid!==false:record.annotation.righthand_valid!==false;mask[j]=p[k*3+2]>0&&valid&&u>=0&&u<192&&v>=0&&v<256?(p[k*3+2]===1?.5:1):0;for(const[n,target,center]of [[384,gx,u*2],[512,gy,v*2]]){let sum=0;for(let q=0;q<n;q++){const z=Math.exp(-.5*((q-center)/6)**2);target[j*n+q]=z;sum+=z;}if(sum>0)for(let q=0;q<n;q++)target[j*n+q]/=sum;}}return{input,mask,x:gx,y:gy,xy,canvas,record};}
export function batch(tf,samples){const n=samples.length,cat=key=>{const a=new Float32Array(samples[0][key].length*n);samples.forEach((s,i)=>a.set(s[key],i*s[key].length));return a;};return{input:tf.tensor(cat('input'),[n,3,256,192]),mask:tf.tensor(cat('mask'),[n,59]),x:tf.tensor(cat('x'),[n,59,384]),y:tf.tensor(cat('y'),[n,59,512])};}
