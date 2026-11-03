import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {setTimeout as delay} from 'node:timers/promises';
import {webcrypto} from 'node:crypto';
import {prefetchAll} from '../public/prefetch.mjs';
import {Dataset} from '../public/cache.mjs';
import {TeacherCache,TEACHERS} from '../public/teacher-cache.mjs';
import {videoFiles} from '../public/video-training.mjs';

globalThis.location = {href:'https://test.invalid/'};
globalThis.crypto ??= webcrypto;
globalThis.createImageBitmap = async () => ({close(){}});
const disk = new Map();
globalThis.caches = {open:async()=>({
  keys:async()=>[...disk.keys()].map(url=>({url})),
  match:async key=>disk.get(String(key))?.clone(),
  put:async(key,response)=>{disk.set(String(key),response.clone());}
})};

for (const concurrency of [8,16,32]) {
  let active=0, peak=0, finished=0;
  await prefetchAll(Array.from({length:70},(_,i)=>i),async()=>{
    peak=Math.max(peak,++active);await delay(1);active--;finished++;
  },{concurrency});
  assert.equal(peak,concurrency);assert.equal(finished,70);assert.equal(active,0);
}

let requests=0;
globalThis.fetch=async(_url,{signal}={})=>{
  requests++;await delay(1,undefined,{signal});return new Response('image',{headers:{'Content-Type':'image/jpeg'}});
};
const ds=new Dataset(()=>{});await ds.init();
const rows=Array.from({length:70},(_,id)=>({id,split:'train'}));
const progress=[];
await ds.download([...rows,rows[0]],undefined,16,(done,total)=>progress.push([done,total]));
assert.equal(requests,70);assert.deepEqual(progress[0],[0,70]);assert.deepEqual(progress.at(-1),[70,70]);
// Completed writes, cache reuse, and every subsequent training read require no network.
await ds.download(rows,undefined,32);
await Promise.all(rows.map(r=>ds.image(r)));
assert.equal(requests,70);
await assert.rejects(ds.image({id:999,split:'train'}),/画像キャッシュがありません/);
assert.equal(requests,70);

// A cache-only reader must not silently join an in-flight download.
const extra={id:1000,split:'train'},pending=ds.image(extra,undefined,{cacheOnly:false});
await assert.rejects(ds.image(extra),/画像キャッシュがありません/);
await pending;

// An aborted or failed prefetch never releases the training barrier early.
let active=0, trainingUpdates=0;
globalThis.fetch=async(_url,{signal}={})=>{
  active++;try{await delay(20,undefined,{signal});throw Error('upstream failure');}finally{active--;}
};
await assert.rejects((async()=>{await ds.download(rows.map(r=>({...r,id:r.id+2000})),undefined,8);trainingUpdates++;})(),/upstream failure/);
assert.equal(active,0);assert.equal(trainingUpdates,0);assert.equal(ds.pending.size,0);
const controller=new AbortController();
const aborted=prefetchAll(rows,async(_r,signal)=>{
  active++;try{await delay(20,undefined,{signal});}finally{active--;}
},{signal:controller.signal,concurrency:16});
controller.abort();await assert.rejects(aborted);assert.equal(active,0);
await assert.rejects(prefetchAll(rows,()=>{throw Error('must not run');},{signal:controller.signal}),{name:'AbortError'});

// Exercise the actual teacher SHA validation and persistent cache with a real JPEG.
disk.clear();const cache=new TeacherCache();
const files=Object.keys(TEACHERS).filter(f=>f.endsWith('.jpg')).slice(0,2);
assert.equal(files.length,2);requests=0;
globalThis.fetch=async url=>{
  requests++;return new Response(await readFile(new URL('../public'+new URL(url).pathname,import.meta.url)));
};
await assert.rejects(cache.bytes(files[0],undefined,{cacheOnly:true}),/動画教師キャッシュがありません/);
assert.equal(requests,0);
await cache.prefetch([files[0],files[0]],undefined,8);
await cache.bytes(files[0],undefined,{cacheOnly:true});assert.equal(requests,1);
// Corrupted cached bytes are not passed to training or repaired with a hidden fetch.
for(const key of disk.keys())disk.set(key,new Response('corrupt'));
await assert.rejects(cache.bytes(files[0],undefined,{cacheOnly:true}),/動画教師キャッシュがありません/);
assert.equal(requests,1);
await cache.prefetch(files,undefined,8);assert.equal(requests,3);
await Promise.all(files.map(f=>cache.bytes(f,undefined,{cacheOnly:true})));assert.equal(requests,3);

const manifest=JSON.parse(await readFile(new URL('../public/paired.json',import.meta.url),'utf8'));
assert.equal(videoFiles({manifest}).length,5009);
const app=await readFile(new URL('../public/app.mjs',import.meta.url),'utf8');
assert.match(app,/await ensureData\(\);if\(state.phase<=2\)await ensureImages\(\[\.\.\.trainRows,\.\.\.valRows\]\);\s*while\(state.phase<4\)/);
assert.match(app,/await ensureImages\(valRows\);await evaluate2d\(\)/);
const video=await readFile(new URL('../public/video-workflow.mjs',import.meta.url),'utf8');
const train=video.slice(video.indexOf('async function train('));
assert(train.indexOf('await prefetchVideo(')<train.indexOf('await videoBatch('));
console.log('PASS: 8/16/32 parallelism; deduplication; persistent cache; cache-only training; failure/abort barrier; real teacher hash validation; all 5,009 video frames; training/evaluation entry points.');
