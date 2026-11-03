import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {setTimeout as delay} from 'node:timers/promises';
import {handler} from '../worker/index.mjs';
import {sourceArchive} from '../public/source-cache.mjs';
import {SHA256} from '../public/sha256.mjs';
const stores=new Map();
globalThis.caches={open:async name=>{
  if(!stores.has(name))stores.set(name,new Map());const map=stores.get(name),key=k=>typeof k==='string'?k:k.url;
  return{match:async k=>map.get(key(k))?.clone(),put:async(k,r)=>{assert.equal(r.status,200);map.set(key(k),r.clone());},delete:async k=>map.delete(key(k))};
}};
for(const size of [0,1,55,56,63,64,65,127,128,129,1e6]){
  const bytes=crypto.randomBytes(size),hash=new SHA256();for(let i=0;i<size;i+=137)hash.update(bytes.subarray(i,i+137));
  assert.equal(hash.hex(),crypto.createHash('sha256').update(bytes).digest('hex'));
}
const bytes=crypto.randomBytes(101),asset={url:'https://origin.invalid/data.zip',bytes:101,chunkBytes:16,sha256:crypto.createHash('sha256').update(bytes).digest('hex')},worker=handler(new Set(),{sample:asset});
let origins=0,active=0,peak=0;
const origin=async(url,options)=>{
  assert.equal(url,asset.url);assert.equal(options.headers.Cookie,undefined);assert.equal(options.headers.Authorization,undefined);
  origins++;peak=Math.max(peak,++active);await delay(5);active--;
  const match=options.headers.Range.match(/^bytes=(\d+)-(\d+)$/),start=Number(match[1]),end=Number(match[2]);
  return new Response(bytes.subarray(start,end+1),{status:206,headers:{'Content-Range':`bytes ${start}-${end}/${asset.bytes}`}});
};
globalThis.fetch=origin;
const url='https://site.invalid/api/teacher-assets/sample/0';
const responses=await Promise.all(Array.from({length:8},()=>worker.fetch(new Request(url,{headers:{Cookie:'private',Authorization:'private'}}),{})));
assert.equal(origins,1);for(const response of responses)assert.deepEqual(Buffer.from(await response.arrayBuffer()),bytes.subarray(0,16));
assert.equal((await worker.fetch(new Request(url),{})).headers.get('X-Teacher-Cache'),'HIT');
assert.equal((await worker.fetch(new Request(url.replace('/0','/99')),{})).status,404);
const before=origins;await worker.fetch(new Request(url.replace('/0','/1'),{method:'HEAD'}),{});assert.equal(origins,before);
// Exercise client + Worker + origin, then reproduce using only the browser cache.
globalThis.location={href:'https://site.invalid/'};
let siteRequests=0;
globalThis.fetch=async(url,options)=>{
  if(String(url).startsWith('https://origin.invalid/'))return origin(url,options);
  assert.equal(new URL(url).origin,'https://site.invalid');siteRequests++;
  return worker.fetch(new Request(url,options),{});
};
const blob=await sourceArchive('sample',asset);assert.deepEqual(Buffer.from(await blob.arrayBuffer()),bytes);assert.equal(origins,7);assert(peak<=4);
const requests=siteRequests;await sourceArchive('sample',asset);assert.equal(siteRequests,requests);
// A same-size corrupted browser cache is rejected by the full archive hash.
const browser=stores.get('wholebody73-source-assets-v1'),first=browser.keys().next().value;
browser.set(first,new Response(new Uint8Array(16)));
await assert.rejects(sourceArchive('sample',asset),/SHA-256/);assert.equal(browser.size,0);
console.log('PASS: SHA-256 boundaries; named Cloudflare cache; concurrent fetch sharing; no forwarded credentials; exact range; invalid chunks; four-way Sites prefetch; persistent reuse; corruption rejection.');
