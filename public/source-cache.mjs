import {prefetchAll} from './prefetch.mjs?revision=8';
import {SHA256} from './sha256.mjs?revision=10';
export async function sourceArchive(name,item,signal,progress=()=>{}){
  signal?.throwIfAborted();let disk;try{disk=await caches.open('wholebody73-source-assets-v1');}catch{}
  const parts=Array(Math.ceil(item.bytes/item.chunkBytes)),keys=[],stats={hits:0,network:0,edgeHits:0};let completed=0;
  await prefetchAll(Array.from({length:parts.length},(_,i)=>i),async(i,s)=>{
    const url=new URL(`./api/teacher-assets/${name}/${i}`,location.href);url.searchParams.set('source-sha256',item.sha256);keys[i]=url.href;
    const size=Math.min(item.chunkBytes,item.bytes-i*item.chunkBytes);let blob;
    if(disk){const hit=await disk.match(url.href);if(hit?.ok){const cached=await hit.blob();if(cached.size===size){blob=cached;stats.hits++;}}}
    if(!blob){const response=await fetch(url.href,{signal:s,cache:'no-store',credentials:'same-origin'});if(!response.ok)throw Error(`${name}: Sites HTTP ${response.status}`);if(response.headers.get('X-Teacher-SHA256')!==item.sha256)throw Error('教師データ版が変わりました。ページを再読み込みしてください');blob=await response.blob();if(blob.size!==size)throw Error(`${name}: 教師チャンクのサイズ不一致`);stats.network++;if(response.headers.get('X-Teacher-Cache')==='HIT')stats.edgeHits++;if(disk)try{await disk.put(url.href,new Response(blob));}catch{progress(completed,item.bytes,'永続保存の空きが不足しています。このセッションでは取得済みデータを保持します。');}}
    parts[i]=blob;completed+=blob.size;progress(completed,item.bytes,`${name} · ブラウザ ${stats.hits} / Cloudflare ${stats.edgeHits} / 取得 ${stats.network}チャンク`);
  },{signal,concurrency:4});
  const hash=new SHA256();let checked=0;
  for(const part of parts){signal?.throwIfAborted();hash.update(new Uint8Array(await part.arrayBuffer()));checked+=part.size;progress(checked,item.bytes,`${name} · SHA-256照合`);}
  if(hash.hex()!==item.sha256){if(disk)for(const key of keys)await disk.delete(key);throw Error(`${name}: 公式SHA-256と一致しません。破損したキャッシュを削除しました。`);}
  return new Blob(parts);
}
