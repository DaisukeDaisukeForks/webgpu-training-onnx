import {prefetchAll} from './prefetch.mjs?revision=8';
import {TEACHERS} from './teacher-index.mjs?revision=12';
export {TEACHERS};
export class TeacherCache{
 constructor(){this.pending=new Map();this.memory=new Map();this.saved=new Set();this.stats={hits:0,downloads:0,bytes:0,stored:0};this.notice=()=>{};}
 emit(message){this.stats.stored=this.saved.size;this.notice(message,{...this.stats});}
 async init(){if(this.initializing)return this.initializing;this.initializing=(async()=>{try{this.disk=await caches.open('wholebody73-teachers-v1');for(const r of await this.disk.keys())this.saved.add(r.url);}catch{}try{this.legacy=await caches.open('wb73-labels-v1');}catch{}try{this.db=await new Promise((resolve,reject)=>{const r=indexedDB.open('wholebody73-teachers',1);r.onupgradeneeded=()=>r.result.createObjectStore('files');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});for(const k of await this.record('keys'))this.saved.add(k);}catch{}this.emit('教師データの保存先を確認');})();return this.initializing;}
 record(mode,key,value){return new Promise((resolve,reject)=>{const tx=this.db.transaction('files',mode==='put'?'readwrite':'readonly'),s=tx.objectStore('files'),r=mode==='keys'?s.getAllKeys():mode==='put'?s.put(value,key):s.get(key);let result;r.onsuccess=()=>result=r.result;tx.oncomplete=()=>resolve(result);tx.onabort=()=>reject(tx.error||Error('教師キャッシュ保存中断'));tx.onerror=()=>reject(tx.error);});}
 async valid(blob,item){if(!blob||blob.size!==item.bytes)return false;const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await blob.arrayBuffer()))).map(b=>b.toString(16).padStart(2,'0')).join('');return hash===item.sha256;}
 async store(key,blob){let persisted=false;if(this.disk)try{await this.disk.put(key,new Response(blob));persisted=true;}catch{}if(!persisted&&this.db)try{await this.record('put',key,blob);persisted=true;}catch{}if(persisted)this.saved.add(key);else{this.memory.set(key,blob);this.emit('ブラウザの永続保存が利用できないため、教師データはこのセッションのメモリを使用します。');}this.emit();}
 async bytes(file,signal,{cacheOnly=false}={}){signal?.throwIfAborted();await this.init();const item=TEACHERS[file];if(!item)throw Error(`未登録の教師ファイル ${file}`);const canonical=new URL(file,location.href).href,key=new URL(canonical);key.searchParams.set('teacher-sha256',item.sha256);const id=key.href,pendingKey=`${cacheOnly?'cache':'download'}:${id}`;if(this.pending.has(pendingKey))return this.pending.get(pendingKey);const p=(async()=>{
  let blob=this.memory.get(id);if(blob&&!await this.valid(blob,item))blob=null;
  if(!blob&&this.disk)try{blob=await(await this.disk.match(id))?.blob();if(blob&&!await this.valid(blob,item))blob=null;}catch{}
  if(!blob&&this.db)try{blob=await this.record('get',id);if(blob&&!await this.valid(blob,item))blob=null;}catch{}
  if(blob){this.stats.hits++;this.emit();return new Uint8Array(await blob.arrayBuffer());}
  if(file.endsWith('.bin')&&this.legacy){try{blob=await(await this.legacy.match(canonical))?.blob();}catch{}if(await this.valid(blob,item)){await this.store(id,blob);this.stats.hits++;this.emit('既存の3D教師キャッシュを再利用');return new Uint8Array(await blob.arrayBuffer());}}
  if(cacheOnly)throw Error(`動画教師キャッシュがありません。動画学習を再開して事前取得してください: ${file}`);
  signal?.throwIfAborted();
  const transport=item.transport,fetchURL=transport?new URL(transport.file,location.href):new URL(id);if(transport)fetchURL.searchParams.set('teacher-sha256',transport.sha256);const response=await fetch(fetchURL.href,{signal,cache:'no-store'});if(!response.ok)throw Error(`${file}: HTTP ${response.status}`);blob=await response.blob();const receivedBytes=blob.size;if(transport){if(!await this.valid(blob,transport))throw Error(`圧縮教師データのハッシュ・サイズ不一致: ${file}`);blob=await new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).blob();}if(!await this.valid(blob,item))throw Error(`教師データのハッシュ・サイズ不一致: ${file}`);await this.store(id,blob);this.stats.downloads++;this.stats.bytes+=receivedBytes;this.emit();return new Uint8Array(await blob.arrayBuffer());
 })();this.pending.set(pendingKey,p);try{return await p;}finally{this.pending.delete(pendingKey);}}
 async prefetch(files,signal,concurrency=16,onProgress){return prefetchAll([...new Set(files)],(file,s)=>this.bytes(file,s),{signal,concurrency,onProgress});}
 async json(file,signal){return JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(await this.bytes(file,signal)));}
 async persistence(){try{const durable=await navigator.storage.persisted()||await navigator.storage.persist();this.emit(durable?'教師データの永続保持をブラウザが許可しました。':'教師データをブラウザに保存します。保持期間はブラウザの設定に従います。');}catch{}}
}
export const teachers=new TeacherCache();
