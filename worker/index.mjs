// Dataset transport only. Training, gradients and inference run in the browser.
const teacherPending=new Map();
async function teacherAsset(request,url,assets){
  if(request.method!=='GET'&&request.method!=='HEAD')return new Response('Method not allowed',{status:405});
  if(url.pathname==='/api/teacher-assets/index')return Response.json(assets,{headers:{'Cache-Control':'no-cache'}});
  const match=url.pathname.match(/^\/api\/teacher-assets\/([a-z0-9-]+)\/(\d+)$/),item=match&&assets[match[1]],index=match&&Number(match[2]);
  if(!item||!Number.isSafeInteger(index)||index<0||index>=Math.ceil(item.bytes/item.chunkBytes))return new Response('Unknown dataset chunk',{status:404});
  const start=index*item.chunkBytes,end=Math.min(item.bytes,start+item.chunkBytes)-1,length=end-start+1;
  // Cache each complete chunk as 200, since Cache.put rejects 206 responses.
  const key=new Request(`${url.origin}/api/teacher-cache/${match[1]}/${item.sha256}/${index}`),cache=await caches.open('wholebody73-teacher-chunks-v1');
  const hit=await cache.match(key);
  if(hit){const response=new Response(request.method==='HEAD'?null:hit.body,hit);response.headers.set('X-Teacher-Cache','HIT');return response;}
  if(request.method==='HEAD')return new Response(null,{headers:{'Content-Length':String(length),'X-Teacher-Cache':'MISS','Cache-Control':'no-store'}});
  const id=key.url;
  if(!teacherPending.has(id)){
    const work=(async()=>{
      const range=item.bytes>item.chunkBytes,headers={'Accept-Encoding':'identity'};
      if(range)headers.Range=`bytes=${start}-${end}`;
      const upstream=await fetch(item.url,{headers,redirect:'follow'});
      if(upstream.status!==(range?206:200)) {await upstream.body?.cancel();throw Error(`Dataset upstream HTTP ${upstream.status}`);}
      if(range&&upstream.headers.get('Content-Range')!==`bytes ${start}-${end}/${item.bytes}`){await upstream.body?.cancel();throw Error('Dataset range mismatch');}
      const bytes=await upstream.arrayBuffer();
      if(bytes.byteLength!==length)throw Error('Dataset chunk length mismatch');
      const response=new Response(bytes,{headers:{'Content-Type':'application/octet-stream','Content-Length':String(length),'Cache-Control':'public, max-age=31536000, immutable','X-Content-Type-Options':'nosniff','X-Teacher-Cache':'MISS','X-Teacher-SHA256':item.sha256,'X-Teacher-Chunk':String(index)}});
      // Wait for the cache write before releasing concurrent readers.
      await cache.put(key,response.clone());
      return response;
    })();teacherPending.set(id,work);work.finally(()=>teacherPending.delete(id)).catch(()=>{});
  }
  try{return (await teacherPending.get(id)).clone();}catch(error){console.error('teacher_asset_fetch_failed',{asset:match[1],chunk:index,message:String(error)});return new Response(String(error),{status:502,headers:{'Cache-Control':'no-store'}});}
}
export function handler(allowed,assets={}){return {async fetch(request,env,ctx){
  const url=new URL(request.url),match=url.pathname.match(/^\/api\/images\/(train|val)\/(\d{12})\.jpg$/);
  if(url.pathname.startsWith('/api/teacher-assets/'))return teacherAsset(request,url,assets);
  if(url.pathname.startsWith('/api/')){
    if(request.method!=='GET'&&request.method!=='HEAD')return new Response('Method not allowed',{status:405});
    if(!match||!allowed.has(`${match[1]}/${Number(match[2])}`))return new Response('Image not in selected dataset',{status:404});
    // These are public COCO image bytes only; never cache a visitor's request
    // headers, cookies, private assets, errors, or authorization responses.
    const key=`coco-wholebody-v1/${match[1]}2017/${match[2]}.jpg`,cache=env.BUCKET;
    try{if(cache){const hit=await cache.get(key);if(hit){return new Response(request.method==='HEAD'?null:hit.body,{headers:{'Content-Type':hit.httpMetadata?.contentType||'image/jpeg','Cache-Control':'public, max-age=31536000, immutable','X-Content-Type-Options':'nosniff','X-Image-Cache':'HIT'}});}}}catch(e){console.warn('image_cache_read_failed',{image:match[2],message:String(e)});}
    try{const upstream=await fetch(`https://s3.amazonaws.com/images.cocodataset.org/${match[1]}2017/${match[2]}.jpg`,{method:request.method,redirect:'manual'});
      if(!upstream.ok){console.error('s3_image_http_error',{image:match[2],status:upstream.status});return new Response(`S3 returned ${upstream.status}`,{status:502,headers:{'Cache-Control':'no-store'}});}
      if(!upstream.headers.get('Content-Type')?.startsWith('image/'))return new Response('S3 returned a non-image response',{status:502,headers:{'Cache-Control':'no-store'}});
      const response=new Response(upstream.body,{headers:{'Content-Type':upstream.headers.get('Content-Type'),'Cache-Control':'public, max-age=31536000, immutable','X-Content-Type-Options':'nosniff','X-Image-Cache':'MISS'}});
      if(cache&&request.method==='GET'){try{await cache.put(key,response.clone().body,{httpMetadata:{contentType:upstream.headers.get('Content-Type')}});}catch(e){console.warn('image_cache_write_failed',{image:match[2],message:String(e)});response.headers.set('X-Image-Cache','BYPASS');}}
      else if(!cache)response.headers.set('X-Image-Cache','BYPASS');
      return response;
    }catch(e){console.error('s3_image_fetch_failed',{image:match[2],message:String(e)});return new Response(`S3 image fetch failed: ${String(e)}`,{status:502,headers:{'Cache-Control':'no-store'}});}
  }
  return env.ASSETS.fetch(request);
}};}
