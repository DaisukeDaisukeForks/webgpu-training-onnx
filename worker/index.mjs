// Image transport only. Training, gradients and inference never run on the server.
export function handler(allowed){return {async fetch(request,env,ctx){
  const url=new URL(request.url),match=url.pathname.match(/^\/api\/images\/(train|val)\/(\d{12})\.jpg$/);
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
