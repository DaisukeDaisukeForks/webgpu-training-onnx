import {modelCard} from './publication-card.mjs?revision=12';
export async function publishModel({repo,token,bytes,spec,progress}){
 if(!/^[A-Za-z0-9][A-Za-z0-9_-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(repo))throw Error('公開先は owner/model-name の形式で指定してください');
 const headers={Authorization:`Bearer ${token}`},request=async(path,options={})=>{const r=await fetch(`https://huggingface.co${path}`,{...options,headers:{...headers,...options.headers},redirect:'error'});if(!r.ok)throw Error(`Hugging Face HTTP ${r.status}: ${(await r.text()).slice(0,250)}`);return r;};
 const who=await(await request('/api/whoami-v2')).json(),[owner,name]=repo.split('/');progress('公開先を確認',0,1);
 const exists=await fetch(`https://huggingface.co/api/models/${repo}`,{headers,redirect:'error'});if(exists.status===404){await request('/api/repos/create',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:'model',name,private:false,...(owner===who.name?{}:{organization:owner})})});}else if(!exists.ok)throw Error(`公開先確認 HTTP ${exists.status}`);
 const b64=data=>{let text='';for(let i=0;i<data.length;i+=32768)text+=String.fromCharCode(...data.subarray(i,i+32768));return btoa(text);};
 const card=modelCard(spec);
 const files=[['wholebody73-vrc-trained.onnx',bytes],['model-spec.json',new TextEncoder().encode(JSON.stringify(spec,null,2))],['README.md',new TextEncoder().encode(card)],['H3WB-LICENSE.txt',new Uint8Array(await(await fetch('H3WB-LICENSE.txt')).arrayBuffer())]];
 const lines=[JSON.stringify({key:'header',value:{summary:'Publish independently trained WholeBody73 VRC model'}}),...files.map(([path,data])=>JSON.stringify({key:'file',value:{path,encoding:'base64',content:b64(data)}}))];progress('学習済みONNXを公開',0,1);
 await request(`/api/models/${repo}/commit/main`,{method:'POST',headers:{'Content-Type':'application/x-ndjson'},body:lines.join('\n')+'\n'});
 const saved=await fetch(`https://huggingface.co/${repo}/resolve/main/wholebody73-vrc-trained.onnx`,{headers,redirect:'follow'});if(!saved.ok)throw Error(`公開後の取得 HTTP ${saved.status}`);const downloaded=new Uint8Array(await saved.arrayBuffer());if(downloaded.length!==bytes.length)throw Error('公開済みONNXのサイズ不一致');const digest=async d=>new Uint8Array(await crypto.subtle.digest('SHA-256',d));const[a,b]=await Promise.all([digest(bytes),digest(downloaded)]);if(a.some((v,i)=>v!==b[i]))throw Error('公開済みONNXのハッシュ不一致');return `https://huggingface.co/${repo}`;
}
