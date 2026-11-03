import {WB73} from './model.mjs?revision=7';
import {Video73} from './video-model.mjs';
export async function convertCheckpoint(file){
 const raw=new Uint8Array(await file.arrayBuffer());
 const text=raw[0]===31&&raw[1]===139?await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'))).text():new TextDecoder('utf-8',{fatal:true}).decode(raw);
 const cp=JSON.parse(text);
 if(cp.format!=='wholebody73-webgpu-training-v1'||cp.schema!=='compact73-coco-h3wb-v2'||!Array.isArray(cp.weights))throw Error('対応するWholeBody73チェックポイントではありません');
 const weights=new Map();
 // Construct only tensor descriptors: exporting never initializes random weights.
 const tf={randomNormal:(shape)=>({shape,dispose(){}}),fill:(shape)=>({shape,dispose(){}}),variable:t=>t};
 const temporal=cp.weights.some(w=>w.name.startsWith('video.'));
 const expected=new (temporal?Video73:WB73)(tf);
 if(cp.weights.length!==expected.weights.size)throw Error('重みの数がモデル定義と一致しません');
 for(const w of cp.weights){const e=expected.weights.get(w.name);if(weights.has(w.name)||!e||JSON.stringify(e.shape)!==JSON.stringify(w.shape)||!Array.isArray(w.data)||w.data.length!==w.shape.reduce((a,b)=>a*b,1)||w.data.some(v=>!Number.isFinite(v)))throw Error('不正な重み: '+w.name);const data=Float32Array.from(w.data);weights.set(w.name,{shape:w.shape,data:async()=>data});}
 const model=Object.create((temporal?Video73:WB73).prototype);model.weights=weights;
 return {bytes:await model.export(),temporal,summary:`${cp.state?.updates2??'?'} / ${cp.state?.updates3??'?'} updates · ${weights.size} tensors · ${temporal?'3-frame':'single-frame'}`};
}
