import {IDS,K,BINS,SPEC,Z_RANGE} from './schema.mjs';
import {exposeSimCC} from './raw-onnx.mjs';
// Keep the teacher's soft distributions, including secondary depth hypotheses.
export function compressTeacher(raw){
  const position=new Float32Array(K*3),mask=new Float32Array(K),axes=BINS.map(n=>new Uint16Array(K*n));
  const allPoints=new Float32Array(133*3),allScores=new Float32Array(133);
  for(let j=0;j<133;j++){
    const peak=[];
    for(let a=0;a<3;a++){const n=SPEC.teacher.rawBins[a],values=raw[a];let max=-Infinity,index=0;for(let b=0;b<n;b++){const v=values[j*n+b];if(!Number.isFinite(v))throw Error('教師の出力に非有限値があります');if(v>max){max=v;index=b;}}peak.push(max);allPoints[j*3+a]=a===2?(index/n*2-1)*Z_RANGE:index/2;}
    allScores[j]=Math.min(peak[0],peak[1]);
  }
  for(let k=0;k<K;k++){
    const j=IDS[k];mask[k]=allScores[j]>.25?Math.min(1,allScores[j]):0;
    for(let a=0;a<3;a++){
      const n=SPEC.teacher.rawBins[a],out=BINS[a],v=raw[a];let max=-Infinity;for(let i=0;i<n;i++)max=Math.max(max,v[j*n+i]);
      const sums=new Float64Array(out);let total=0;
      // Official head trains with beta=10. Temperature 2 keeps informative alternatives.
      for(let i=0;i<n;i++){const e=Math.exp((v[j*n+i]-max)*5);sums[Math.floor(i*out/n)]+=e;total+=e;}
      for(let i=0;i<out;i++)axes[a][k*out+i]=Math.round(sums[i]/total*65535);
      position[k*3+a]=a===2?(allPoints[j*3+a]/Z_RANGE+1)/2:allPoints[j*3+a]/(a===0?288:384);
    }
  }
  return{position,mask,axes,allPoints,allScores};
}
export function unpackAxis(packed,n){const out=new Float32Array(packed.length);for(let j=0;j<K;j++){let sum=0;for(let i=0;i<n;i++)sum+=packed[j*n+i];if(!sum)throw Error('教師分布が空です');for(let i=0;i<n;i++)out[j*n+i]=packed[j*n+i]/sum;}return out;}
export async function sha256(bytes){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');}
export async function openTeacher(ort,file){
  const bytes=new Uint8Array(await file.arrayBuffer()),hash=await sha256(bytes);
  const session=await ort.InferenceSession.create(exposeSimCC(bytes),{executionProviders:['webgpu'],graphOptimizationLevel:'all'});
  if(!['simcc_x','simcc_y','simcc_z'].every(n=>session.outputNames.includes(n))){await session.release();throw Error('raw SimCC教師ONNXが必要です。tools/distill/export_teacher.pyで作成してください。');}
  return{session,hash,name:file.name};
}
export async function labelCrop(ort,teacher,canvas){
  const c=new OffscreenCanvas(288,384),ctx=c.getContext('2d',{willReadFrequently:true});ctx.drawImage(canvas,0,0,288,384);
  const rgba=ctx.getImageData(0,0,288,384).data,n=288*384,input=new Float32Array(n*3);
  for(let i=0;i<n;i++)for(let a=0;a<3;a++)input[a*n+i]=(rgba[i*4+a]-SPEC.teacher.mean[a])/SPEC.teacher.std[a];
  const t=new ort.Tensor('float32',input,[1,3,384,288]);let result;
  try{result=await teacher.session.run({input:t});const arrays=[];for(const[a,name]of ['simcc_x','simcc_y','simcc_z'].entries()){const v=result[name];if(v.type!=='float32'||v.dims.join(',')!==`1,133,${SPEC.teacher.rawBins[a]}`)throw Error('教師ONNXの型または形状が違います。FP32の教師を選んでください。');arrays.push(v.data);}return compressTeacher(arrays);}finally{t.dispose();if(result)Object.values(result).forEach(v=>v.dispose());}
}
