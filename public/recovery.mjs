import {WB73} from './model.mjs?revision=7';
const bad=w=>!Array.isArray(w.data)||w.data.some(x=>typeof x!=='number'||!Number.isFinite(x));
export async function readCheckpoint(file){let stream=file.stream();const head=new Uint8Array(await file.slice(0,2).arrayBuffer());if(head[0]===31&&head[1]===139)stream=stream.pipeThrough(new DecompressionStream('gzip'));return JSON.parse(await new Response(stream).text());}
export async function recoverCheckpoint(p,tf,notice=()=>{}){
 if(p?.format!=='wholebody73-webgpu-training-v1')return p;
 const broken=(p.weights||[]).filter(bad),optBroken=(p.opt2||[]).some(bad);
 if(broken.some(w=>!w.name.startsWith('pose.'))||(p.opt3||[]).some(bad))throw Error('3Dの重み・Adamに破損があります。このデータから安全に復元できません。');
 if(!broken.length&&!optBroken)return p;
 const copy=structuredClone(p),fresh=new WB73(tf,p.state.config.seed);try{for(const w of copy.weights)if(w.name.startsWith('pose.'))w.data=Array.from(await fresh.weights.get(w.name).data());}finally{fresh.dispose();}
 copy.opt2=[{name:'iter',shape:[],dtype:'int32',data:[0]}];copy.state.phase=1;copy.state.epoch=0;copy.state.cursor=0;copy.state.order=[];copy.state.updates2=0;copy.state.calibrationIndex=0;copy.residuals=[];delete copy.metrics.pck;delete copy.metrics.pck_counts;
 copy.recovery={reset:'image CNN and 2D Adam',preserved:'all 3D weights, 3D Adam and updates3',reason:'nonfinite pose weights saved by revision 6'};notice('破損した画像CNN・2D Adamだけを初期化しました。3D重み・Adam・更新数は維持して2D学習から再開します。');return copy;
}
