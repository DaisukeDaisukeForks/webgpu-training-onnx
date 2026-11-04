import {Student37} from './student.mjs';
import {SPEC,K,IDS,EDGES,Z_RANGE} from './schema.mjs';
import {openTeacher,labelCrop,sha256} from './teacher.mjs';
import {openStore,disk,keysFor} from './storage.mjs';
import {personCrop,realLabels,sourceRows,imageSource,batch,disposeBatch} from './data.mjs';
import {step,seeded,shuffled} from './train.mjs';
import {readWeights,assignWeights,readAdam,restoreAdam} from '../snapshot.mjs?revision=17';
import {readCheckpoint} from '../recovery.mjs?revision=17';
const $=id=>document.getElementById(id),tf=window.tf;
let ort,db,teacher,teacherFile,model,optimizer,ds,busy=false,stopping=false,backupPending=false,sourceImage,videoURL,lastSave=0,lastFile=0,runStarted=0;
let state={epoch:0,cursor:0,order:[],updates:0,rng:20261104,milliseconds:0,teacherHash:null,history:[]};
const dirty=new Set(),fields=['epochs','batch','lr','seed'];for(const k of fields)$(k).addEventListener('input',()=>dirty.add(k));
function log(message){$('log').textContent+=`${new Date().toLocaleTimeString()} ${message}\n`;$('log').scrollTop=$('log').scrollHeight;}
function status(message){$('status').textContent=message;log(message);}
function controls(){for(const k of ['prepare','label-data','label-video','teacher-view','train','restore','infer','evaluate','export','report'])$(k).disabled=busy||(!['prepare','restore'].includes(k)&&!model)||(k==='export'&&!state.updates);$('teacher-file').disabled=busy;$('checkpoint').disabled=busy;for(const k of fields)$(k).disabled=busy;$('stop').disabled=!busy;$('backup').disabled=!model;}
function config(){const c=Object.fromEntries(fields.map(k=>[k,Number($(k).value)]));if(!Number.isInteger(c.epochs)||c.epochs<1||!Number.isInteger(c.batch)||c.batch<1||c.batch>64||!Number.isFinite(c.lr)||c.lr<=0||c.lr>.01||!Number.isInteger(c.seed))throw Error('学習設定を確認してください');return c;}
function download(name,data,type='application/octet-stream'){const url=URL.createObjectURL(new Blob([data],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);}
function elapsed(){if(runStarted){state.milliseconds+=performance.now()-runStarted;runStarted=performance.now();}return state.milliseconds;}
async function pack(){elapsed();const adam=await readAdam(optimizer);return{format:SPEC.format,savedAt:new Date().toISOString(),spec:SPEC,config:config(),state:structuredClone(state),weights:await readWeights(model),adam,training_hours:Number((state.milliseconds/3600000).toFixed(1))};}
async function save(file=false){const p=await pack();await disk(db,'state','put','latest',p);lastSave=performance.now();if(file){download(`rtmw3d-distill37-${new Date().toISOString().replace(/[:.]/g,'-')}.json`,JSON.stringify(p),'application/json');lastFile=state.milliseconds;backupPending=false;}}
async function runtime(){
  if(!navigator.gpu)throw Error('WebGPU対応ブラウザで開いてください');
  if(!ort){const base='https://huggingface.co/RabbitDaisuke/rtmw3d-webgpu-onnx/resolve/main/runtime/';ort=await import(base+'ort.all.min.mjs');ort.env.wasm.wasmPaths=base;ort.env.wasm.numThreads=1;}
  if(!await tf.setBackend('webgpu'))throw Error('WebGPU学習を初期化できません');await tf.ready();
  db??=await openStore();if(!model){const c=config();state.rng=c.seed;model=new Student37(tf,c.seed);optimizer=tf.train.adam(c.lr);log(`軽量モデル ${model.vars.reduce((s,t)=>s+t.size,0).toLocaleString()} パラメータ / 37点`);}
}
async function ensureTeacher(){
  teacherFile=$('teacher-file').files[0]||teacherFile;if(!teacherFile)throw Error('raw SimCC教師ONNXを選んでください');
  if(teacher&&teacher.name===teacherFile.name&&teacher.file===teacherFile)return;
  if(teacher)await teacher.session.release();teacher=null;
  const next=await openTeacher(ort,teacherFile);if(state.teacherHash&&state.teacherHash!==next.hash){await next.session.release();throw Error('復元状態と教師が違います。同じ教師ONNXを選んでください。');}
  teacher={...next,file:teacherFile};state.teacherHash=teacher.hash;$('teacher-status').textContent=`${teacher.name} / WebGPU / ${teacher.hash.slice(0,12)}`;
}
async function counts(){if(!db||!state.teacherHash)return;const keys=await keysFor(db,state.teacherHash);$('cache-status').textContent=`保存済み教師 ${keys.length.toLocaleString()}例`;return keys;}
async function task(fn){if(busy)return;busy=true;stopping=false;controls();try{await fn();}catch(e){status(`処理失敗: ${e.message}`);console.error(e);}finally{busy=false;runStarted=0;controls();}}
async function prepare(){await runtime();await ensureTeacher();await counts();status('WebGPUの準備完了');}
async function storeCrop(key,canvas,extra){if(await disk(db,'samples','get',key))return false;const label=await labelCrop(ort,teacher,canvas);await disk(db,'samples','put',key,{...extra,...label,image:await canvas.convertToBlob({type:'image/jpeg',quality:.96}),teacher:teacher.hash});return true;}
async function generateImages(){
  await ensureTeacher();ds??=await imageSource(log);const split=$('split').value,rows=await sourceRows(split),limit=Math.min(rows.length,Math.max(1,Math.floor(Number($('limit').value))));if(!Number.isFinite(limit))throw Error('取得する例数が不正です');
  for(let i=0;i<limit&&!stopping;i++){
    const r=rows[i],identity=`${r.split}:${r.id}:${r.bbox.join(',')}`,key=`${teacher.hash}:${split}:image:${identity}`;
    if(!await disk(db,'samples','get',key)){
      const im=await createImageBitmap(await ds.image(r,undefined,{cacheOnly:false}));try{const {canvas,rect}=personCrop(im,r.bbox);await storeCrop(key,canvas,{split,source:identity,...realLabels(r,rect)});}finally{im.close();}
    }
    $('progress').value=(i+1)/limit;$('status').textContent=`教師を保存 ${i+1}/${limit}`;if(i%20===0)await counts();
  }
  await counts();status(stopping?'教師作成を停止。保存済みの続きから再開できます。':'教師データの保存完了');
}
function roi(image){const a=$('roi').value.split(',').map(Number),w=image.videoWidth||image.width,h=image.videoHeight||image.height;if(a.length!==4||a.some(v=>!Number.isFinite(v))||a[2]<=0||a[3]<=0)throw Error('人物枠は x,y,幅,高さ の4数値です');return[a[0]*w,a[1]*h,a[2]*w,a[3]*h];}
async function seek(video,t){if(Math.abs(video.currentTime-t)<.001&&video.readyState>=2)return;await new Promise((resolve,reject)=>{const done=()=>{cleanup();resolve();},bad=()=>{cleanup();reject(Error('動画を読み取れません'));},timeout=setTimeout(bad,15000),cleanup=()=>{clearTimeout(timeout);video.removeEventListener('seeked',done);video.removeEventListener('error',bad);};video.addEventListener('seeked',done);video.addEventListener('error',bad);video.currentTime=t;});}
async function generateVideo(){
  await ensureTeacher();const f=$('video-file').files[0],video=$('video');if(!f||!Number.isFinite(video.duration))throw Error('動画を選び、読み込み完了を待ってください');
  const fps=Number($('fps').value);if(!Number.isFinite(fps)||fps<=0||fps>30)throw Error('フレーム数/秒は0より大きく30以下にしてください');
  const source=await sha256(await f.arrayBuffer()),split=$('split').value,old=await disk(db,'state','get','video:'+source);if(old&&old!==split)throw Error('同じ動画を学習と評価に分けて登録できません。別の撮影動画を評価に使ってください。');await disk(db,'state','put','video:'+source,split);
  const box=roi(video),rectId=box.join(','),count=Math.ceil(video.duration*fps);video.pause();
  for(let i=0;i<count&&!stopping;i++){const t=Math.min(video.duration-.001,i/fps),key=`${teacher.hash}:${split}:video:${source}:${rectId}:${t.toFixed(4)}`;if(!await disk(db,'samples','get',key)){await seek(video,t);const {canvas}=personCrop(video,box);await storeCrop(key,canvas,{split,source,time:t,frameRate:fps});} $('progress').value=(i+1)/count;$('status').textContent=`動画の教師を保存 ${i+1}/${count}`;if(i%20===0)await counts();}
  await counts();status(stopping?'動画の教師作成を停止しました':'動画の教師データを保存しました');
}
async function train(){
  const c=config();if(!state.teacherHash)throw Error('教師を準備するか復元JSONを読み込んでください');
  if(teacher){await teacher.session.release();teacher=null;}
  const all=await counts(),keys=[];for(const key of all)if(key.startsWith(state.teacherHash+':train:'))keys.push(key);if(!keys.length)throw Error('学習用の教師データがありません');
  const present=new Set(keys);if(state.order.some(k=>!present.has(k)))throw Error('続きの教師データが不足しています。元の教師データを作成してから再開してください。');
  const random=seeded(state);if(dirty.has('seed')){state.rng=c.seed;dirty.delete('seed');}runStarted=performance.now();lastSave=performance.now();lastFile=state.milliseconds;
  try{
    while(state.epoch<c.epochs&&!stopping){
      if(!state.order.length){state.order=shuffled(keys,random);state.cursor=0;}
      while(state.cursor<state.order.length&&!stopping){
        const chosen=state.order.slice(state.cursor,state.cursor+c.batch),records=await Promise.all(chosen.map(k=>disk(db,'samples','get',k)));if(records.some(r=>!r))throw Error('保存した教師が見つかりません');
        const rngBefore=state.rng,b=await batch(tf,records,random);let loss;
        try{optimizer.learningRate=c.lr*Math.min(1,(state.updates+1)/100);loss=await step(tf,model,optimizer,b);}catch(e){state.rng=rngBefore;throw e;}finally{disposeBatch(b);}
        state.cursor+=chosen.length;state.updates++;state.loss=loss;elapsed();$('progress').value=(state.epoch+state.cursor/state.order.length)/c.epochs;
        $('status').textContent=`蒸留 ${state.epoch+1}/${c.epochs} epoch · ${state.cursor}/${state.order.length}`;$('stats').textContent=`損失 ${loss.toFixed(5)} · ${state.updates.toLocaleString()}更新 · ${(state.milliseconds/3600000).toFixed(1)}時間`;
        if(backupPending||state.milliseconds-lastFile>=1800000||performance.now()-lastSave>=60000)await save(backupPending||state.milliseconds-lastFile>=1800000);
      }
      if(state.cursor===state.order.length){state.epoch++;state.cursor=0;state.order=[];state.history.push({epoch:state.epoch,loss:state.loss,hours:state.milliseconds/3600000});await save();}
    }
  }finally{elapsed();runStarted=0;await save(backupPending);}
  status(stopping?'停止して学習状態を保存しました':'指定エポックまで蒸留を完了しました');
}
async function restore(p){
  if(p?.format!==SPEC.format||p.spec?.coco133_ids?.join(',')!==IDS.join(','))throw Error('37点蒸留モデルの復元JSONを選んでください。WB73とは別モデルです。');
  await runtime();if(!p.state?.teacherHash||!Number.isInteger(p.state.epoch)||p.state.epoch<0||!Array.isArray(p.state.order)||!Number.isInteger(p.state.cursor)||p.state.cursor<0||p.state.cursor>p.state.order.length)throw Error('保存した学習位置が不正です');
  if(teacher&&teacher.hash!==p.state.teacherHash){await teacher.session.release();teacher=null;teacherFile=null;$('teacher-file').value='';}
  const next=new Student37(tf,p.config.seed);let opt;
  try{assignWeights(tf,next,p.weights);opt=await restoreAdam(tf,p.adam.weights,p.adam.metadata,next.vars,p.config.lr);}catch(e){next.dispose();throw e;}
  model.dispose();optimizer.dispose();model=next;optimizer=opt;state=structuredClone(p.state);
  for(const k of fields)if(!dirty.has(k))$(k).value=p.config[k];lastFile=state.milliseconds;await counts();status(`${state.epoch} epoch / ${state.updates}更新から復元しました`);
}
function draw(canvas,points,teacherMode=false){const view=$('overlay'),c=view.getContext('2d');view.width=576;view.height=768;c.drawImage(canvas,0,0,576,768);const sx=teacherMode?2:3,sy=teacherMode?2:3,p=teacherMode?IDS.map(i=>[points[i*3]*sx,points[i*3+1]*sy]):Array.from({length:K},(_,i)=>[points[i*3]*sx,points[i*3+1]*sy]);c.lineWidth=2;for(const[a,b]of EDGES){c.strokeStyle=a>=15&&b>=15?'#ffcb55':'#80d9ff';c.beginPath();c.moveTo(...p[a]);c.lineTo(...p[b]);c.stroke();}p.forEach(([x,y],i)=>{c.fillStyle=i>=17&&i<23?'#ffcb55':'#a6e77c';c.beginPath();c.arc(x,y,3,0,Math.PI*2);c.fill();});}
function inputCrop(){const image=sourceImage||($('video').readyState>=2?$('video'):null);if(!image)throw Error('画像または動画を選んでください');return personCrop(image,roi(image)).canvas;}
async function infer(){
  const canvas=inputCrop(),c=new OffscreenCanvas(192,256);c.getContext('2d').drawImage(canvas,0,0,192,256);const image=tf.tidy(()=>tf.browser.fromPixels(c).toFloat().div(255).expandDims(0));let out;
  try{out=model.predict(image);await out.keypoints.data();Object.values(out).forEach(t=>t.dispose());out=null;const start=performance.now();out=model.predict(image);const points=await out.keypoints.data();$('latency').textContent=`軽量モデル ${ (performance.now()-start).toFixed(1)} ms（ウォームアップ後・読戻し込み）`;draw(canvas,points);}finally{image.dispose();if(out)Object.values(out).forEach(t=>t.dispose());}
}
async function evaluate(){
  const keys=(await counts()).filter(k=>k.startsWith(state.teacherHash+':val:'));if(!keys.length)throw Error('評価用の教師データを先に作成してください');
  const totals={body:[0,0,0,0,0,0],feet:[0,0,0,0,0,0],hands:[0,0,0,0,0,0]};let done=0;
  for(const key of keys){if(stopping)break;const r=await disk(db,'samples','get',key),b=await batch(tf,[r],()=>.5,false);let p;
    try{p=model.predict(b.image);const values=await p.keypoints.data();for(let j=0;j<K;j++){const t=totals[j<17?'body':j<23?'feet':'hands'];if(r.mask[j]){const dx=values[j*3]/192-r.position[j*3],dy=values[j*3+1]/256-r.position[j*3+1];t[0]+=Math.hypot(dx,dy);t[1]+=Math.abs(values[j*3+2]-(r.position[j*3+2]*2-1)*Z_RANGE)*1000;t[2]++;}if(r.gtMask?.[j]){t[3]+=Math.hypot(values[j*3]/192-r.gtXY[j*2],values[j*3+1]/256-r.gtXY[j*2+1])<=.05?1:0;t[4]++;}}}finally{disposeBatch(b);if(p)Object.values(p).forEach(t=>t.dispose());}done++;$('progress').value=done/keys.length;
  }
  state.evaluation={at:new Date().toISOString(),epoch:state.epoch,frames:done,groups:Object.fromEntries(Object.entries(totals).map(([k,t])=>[k,{teacherXYMean:t[2]?t[0]/t[2]:null,teacherZMeanMM:t[2]?t[1]/t[2]:null,teacherPoints:t[2],gtPCK05:t[4]?t[3]/t[4]:null,gtPoints:t[4]}]))};
  $('metrics').textContent=Object.entries(state.evaluation.groups).map(([name,g])=>`${name}: 教師Z差 ${g.teacherZMeanMM?.toFixed(1)??'—'}mm / 正解2D PCK ${g.gtPCK05===null?'—':(g.gtPCK05*100).toFixed(1)+'%'}`).join(' · ');await save();status('評価を保存しました。教師との差は実際の3D正解に対する誤差ではありません。');
}
for(const[id,fn]of Object.entries({prepare,'label-data':generateImages,'label-video':generateVideo,train,infer,evaluate,'teacher-view':async()=>{await ensureTeacher();const canvas=inputCrop(),r=await labelCrop(ort,teacher,canvas);draw(canvas,r.allPoints,true);},restore:async()=>{await runtime();const p=await disk(db,'state','get','latest');if(!p)throw Error('保存した学習がありません');await restore(p);},export:async()=>{download('distill37.onnx',await model.export());download('distill37.schema.json',JSON.stringify(SPEC,null,2),'application/json');},report:async()=>download('distill37-result.json',JSON.stringify({spec:SPEC,state,training_hours:Number((elapsed()/3600000).toFixed(1))},null,2),'application/json')}))$(id).onclick=()=>task(fn);
$('stop').onclick=()=>{stopping=true;};$('backup').onclick=()=>{if(busy){backupPending=true;log('現在のバッチが完了したら復元JSONを保存します');}else task(()=>save(true));};
$('checkpoint').onchange=()=>task(async()=>{await restore(await readCheckpoint($('checkpoint').files[0]));await train();});
$('video-file').onchange=()=>{sourceImage?.close();sourceImage=null;if(videoURL)URL.revokeObjectURL(videoURL);const f=$('video-file').files[0];if(f){videoURL=URL.createObjectURL(f);$('video').src=videoURL;}};
$('image-file').onchange=async()=>{sourceImage?.close();sourceImage=await createImageBitmap($('image-file').files[0]);};
window.addEventListener('beforeunload',e=>{if(busy){e.preventDefault();e.returnValue='';}});controls();
