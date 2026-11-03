import {WB73} from './model.mjs?revision=7';import {Dataset,Data3D,crop,batch2d,rng,shuffle} from './data.mjs?revision=17';import {SPEC,GROUPS,NAMES} from './schema.mjs';
import {mountVideo} from './video-workflow.mjs?revision=18';
import {CONFIG_KEYS,applyTrainingSettings,selectNextPhase} from './training-progress.mjs?revision=18';
import {readWeights,assignWeights,readAdam,restoreAdam} from './snapshot.mjs?revision=17';
import {stableStep} from './gradients.mjs?revision=15';
import {recoverCheckpoint,readCheckpoint} from './recovery.mjs?revision=17';
import {publishModel} from './publish.mjs?revision=12';
import {teachers,TEACHERS} from './teacher-cache.mjs?revision=17';
const tf=window.tf,$=id=>document.getElementById(id),dispose=o=>Object.values(o).forEach(t=>t?.dispose?.());let model,ds,d3,trainRows,valRows,footValRows=[],footSource,opt2,opt3,busy=false,abort,state,residuals=[],metrics={},lastSave=0,lastPrediction,loadedImage;
const settingOverrides={},shownSettings={};let restored=false,trainingStartedAt=null,baseRunActive=false,videoWorkflow,fileBackupPending=false,lastFileTrainingMs=null;
const imageDatasetRevision='coco40k-hand-cmu-foot-v1',snapshotFiles=['coco-train.json','cmu-foot-train.json','coco-extra-train-0.json','coco-extra-train-1.json','coco-val.json','cmu-foot-val.json','h3wb.json','aist.json','paired.json'];
const snapshotUI=['prefetch-concurrency','video-epochs','video-batch','yaw','ox','oy','oz'];
const initial=()=>({version:1,phase:0,epoch:0,cursor:0,order:[],rng:Number($('seed').value),updates2:0,updates3:0,calibrationIndex:0,history:[],trainingMilliseconds:0,trainingTimeComplete:true,config:Object.fromEntries(CONFIG_KEYS.map(k=>[k,Number($(k).value)]))});
for(const key of [...CONFIG_KEYS,...snapshotUI]){const field=$(key);shownSettings[key]=field.defaultValue??field.value;for(const event of ['input','change'])field.addEventListener(event,()=>{settingOverrides[key]=field.value;});}
function manualValue(key,saved){if($(key).value!==shownSettings[key])settingOverrides[key]=$(key).value;return settingOverrides[key]??saved;}
function showSetting(key,value){$(key).value=value;shownSettings[key]=$(key).value;}
function recordTrainingTime(){if(!state)return;if(!Number.isFinite(state.trainingMilliseconds)){state.trainingMilliseconds=0;state.trainingTimeComplete=!state.updates2&&!state.updates3;}if(trainingStartedAt!==null){const now=performance.now();state.trainingMilliseconds+=now-trainingStartedAt;trainingStartedAt=now;}$('training-hours').textContent=`${(state.trainingMilliseconds/3600000).toFixed(1)} 時間${state.trainingTimeComplete?'':'（以前の未計測分を除く）'}`;}
function startTrainingTime(){recordTrainingTime();if(lastFileTrainingMs===null)lastFileTrainingMs=state.trainingMilliseconds;if(trainingStartedAt===null)trainingStartedAt=performance.now();}
function stopTrainingTime(){recordTrainingTime();trainingStartedAt=null;}
function trainingTime(){recordTrainingTime();return{training_hours:Number(((state?.trainingMilliseconds||0)/3600000).toFixed(1)),training_time_complete:state?.trainingTimeComplete??true};}
function trainingTimeState(){recordTrainingTime();return{trainingMilliseconds:state.trainingMilliseconds,trainingTimeComplete:state.trainingTimeComplete};}
function restoreTrainingTime(value){if(value&&Number.isFinite(value.trainingMilliseconds)){state.trainingMilliseconds=value.trainingMilliseconds;state.trainingTimeComplete=value.trainingTimeComplete===true;}else state.trainingTimeComplete=false;recordTrainingTime();}
function applySettings(){const before={...state.config};applyTrainingSettings(state,Object.fromEntries(CONFIG_KEYS.map(k=>[k,Number($(k).value)])));if(before.lr2!==state.config.lr2)opt2.learningRate=state.config.lr2;if(before.lr3!==state.config.lr3)opt3.learningRate=state.config.lr3;for(const k of CONFIG_KEYS)showSetting(k,state.config[k]);}
function advancePhase(){if(selectNextPhase(state))residuals=[];}
function autoFileDue(){recordTrainingTime();return lastFileTrainingMs!==null&&state.trainingMilliseconds-lastFileTrainingMs>=30*60*1000;}
function downloadCheckpoint(p,automatic=false){const stamp=new Date().toISOString().replace(/[:.]/g,'-');download(`wholebody73-training-${stamp}.json`,JSON.stringify(p),'application/json');lastFileTrainingMs=state.trainingMilliseconds;fileBackupPending=false;$('file-backup-status').textContent=`${new Date().toLocaleTimeString()} JSONのダウンロードを要求しました。`;log(`${automatic?'30分ごとの自動':'手動'}バックアップJSONを出力。重み・Adam・乱数・途中順序・校正・動画学習を含みます。`);}
async function checkpointBoundary(force=false){if(!force&&!fileBackupPending&&!autoFileDue()&&performance.now()-lastSave<60000)return;try{await save();}catch(e){lastSave=performance.now();log(`チェックポイント保存失敗: ${e.message}`);}}
async function requestBackup(){if(!model)return;if(baseRunActive||videoWorkflow?.isTraining()){fileBackupPending=true;$('file-backup-status').textContent='現在のバッチ完了後にJSONを出力します。';return;}if(busy)return;busy=true;controls();try{applySettings();downloadCheckpoint(await pack());}catch(e){problem(e);}finally{busy=false;controls();}}
const random=()=>{state.rng=(Math.imul(1664525,state.rng)+1013904223)>>>0;return state.rng/4294967296;};
function log(s){$('log').textContent+=`${new Date().toLocaleTimeString()} ${s}\n`;console.log(s);}function status(s){$('status').textContent=s;log(s);}function progress(title,n,total,detail='',eta='—'){recordTrainingTime();$('phase').textContent=title;$('progress').value=total?n/total:0;$('percent').textContent=total?`${(100*n/total).toFixed(1)}%`:'…';$('detail').textContent=detail;$('eta').textContent=eta;$('updates').textContent=`${state?.updates2||0} / ${state?.updates3||0}`;$('cached').textContent=ds?.stats.stored||0;}
function prefetchConcurrency(){const n=Number($('prefetch-concurrency').value);if(![8,16,32].includes(n))throw Error('取得並列数を確認してください');return n;}
function controls(){$('prefetch-concurrency').disabled=busy;for(const k of ['start','save','eval','manifest','frame-map'])$(k).disabled=busy||!model;$('backup').disabled=!model||(busy&&!baseRunActive&&!videoWorkflow?.isTraining());$('backup-video').disabled=$('backup').disabled;$('resume').disabled=busy;$('init').disabled=busy||!!model;$('stop').disabled=!busy||!abort;$('export').disabled=busy||!state?.updates2||!state?.updates3;$('publish').disabled=$('export').disabled;$('infer').disabled=busy||!model||!loadedImage;$('positions').disabled=busy||!lastPrediction;$('checkpoint-file').disabled=busy;for(const k of CONFIG_KEYS)$(k).disabled=busy;for(const k of ['train-video','resume-video','infer-video','video-epochs','video-batch'])$(k).disabled=busy;}
function download(name,data,type='application/octet-stream'){const u=URL.createObjectURL(new Blob([data],{type})),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),30000);}
function problem(e){status(e.name==='AbortError'?'停止しました。保存された途中位置から継続できます。':`処理失敗: ${e.message}`);log(e.stack||String(e));}
async function gradients(fn,vars,optimizer,apply=true){return stableStep(tf,fn,vars,optimizer,apply);}
async function verify(){progress('WebGPU計算・ONNX互換性',0,1);const image=tf.randomUniform([1,256,192,3],0,1,'float32',20261103),target=tf.fill([1,73,2],.5),mask=tf.ones([1,73]),xyz=tf.zeros([1,73,3]);try{await gradients(()=>model.loss2d(image,target,mask),model.poseVars,opt2,false);await gradients(()=>model.loss3d(target,mask,xyz,mask),model.liftVars,opt3,false);log('画像CNNと3D推定器の逆伝播を確認');const bytes=await model.export();ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';ort.env.wasm.numThreads=1;const session=await ort.InferenceSession.create(bytes,{executionProviders:['webgpu']});let expected,rot,origin,input;
 try{rot=tf.eye(3);origin=tf.zeros([1,1,3]);input=image.transpose([0,3,1,2]);expected=model.predict(image,rot,origin);const outputs=await session.run({image:new ort.Tensor('float32',await input.data(),[1,3,256,192]),tracking_rotation:new ort.Tensor('float32',await rot.data(),[3,3]),tracking_origin:new ort.Tensor('float32',await origin.data(),[1,1,3])});let max=0;for(const[n,t]of [['keypoints2d',expected.xy],['confidence2d',expected.confidence],['vrc_relative',expected.relative],['vrc_positions',expected.positions]]){const a=await t.data(),b=outputs[n].data;if(a.length!==b.length)throw Error(`ONNX形状不一致 ${n}`);for(let i=0;i<a.length;i++){if(!Number.isFinite(a[i])||!Number.isFinite(b[i]))throw Error('ONNX出力に非有限値');max=Math.max(max,Math.abs(a[i]-b[i]));}}Object.values(outputs).forEach(t=>t.dispose?.());if(max>.002)throw Error(`ONNX出力差が大きすぎます ${max}`);log(`ONNX / 学習経路の最大出力差 ${max.toExponential(3)}`);
 }finally{if(expected)dispose(expected);rot?.dispose();origin?.dispose();input?.dispose();await session.release();}}finally{[image,target,mask,xyz].forEach(t=>t.dispose());}}
async function prepareModel(check=true){
 try{
  if(!tf||!window.ort)throw Error('学習ライブラリを取得できません');
  ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';ort.env.wasm.numThreads=1;
  if(!navigator.gpu||!await tf.setBackend('webgpu'))throw Error('WebGPUを利用できません');
  await tf.ready();if(tf.getBackend()!=='webgpu')throw Error('WebGPU以外へは切替しません');
  state=initial();model=new WB73(tf,state.config.seed);opt2=tf.train.adam(state.config.lr2);opt3=tf.train.adam(state.config.lr3);
  let info=$('teacher-status');if(!info){info=document.createElement('p');info.id='teacher-status';info.className='subtle';$('status').after(info);}
  teachers.notice=(message,stats)=>{info.textContent=`教師ファイル保存 ${stats.stored}件 / キャッシュ参照 ${stats.hits}回 / 新規取得 ${stats.downloads}件 (${(stats.bytes/2**20).toFixed(1)} MiB)`;if(message)log(message);};
  const metadata=await Promise.all(['coco-train.json','coco-val.json','h3wb.json','cmu-foot-train.json','cmu-foot-val.json','aist.json','coco-extra-train-0.json','coco-extra-train-1.json'].map(name=>teachers.json(name)));
  // Keep every legacy index stable, including the CMU rows at 10000..24613.
  trainRows=[...metadata[0].rows,...metadata[3].rows,...metadata[6].rows,...metadata[7].rows];valRows=metadata[1].rows;footValRows=metadata[4].rows;footSource={train:metadata[3].count,val:metadata[4].count,source:metadata[3].source,mirror:metadata[3].mirror,license:metadata[3].license,train_sha256:metadata[3].source_sha256,val_sha256:metadata[4].source_sha256};const valIds=new Set([...valRows,...footValRows].map(r=>r.id));if(trainRows.some(r=>valIds.has(r.id)))throw Error('2Dデータ分割に重複があります');$('image-count').textContent=new Set(trainRows.map(r=>r.id)).size.toLocaleString();log(`2D教師: COCO-WholeBody ${metadata[0].count+metadata[6].count+metadata[7].count}例 + CMU Human Foot ${metadata[3].count}例。追加${metadata[6].hand_annotations+metadata[7].hand_annotations}例すべてに手指ラベルがあります。検証は画像IDで分離済み。`);
  ds=new Dataset((s,n,total,bytes)=>{if(s)log(s);if(n)progress('画像の事前取得',n,total,`保存${ds.stats.stored}枚 / キャッシュ読込${ds.stats.hits}回 / 通信${(bytes/2**20).toFixed(1)} MiB`);});
  await ds.init();d3=new Data3D(metadata[2],(n,total)=>progress('3D教師データ取得',n,total,'ハッシュを確認しブラウザ内キャッシュを優先'),metadata[5],progress);
  if(check)await verify();$('gpu').textContent='● WebGPU / 独自モデル';const params=[...model.weights.values()].reduce((s,w)=>s+w.size,0);
  if(check){status(`準備完了。${params.toLocaleString()}パラメータをランダム初期化。既存重みは未使用。学習はまだ開始していません。`);progress('準備完了',1,1,'学習開始または保存済み学習を再開');}
 }catch(e){model?.dispose();model=null;opt2?.dispose();opt3?.dispose();throw e;}
}
async function init(){if(busy)return;busy=true;controls();try{await prepareModel();}catch(e){problem(e);}finally{busy=false;controls();}}
function db(mode,value){return new Promise((resolve,reject)=>{const q=indexedDB.open('wholebody73-training',1);q.onupgradeneeded=()=>q.result.createObjectStore('runs');q.onerror=()=>reject(q.error);q.onsuccess=()=>{const d=q.result,tx=d.transaction('runs',mode==='get'?'readonly':'readwrite'),s=tx.objectStore('runs'),r=mode==='get'?s.get('latest'):s.put(value,'latest');let result;r.onsuccess=()=>result=r.result;tx.oncomplete=()=>{d.close();resolve(result);};tx.onerror=()=>{d.close();reject(tx.error);};};});}
async function pack(){
 applySettings();
 const weights=await readWeights(model),adam2=await readAdam(opt2),adam3=await readAdam(opt3),video=await videoWorkflow.pack();
 recordTrainingTime();
 return{format:'wholebody73-webgpu-training-v1',checkpoint_revision:2,schema:SPEC.schema,state:structuredClone(state),weights,opt2:adam2.weights,opt3:adam3.weights,optimizer2:adam2.metadata,optimizer3:adam3.metadata,residuals:structuredClone(residuals),metrics:structuredClone(metrics),video,ui:Object.fromEntries(snapshotUI.map(k=>[k,$(k).value])),display:{loss:$('loss').textContent},teacher_hashes:Object.fromEntries(snapshotFiles.map(k=>[k,TEACHERS[k].sha256])),source_sha256:d3.manifest.sha256,saved_at:new Date().toISOString()};
}
async function save(){
 const automatic=autoFileDue(),p=await pack();let error;
 if(fileBackupPending||automatic)downloadCheckpoint(p,automatic);
 try{await db('put',p);if(p.video)await videoWorkflow.persist(p.video);}catch(e){error=e;}
 lastSave=performance.now();
 if(error)throw error;
 log(`重み・Adam・途中順序・乱数・校正・動画を保存 (${state.updates2}/${state.updates3})`);return p;
}
async function restore(p){
 if(p?.format!=='wholebody73-webgpu-training-v1'||p.schema!==SPEC.schema||p.source_sha256!==d3.manifest.sha256)throw Error('チェックポイントの形式または教師データが一致しません');
 if(p.teacher_hashes)for(const[file,hash]of Object.entries(p.teacher_hashes))if(TEACHERS[file]?.sha256!==hash)throw Error(`保存時の教師と一致しません: ${file}`);
 const options=p.state?.config;
 if(!options||!['ep2','ep3','warm','bs2','bs3','seed'].every(k=>Number.isInteger(options[k])&&options[k]>0)||!['lr2','lr3'].every(k=>Number.isFinite(options[k])&&options[k]>0))throw Error('不正な学習設定');
 if(!Number.isInteger(p.state.phase)||p.state.phase<0||p.state.phase>4||!Number.isInteger(p.state.rng)||!Array.isArray(p.state.order))throw Error('不正な学習位置');
 const requested=Object.fromEntries(CONFIG_KEYS.map(k=>[k,Number(manualValue(k,k==='ep2'&&!p.checkpoint_revision?Math.max(options[k],100):options[k]))])),nextState=structuredClone(p.state);
 applyTrainingSettings(nextState,requested);
 const next=new WB73(tf,options.seed);let next2,next3;
 try{assignWeights(tf,next,p.weights);next2=await restoreAdam(tf,p.opt2,p.optimizer2,next.poseVars,options.lr2);next3=await restoreAdam(tf,p.opt3,p.optimizer3,next.liftVars,options.lr3);if(requested.lr2!==options.lr2)next2.learningRate=requested.lr2;if(requested.lr3!==options.lr3)next3.learningRate=requested.lr3;}
 catch(e){next.dispose();next2?.dispose();next3?.dispose();throw e;}
 model.dispose();opt2.dispose();opt3.dispose();model=next;opt2=next2;opt3=next3;
 state=nextState;restored=true;trainingStartedAt=null;lastFileTrainingMs=null;fileBackupPending=false;residuals=p.residuals||[];metrics=p.metrics||{};
 for(const key of CONFIG_KEYS)showSetting(key,state.config[key]);
 for(const key of snapshotUI)if(p.ui?.[key]!=null)showSetting(key,manualValue(key,p.ui[key]));
 if(p.display?.loss)$('loss').textContent=p.display.loss;
 recordTrainingTime();showMetrics();log(`チェックポイントを復元。phase=${state.phase}, epoch=${state.epoch}, cursor=${state.cursor}, updates=${state.updates2}/${state.updates3}`);
}
async function ensureData(){
 const revision=d3.manifest.revision||'h3wb-subset-v1';
 if(state.datasetRevision!==revision){delete metrics.mpjpe;delete metrics.mpjpe_counts;delete metrics.mpjpe_frame;delete metrics.aspset3d;delete metrics.aist;state.datasetRevision=revision;showMetrics();}
 if(state.imageDatasetRevision!==imageDatasetRevision){log('2D教師を54,614例へ拡張。旧データの番号を維持し、保存された途中順序を完了した次のエポックから全件を使います。');state.imageDatasetRevision=imageDatasetRevision;}
 await teachers.persistence();await d3.load(abort.signal);$('teacher3d-count').textContent=d3.manifest.train_count.toLocaleString();
}
async function ensureImages(rows){const concurrency=prefetchConcurrency(),before={...ds.stats},bytes=ds.bytes;$('speed').textContent='—';status(`画像を${concurrency}並列で事前取得しています。`);await ds.download(rows,abort.signal,concurrency,(n,total)=>progress('画像の事前取得',n,total,`${n}/${total}枚 · ${concurrency}並列 · キャッシュ ${ds.stats.hits-before.hits}枚 / 新規 ${ds.stats.network-before.network}枚 / ${((ds.bytes-bytes)/2**20).toFixed(1)} MiB`));abort.signal.throwIfAborted();log('画像の事前取得完了。学習・校正・評価はキャッシュのみを使用します。');status('事前取得が完了しました。');}
async function evaluate2d(){const values=GROUPS.map(()=>({correct:0,count:0})),foot={correct:0,count:0},rows=[...valRows,...footValRows];for(let i=0;i<rows.length;i++){if(abort.signal.aborted)throw new DOMException('Stopped','AbortError');const s=await crop(ds,rows[i],false,()=>.5,abort.signal),b=batch2d(tf,[s]);let p;try{p=model.pose(b.image);const xy=await p.xy.data();for(let g=0;g<5;g++){if(i>=valRows.length&&g!==3)continue;const score=i>=valRows.length?foot:values[g];for(let j=GROUPS[g][1];j<GROUPS[g][2];j++)if(s.mask[j]){score.count++;if(Math.hypot((xy[j*2]-s.xy[j*2])*.75,xy[j*2+1]-s.xy[j*2+1])<=.0625)score.correct++;}}}finally{if(p)dispose(p);dispose(b);}progress('2D検証 / COCO・CMU Foot',i+1,rows.length);if(baseRunActive)await checkpointBoundary();}metrics.pck=values.map(v=>v.count?v.correct/v.count:null);metrics.pck_counts=values.map(v=>v.count);metrics.cmuFoot={pck:foot.count?foot.correct/foot.count:null,joints:foot.count,annotations:footValRows.length};showMetrics();}
async function evaluate3d(){
 const sums=GROUPS.map(()=>({error:0,count:0})),asp={error:0,count:0},aist={error:0,count:0},primary=d3.manifest.primary_val_count||d3.manifest.val_count;
 for(let start=0;start<d3.manifest.val_count;start+=64){
  if(abort.signal.aborted)throw new DOMException('Stopped','AbortError');const ids=Array.from({length:Math.min(64,d3.manifest.val_count-start)},(_,i)=>start+i),b=d3.batch(tf,'val',ids,()=>.5,false);let pred;
  try{pred=model.lift(b.xy,b.confidence);const[p,t,m]=await Promise.all([pred.data(),b.target.data(),b.mask.data()]);
   for(let n=0;n<ids.length;n++)for(let g=0;g<5;g++)for(let j=GROUPS[g][1];j<GROUPS[g][2];j++)if(m[n*73+j]){
    const k=n*219+j*3,anchor=g===1?4:g===2?5:g===3?(j<55?10:11):g===4?60:-1;let dx=p[k]-t[k],dy=p[k+1]-t[k+1],dz=p[k+2]-t[k+2];
    if(anchor>=0){if(!m[n*73+anchor])continue;const a=n*219+anchor*3;dx-=p[a]-t[a];dy-=p[a+1]-t[a+1];dz-=p[a+2]-t[a+2];}
    const dest=ids[n]<primary?sums[g]:ids[n]<(d3.manifest.base_val_count||d3.manifest.val_count)?asp:aist;dest.error+=1000*Math.hypot(dx,dy,dz);dest.count++;
   }
  }finally{pred?.dispose();dispose(b);}progress('3D検証 / H3WB・ASPset・AIST++',start+ids.length,d3.manifest.val_count);if(baseRunActive)await checkpointBoundary();
 }
 metrics.mpjpe=sums.map(s=>s.count?s.error/s.count:null);metrics.mpjpe_counts=sums.map(s=>s.count);metrics.mpjpe_frame='H3WB: body hip-relative; hands wrist-relative; feet ankle-relative; face nose-relative';metrics.aspset3d={body_mpjpe_mm:asp.count?asp.error/asp.count:null,joint_count:asp.count};metrics.aist={body_mpjpe_mm:aist.count?aist.error/aist.count:null,joint_count:aist.count};showMetrics();
}
function showMetrics(){$('quality').replaceChildren();GROUPS.forEach(([name],i)=>{const r=document.createElement('tr');for(const v of [name,metrics.pck?.[i]!=null?(100*metrics.pck[i]).toFixed(2)+'%':'—',metrics.mpjpe?.[i]!=null?metrics.mpjpe[i].toFixed(1):'—',metrics.mpjpe_counts?.[i]??'—']){const t=document.createElement('td');t.textContent=v;r.append(t);}$('quality').append(r);});const row=document.createElement('tr');for(const v of ['身体 / ASPset','—',metrics.aspset3d?.body_mpjpe_mm!=null?metrics.aspset3d.body_mpjpe_mm.toFixed(1):'—',metrics.aspset3d?.joint_count??'—']){const cell=document.createElement('td');cell.textContent=v;row.append(cell);}$('quality').append(row);$('foot-quality').textContent=metrics.cmuFoot?.pck!=null?`CMU Foot / PCK@0.05: ${(metrics.cmuFoot.pck*100).toFixed(2)}% · ${metrics.cmuFoot.joints.toLocaleString()}点 / ${metrics.cmuFoot.annotations}人`:'CMU Footの2D精度: 未評価';if(metrics.aist){const r=document.createElement('tr');for(const v of ['身体 / AIST++','—',metrics.aist.body_mpjpe_mm?.toFixed(1)??'—',metrics.aist.joint_count]){const c=document.createElement('td');c.textContent=v;r.append(c);}$('quality').append(r);}const pass=metrics.pck?.every((x,i)=>x!=null&&x>=[.8,.5,.5,.7,.8][i])&&metrics.mpjpe?.slice(0,4).every((v,i)=>v!=null&&v<=[140,40,40,60][i]);$('gate').textContent=pass?'表示した指標では設定した基準を通過。実写RGBからVRCへの精度は別途確認が必要です。':'品質基準未達または未評価。実用精度を達成した完成モデルとは判定していません。';}
async function calibrate(){for(;state.calibrationIndex<512;){if(abort.signal.aborted)throw new DOMException('Stopped','AbortError');const s=await crop(ds,trainRows[state.calibrationIndex],false,()=>.5,abort.signal),b=batch2d(tf,[s]);let p;try{p=model.pose(b.image);const[xy,conf]=await Promise.all([p.xy.data(),p.confidence.data()]);let radius=0;const hx=(s.xy[12]+s.xy[14])/2,hy=(s.xy[13]+s.xy[15])/2;for(let j=0;j<12;j++)radius+=((s.xy[j*2]-hx)*.75)**2+(s.xy[j*2+1]-hy)**2;radius=Math.sqrt(radius/12)||.1;const delta=Array.from(xy,(v,i)=>(v-s.xy[i])*(i%2?1:.75)/radius);residuals.push({xy:delta,conf:Array.from(conf,(v,j)=>s.mask[j]?v:0)});}finally{if(p)dispose(p);dispose(b);}state.calibrationIndex++;progress('学習画像の認識誤差を校正',state.calibrationIndex,512);await checkpointBoundary();}advancePhase();await checkpointBoundary(true);}
async function finishEpochEvaluation(){
 const pending=state.pendingEvaluation;if(!pending)return;
 if(pending.evaluated){if(pending.phase===1)await evaluate2d();else await evaluate3d();}
 state.history.push({...pending,updates2:state.updates2,updates3:state.updates3,metrics:pending.evaluated?structuredClone(metrics):null});delete state.pendingEvaluation;await checkpointBoundary(true);
}
async function run(){
 baseRunActive=true;busy=true;abort=new AbortController();controls();let lock;
 try{
  if(!restored&&!state.updates2&&!state.updates3){const requested=initial();if(requested.config.seed!==state.config.seed){model.dispose();model=new WB73(tf,requested.config.seed);}state=requested;opt2.learningRate=state.config.lr2;opt3.learningRate=state.config.lr3;}
  applySettings();advancePhase();const cfg=state.config;
  if(!['ep2','ep3','warm','bs2','bs3'].every(k=>Number.isInteger(cfg[k])&&cfg[k]>0)||!(cfg.lr2>0&&cfg.lr3>0))throw Error('学習設定を確認してください');
  try{lock=await navigator.wakeLock?.request('screen');}catch{}
  await ensureData();if(state.phase<=2||state.pendingEvaluation?.phase===1)await ensureImages([...trainRows,...valRows,...footValRows]);startTrainingTime();await finishEpochEvaluation();
  while(state.phase<4){
   abort.signal.throwIfAborted();if(state.phase===2){await calibrate();continue;}
   const is2d=state.phase===1,epochs=is2d?cfg.ep2:state.phase===0?cfg.warm:cfg.ep3,totalCount=is2d?trainRows.length:d3.manifest.train_count,bs=is2d?cfg.bs2:cfg.bs3;
   const phaseName=is2d?'画像CNNの全層学習':state.phase===0?'3D推定器の準備学習':'実認識誤差を使う3D仕上げ';
   if(state.epoch>=epochs){advancePhase();await checkpointBoundary(true);continue;}
   if(!state.order.length){state.order=shuffle(Array.from({length:totalCount},(_,i)=>i),random);state.epochLearningRate=null;}
   const count=state.order.length;
   if(!Number.isInteger(state.cursor)||state.cursor<0||state.cursor>count||state.order.some(i=>!Number.isInteger(i)||i<0||i>=totalCount))throw Error('保存された学習順序が教師データと一致しません');
   let n=0;const started=performance.now();
   if(!Number.isFinite(state.epochLearningRate)){const fraction=(state.epoch+state.cursor/count)/epochs,decay=.05+.95*(1+Math.cos(Math.PI*fraction))*.5;state.epochLearningRate=(is2d?cfg.lr2:cfg.lr3)*decay;}
   (is2d?opt2:opt3).learningRate=state.epochLearningRate;
   while(state.cursor<count){
    abort.signal.throwIfAborted();const indices=state.order.slice(state.cursor,state.cursor+bs),beforeRng=state.rng;let b;
    try{
     if(is2d){const samples=await Promise.all(indices.map(i=>crop(ds,trainRows[i],true,random,abort.signal)));b=batch2d(tf,samples);$('loss').textContent=(await gradients(()=>model.loss2d(b.image,b.target,b.mask),model.poseVars,opt2)).toFixed(5);state.updates2++;}
     else{b=d3.batch(tf,'train',indices,random,state.phase===3,residuals);$('loss').textContent=(await gradients(()=>model.loss3d(b.xy,b.confidence,b.target,b.mask),model.liftVars,opt3)).toFixed(5);state.updates3++;}
    }catch(e){state.rng=beforeRng;throw e;}finally{if(b)dispose(b);}
    state.cursor+=indices.length;n+=indices.length;const speed=n/((performance.now()-started)/1000);$('speed').textContent=`${speed.toFixed(1)} 例/秒`;
    const remaining=count-state.cursor+(epochs-state.epoch-1)*totalCount;
    progress(`${phaseName} · ${state.epoch+1}/${epochs} epoch`,state.epoch*totalCount+state.cursor,epochs*totalCount,`${state.cursor}/${count}${count!==totalCount?'（旧チェックポイントの途中順序）':''} · GPUテンソル ${(tf.memory().numBytes/2**20).toFixed(0)} MiB`,`残り約${Math.ceil(remaining/speed/60)}分`);
    await checkpointBoundary();await tf.nextFrame();
   }
   state.epoch++;state.cursor=0;state.order=[];state.epochLearningRate=null;state.pendingEvaluation={phase:state.phase,epoch:state.epoch,evaluated:is2d||state.epoch%60===0||state.epoch>=epochs};await checkpointBoundary(true);await finishEpochEvaluation();
  }
  stopTrainingTime();await videoWorkflow.train();progress('設定した学習を完了',1,1,'動画用ONNXを保存できます');status('学習を完了しました。');
 }catch(e){problem(e);}finally{stopTrainingTime();if(model)await checkpointBoundary(true);try{await lock?.release();}finally{abort=null;baseRunActive=false;busy=false;controls();}}
}
$('init').onclick=init;$('start').onclick=()=>run();$('stop').onclick=()=>abort?.abort();
$('save').onclick=async()=>{if(busy)return;busy=true;controls();try{applySettings();await save();status('チェックポイントを保存しました。');}catch(e){problem(e);}finally{busy=false;controls();}};
$('backup').onclick=requestBackup;
async function restoreAndRun(read){
 if(busy)return;busy=true;abort=null;controls();
 try{
  const p=await read();if(!p)throw Error('保存済み学習がありません');
  if(p.format!=='wholebody73-webgpu-training-v1')throw Error('2D・3D学習のバックアップを選択してください');
  if(!model){status('バックアップから再開するためWebGPUを準備しています。');await prepareModel(false);}
  await restore(await recoverCheckpoint(p,tf,log));videoWorkflow.reset();if(p.video)await videoWorkflow.restore(p.video);
  status('重み・Adam・途中位置を復元しました。続きの学習を再開します。');
  await run();
 }catch(e){problem(e);}finally{busy=false;controls();}
}
$('resume').onclick=()=>restoreAndRun(()=>db('get'));
$('checkpoint-file').onchange=async()=>{const input=$('checkpoint-file'),file=input.files[0];if(!file)return;try{await restoreAndRun(()=>readCheckpoint(file));}finally{input.value='';}};
$('eval').onclick=async()=>{busy=true;abort=new AbortController();controls();try{await ensureData();await ensureImages([...valRows,...footValRows]);await evaluate2d();await evaluate3d();await save();}catch(e){problem(e);}finally{busy=false;controls();}};
function report(){return{...SPEC,...trainingTime(),calibration_inputs:[{name:'tracking_rotation',shape:[3,3],meaning:'row-vector Unity-axis rotation from camera-aligned space to calibrated tracking space'},{name:'tracking_origin',shape:[1,1,3],unit:'meter',meaning:'calibrated hip origin; not inferred absolute camera/world position'}],outputs:[...SPEC.outputs.filter(x=>x.name!=='vrc_positions'),{name:'vrc_relative',shape:[1,73,3],unit:'meter, hip-relative Unity axes'},{name:'vrc_positions',shape:[1,73,3],unit:'meter, calibrated Unity tracking space'}],training:{...state,residuals:residuals.length,teacher_weights_used:false,pretrained_weights_used:false},evaluation:metrics,distribution:'ONNX file download generated on this browser; no automatic Hugging Face publication or token transmission',data:{coco_train:40000,coco_train_images:32195,coco_with_hand_labels:32558,coco_val:256,cmu_foot:footSource,h3wb:d3.manifest,aist:d3.manifest.aist},limits:'Monocular relative 3D; metric scale is learned, not a measured absolute distance. Calibration is required for tracker placement. H3WB annotation-only evaluation is not end-to-end RGB-to-VRChat validation.'};}
$('manifest').onclick=()=>download('wholebody73-model-spec.json',JSON.stringify(report(),null,2),'application/json');
$('frame-map').onclick=async()=>{if(!d3)return;busy=true;controls();try{const mapping={};for(const[split,file]of Object.entries(d3.manifest.pairing||{}))mapping[split]=await teachers.json(file);for(const split of ['train','val'])mapping['aspset_'+split]=await teachers.json(`aspset3d-${split}-ids.json`);if(d3.aist?.loaded)mapping.aist=d3.aist.references();download('wholebody73-3d-frame-correspondence.json',JSON.stringify(mapping),'application/json');status('3Dラベルと元フレームの対応表を保存しました。RGB画像本体とRGB→3Dの対応学習は含みません。');}catch(e){problem(e);}finally{busy=false;controls();}};
$('export').onclick=async()=>{busy=true;controls();try{if(!state.updates2||!state.updates3)throw Error('2Dと3Dの実学習が必要です');progress('VRC出力ONNXを書き出し',0,1);const bytes=await model.export();download('wholebody73-vrc-trained.onnx',bytes);progress('ONNX保存完了',1,1,`${(bytes.length/2**20).toFixed(2)} MiB / Unity axes / meter`);status('実学習した独自重みからONNXを生成しました。品質は精度表に示した状態です。');}catch(e){problem(e);}finally{busy=false;controls();}};
// The preview uses the same learned graph and calibration inputs as ONNX.
$('picture').onchange=async()=>{const f=$('picture').files[0];if(!f)return;loadedImage?.close();loadedImage=await createImageBitmap(f);controls();};
function calibration(){const angle=Number($('yaw').value)*Math.PI/180,c=Math.cos(angle),s=Math.sin(angle);return{rotation:[c,0,-s,0,1,0,s,0,c],origin:[Number($('ox').value),Number($('oy').value),Number($('oz').value)]};}
$('infer').onclick=async()=>{busy=true;controls();let image,rotation,origin,pred;try{const canvas=new OffscreenCanvas(192,256),c=canvas.getContext('2d');c.drawImage(loadedImage,0,0,192,256);image=tf.tidy(()=>tf.browser.fromPixels(canvas).toFloat().div(255).expandDims());const cal=calibration();rotation=tf.tensor(cal.rotation,[3,3]);origin=tf.tensor(cal.origin,[1,1,3]);pred=model.predict(image,rotation,origin);const[p,confidence]=await Promise.all([pred.positions.data(),pred.confidence.data()]);lastPrediction={coordinate_system:'Unity left-handed +X right +Y up +Z forward',unit:'meter',space:'calibrated tracking space',calibration:cal,trained:{updates2:state.updates2,updates3:state.updates3},positions:NAMES.map((name,j)=>({name,x:p[j*3],y:p[j*3+1],z:p[j*3+2],confidence2d:confidence[j]}))};$('coords').textContent=JSON.stringify(lastPrediction,null,2);draw(lastPrediction.positions);status('学習中の重みでVRC座標を推定しました。未学習・低精度の状態では、この座標も正しいとは限りません。');}catch(e){problem(e);}finally{image?.dispose();rotation?.dispose();origin?.dispose();if(pred)dispose(pred);busy=false;controls();}};
function draw(points){const c=$('view'),g=c.getContext('2d');g.fillStyle='#10191e';g.fillRect(0,0,960,400);g.fillStyle='#a1b3bc';g.font='14px system-ui';g.fillText('FRONT · X / Y',30,25);g.fillText('SIDE · Z / Y',510,25);for(const p of points){g.fillStyle=p.confidence2d>.1?'#a9ef86':'#60706a';for(const x of [240+p.x*160,720+p.z*160]){g.beginPath();g.arc(x,360-p.y*160,3,0,Math.PI*2);g.fill();}}}
$('positions').onclick=()=>download('wholebody73-vrc-positions.json',JSON.stringify(lastPrediction,null,2),'application/json');window.addEventListener('beforeunload',e=>{if(busy){e.preventDefault();e.returnValue='';}});controls();
videoWorkflow=mountVideo({tf,prefetchConcurrency,manualValue,showSetting,applySettings,getBase:()=>model,getState:()=>state,getDataReport:()=>({coco_train:40000,coco_with_hand_labels:32558,cmu_foot:footSource,h3wb:d3?.manifest,aist:d3?.manifest.aist}),gradients,progress,status,log,draw,download,startTrainingTime,stopTrainingTime,trainingTime,trainingTimeState,restoreTrainingTime,checkpointBoundary,requestBackup,setBusy:value=>{busy=value||baseRunActive;controls();},getSignal:()=>abort?.signal,setAbort:value=>abort=value});
queueMicrotask(()=>videoWorkflow.activate());
$('publish').onclick=async()=>{busy=true;controls();try{const token=$('hf-token').value.trim(),repo=$('hf-repo').value.trim();if(!token)throw Error('Hugging Faceの書込トークンを入力してください');if(!state.updates2||!state.updates3)throw Error('2Dと3Dの実学習が必要です');const url=await publishModel({repo,token,bytes:await model.export(),spec:report(),progress});const a=document.createElement('a');a.href=url;a.textContent=url;a.target='_blank';a.rel='noopener';$('published').replaceChildren(a);progress('Hugging Face公開・取得確認完了',1,1,'ONNXを再取得し、サイズとSHA-256を確認');status('学習済みONNXと出典・ライセンスを公開しました。');}catch(e){problem(e);}finally{$('hf-token').value='';busy=false;controls();}};
