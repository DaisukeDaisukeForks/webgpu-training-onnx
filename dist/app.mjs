import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {VRMLoaderPlugin,VRMUtils} from '@pixiv/three-vrm';
import {convertCheckpoint} from './checkpoint.mjs';
import {frameTransform,frameInput} from './frame-input.mjs';
const $=id=>document.getElementById(id),video=$('video'),overlay=$('overlay'),ctx=overlay.getContext('2d');
const palette=[0x91efc5,0x72b5ff,0xffb16e,0xd59fff,0xff8caa,0xffe282];
const edges=[[0,1],[0,2],[2,4],[1,3],[3,5],[0,6],[1,7],[6,7],[6,8],[8,10],[7,9],[9,11],[10,52],[10,53],[10,54],[11,55],[11,56],[11,57],[58,60],[59,60],[61,62],[65,66],[69,70],[71,72]];
for(const [w,start]of [[4,12],[5,32]])for(let f=0;f<5;f++){edges.push([w,start+f*4]);for(let j=0;j<3;j++)edges.push([start+f*4+j,start+f*4+j+1]);}
const scene=new THREE.Scene();scene.background=new THREE.Color('#0a1421');
const camera=new THREE.PerspectiveCamera(48,1,.05,100);camera.position.set(5,4,5);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));$('scene').append(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,1,-3);controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xdceeff,0x586c57,2.5));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(3,7,3);scene.add(light);
scene.add(new THREE.GridHelper(30,30,0x52718d,0x23384d));scene.add(new THREE.AxesHelper(.7));
const camIcon=new THREE.Mesh(new THREE.BoxGeometry(.25,.16,.12),new THREE.MeshBasicMaterial({color:0xffffff}));scene.add(camIcon);scene.add(new THREE.ArrowHelper(new THREE.Vector3(0,0,-1),new THREE.Vector3(),.9,0xffffff));
new ResizeObserver(()=>{const w=$('scene').clientWidth,h=$('scene').clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}).observe($('scene'));
let tracks=[],nextId=1,session=null,ort=null,modelBytes=null,temporal=false,running=false,busy=false,videoURL=null,lastTime=-1,vrmBytes=null,vrmGeneration=0,loadGeneration=0;
let frameGeneration=0,loopGeneration=0;
let lastRender=performance.now();renderer.setAnimationLoop(now=>{const dt=Math.min((now-lastRender)/1000,.1);lastRender=now;controls.update();for(const t of tracks)if(t.vrm)t.vrm.update(dt);renderer.render(scene,camera);});
function fail(e){$('error').textContent=e?.stack||String(e);$('status').textContent='エラー';}
function script(src){return new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=src;s.onload=resolve;s.onerror=()=>reject(Error('実行ライブラリを取得できません: '+src));document.head.append(s);});}
async function dependencies(){
 await script('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.js');ort=globalThis.ort;ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';ort.env.wasm.numThreads=1;
 $('status').textContent='ファイルを選択してください';ready();
}
function ready(){const ok=!!session&&video.readyState>=2;$('run').disabled=!ok;$('single').disabled=!ok;}
const deps=dependencies().catch(e=>{fail(e);throw e;});deps.catch(()=>{});
function stop(){running=false;loopGeneration++;video.pause();$('run').textContent='推論開始';}
function clearTracks(){frameGeneration++;for(const t of tracks)disposeTrack(t);tracks=[];$('people').replaceChildren();ctx.clearRect(0,0,overlay.width,overlay.height);$('modelInput').getContext('2d').clearRect(0,0,192,256);}
function disposeVRM(t){if(t.vrm){scene.remove(t.vrm.scene);VRMUtils.deepDispose(t.vrm.scene);t.vrm=null;}}
function disposeTrack(t){scene.remove(t.group);t.lines.geometry.dispose();t.points.geometry.dispose();t.lines.material.dispose();t.points.material.dispose();disposeVRM(t);}
$('checkpoint').onchange=async e=>{const file=e.target.files[0];if(!file)return;const generation=++loadGeneration;stop();$('download').disabled=true;$('checkpoint').disabled=true;try{
 $('error').textContent='';$('modelStatus').textContent='JSONを検証し、ONNXを生成中…';
 while(busy)await new Promise(r=>setTimeout(r,20));
 const result=await convertCheckpoint(file);await deps;
 if(generation!==loadGeneration)return;
 let replacement,backend='WASM';
 if(navigator.gpu){try{replacement=await ort.InferenceSession.create(result.bytes,{executionProviders:['webgpu','wasm']});backend='WebGPU / WASM';}catch(e){console.warn('WebGPU session unavailable',e);}}
 if(!replacement)replacement=await ort.InferenceSession.create(result.bytes,{executionProviders:['wasm']});
 if(session)await session.release();session=replacement;modelBytes=result.bytes;temporal=result.temporal;clearTracks();
 $('modelStatus').textContent='読み込み完了';$('download').disabled=false;$('status').textContent='準備完了';
 }catch(err){fail(err);$('modelStatus').textContent='読み込み失敗';}finally{$('checkpoint').disabled=false;ready();}};
$('download').onclick=()=>{if(!modelBytes)return;const url=URL.createObjectURL(new Blob([modelBytes],{type:'application/octet-stream'})),a=document.createElement('a');a.href=url;a.download='wholebody73.onnx';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('videoFile').onchange=async e=>{const file=e.target.files[0];if(!file)return;stop();while(busy)await new Promise(r=>setTimeout(r,20));clearTracks();if(videoURL)URL.revokeObjectURL(videoURL);videoURL=URL.createObjectURL(file);video.src=videoURL;lastTime=-1;video.load();ready();};
video.onloadeddata=()=>{overlay.width=video.videoWidth;overlay.height=video.videoHeight;$('empty').hidden=true;ready();};video.onerror=()=>fail(Error('この動画をブラウザで再生できません'));video.onseeking=()=>{lastTime=-1;clearTracks();};video.onended=()=>{running=false;loopGeneration++;$('run').textContent='推論開始';};
// A VRM is parsed locally for every tracked person; no file or texture is sent.
async function attachVRM(t){if(!vrmBytes||t.vrmLoading)return;const gen=vrmGeneration;t.vrmLoading=true;try{
 const loader=new GLTFLoader();loader.register(p=>new VRMLoaderPlugin(p));
 // Reject external URI resources. Embedded GLB images/buffers use blob/data URLs.
 loader.manager.setURLModifier(url=>{if(/^(blob:|data:)/.test(url))return url;throw Error('VRM内の外部リソース参照は読み込みできません');});
 const gltf=await loader.parseAsync(vrmBytes.slice(0),'');const vrm=gltf.userData.vrm;if(!vrm)throw Error('VRM拡張がありません');
 if(gen!==vrmGeneration||!tracks.includes(t)){VRMUtils.deepDispose(vrm.scene);return;}disposeVRM(t);VRMUtils.rotateVRM0(vrm);t.vrm=vrm;scene.add(vrm.scene);vrm.scene.visible=t.group.visible;if(t.lastPose)poseVRM(t,t.lastPose.positions,t.lastPose.root);$('vrmStatus').textContent='VRM読み込み済み · 主要ボーンを駆動';
 }catch(e){$('vrmStatus').textContent='VRM読み込み失敗: '+e.message;}finally{t.vrmLoading=false;}}
$('vrmFile').onchange=async e=>{if(!e.target.files[0])return;try{vrmBytes=await e.target.files[0].arrayBuffer();vrmGeneration++;for(const t of tracks)disposeVRM(t);$('vrmStatus').textContent='VRMを選択済み · 推論後に表示';for(const t of tracks)await attachVRM(t);}catch(e){fail(e);}};
function makeTrack(box){const color=palette[(nextId-1)%palette.length],group=new THREE.Group();const lines=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color,transparent:true,opacity:.9})),points=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({color,size:.035,transparent:true}));lines.geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(edges.length*6),3).setUsage(THREE.DynamicDrawUsage));points.geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(73*3),3).setUsage(THREE.DynamicDrawUsage));lines.geometry.setDrawRange(0,0);points.geometry.setDrawRange(0,0);lines.frustumCulled=false;points.frustumCulled=false;group.visible=false;group.add(lines,points);scene.add(group);const t={id:nextId++,box,group,lines,points,color,history:[],missed:0,lastSeen:-Infinity,vrm:null};tracks.push(t);attachVRM(t);return t;}
const frame=document.createElement('canvas'),frameCtx=frame.getContext('2d');
const modelCanvas=document.createElement('canvas');modelCanvas.width=192;modelCanvas.height=256;const modelCtx=modelCanvas.getContext('2d',{willReadFrequently:true});
function input(){const transform=frameTransform(frame.width,frame.height);modelCtx.fillStyle='black';modelCtx.fillRect(0,0,192,256);modelCtx.drawImage(frame,transform.padX,transform.padY,transform.width,transform.height);return {data:frameInput(modelCtx.getImageData(0,0,192,256).data),rect:transform.rect};}
function poseVRM(t,positions,root){if(!t.vrm)return;const h=t.vrm.humanoid;if(!h)return;h.resetNormalizedPose();t.vrm.scene.position.copy(root);
 const v=i=>new THREE.Vector3(...positions.slice(i*3,i*3+3));
 // Solve world-space bone directions, then bring them into each parent's frame.
 const chains=[['leftUpperArm','leftLowerArm',0,2],['rightUpperArm','rightLowerArm',1,3],['leftLowerArm','leftHand',2,4],['rightLowerArm','rightHand',3,5],['leftUpperLeg','leftLowerLeg',6,8],['rightUpperLeg','rightLowerLeg',7,9],['leftLowerLeg','leftFoot',8,10],['rightLowerLeg','rightFoot',9,11]];
 const hips=h.getNormalizedBoneNode('hips'),spine=h.getNormalizedBoneNode('spine');
 if(hips&&spine){t.vrm.scene.updateMatrixWorld(true);const rest=spine.getWorldPosition(new THREE.Vector3()).sub(hips.getWorldPosition(new THREE.Vector3())).normalize(),desired=v(0).add(v(1)).multiplyScalar(.5).sub(v(6).add(v(7)).multiplyScalar(.5)).normalize();if(desired.lengthSq()>.1){const delta=new THREE.Quaternion().setFromUnitVectors(rest,desired),parent=hips.parent.getWorldQuaternion(new THREE.Quaternion()),world=hips.getWorldQuaternion(new THREE.Quaternion());hips.quaternion.copy(parent.invert().multiply(delta.multiply(world)));}}
 for(const [a,b,i,j]of chains){const bone=h.getNormalizedBoneNode(a),child=h.getNormalizedBoneNode(b);if(!bone||!child)continue;t.vrm.scene.updateMatrixWorld(true);const rest=child.getWorldPosition(new THREE.Vector3()).sub(bone.getWorldPosition(new THREE.Vector3())).normalize(),desired=v(j).sub(v(i));if(desired.lengthSq()<1e-8)continue;desired.normalize();const delta=new THREE.Quaternion().setFromUnitVectors(rest,desired),parent=bone.parent.getWorldQuaternion(new THREE.Quaternion()),world=bone.getWorldQuaternion(new THREE.Quaternion());bone.quaternion.copy(parent.invert().multiply(delta.multiply(world)));}
 if(hips){t.vrm.scene.updateMatrixWorld(true);const hp=hips.getWorldPosition(new THREE.Vector3());t.vrm.scene.position.add(root.clone().sub(hp));}}
const overlayBuffer=document.createElement('canvas'),bufferCtx=overlayBuffer.getContext('2d');
async function predictFrame(t){
 const {data,rect}=input(),n=data.length,history=t?t.history.slice(-2):[];
 history.push(data);while(history.length<3)history.unshift(data);
 const image=temporal?new Float32Array(n*3):data;if(temporal)history.forEach((f,i)=>image.set(f,n*i));
 const feeds={[temporal?'frames':'image']:new ort.Tensor('float32',image,temporal?[1,3,3,256,192]:[1,3,256,192]),tracking_rotation:new ort.Tensor('float32',Float32Array.of(1,0,0,0,1,0,0,0,1),[3,3]),tracking_origin:new ort.Tensor('float32',new Float32Array(3),[1,1,3])};
 let out;
 try{
  out=await session.run(feeds);
  const xy=Array.from(out.keypoints2d.data),conf=Array.from(out.confidence2d.data),raw=Array.from(out.vrc_relative.data);
  if([...xy,...conf,...raw].some(v=>!Number.isFinite(v)))throw Error('ONNX出力に非有限値があります');
  const focal=frame.width/(2*Math.tan((+$('fov').value||60)*Math.PI/360)),z=Math.max(.1,+$('distance').value||3);
  const hipx=rect[0]+(xy[12]+xy[14])*.5*rect[2];
  const root=new THREE.Vector3((hipx-frame.width/2)*z/focal,1,-z),positions=[];
  for(let j=0;j<73;j++)positions.push(raw[j*3],raw[j*3+1],-raw[j*3+2]);
  return {t,history,pose:{xy,conf,raw,rect,root,positions,z}};
 }finally{Object.values(feeds).forEach(x=>x.dispose());if(out)Object.values(out).forEach(x=>x.dispose());}
}
function updateGeometry(t){
 const {positions,root,conf}=t.lastPose,threshold=+$('jointScore').value;
 const lines=t.lines.geometry.getAttribute('position'),points=t.points.geometry.getAttribute('position');let l=0,p=0;
 for(let j=0;j<73;j++)if(conf[j]>=threshold){points.array.set(positions.slice(j*3,j*3+3),p);p+=3;}
 for(const [a,b]of edges)if(conf[a]>=threshold&&conf[b]>=threshold){lines.array.set(positions.slice(a*3,a*3+3),l);lines.array.set(positions.slice(b*3,b*3+3),l+3);l+=6;}
 lines.needsUpdate=true;points.needsUpdate=true;t.lines.geometry.setDrawRange(0,l/3);t.points.geometry.setDrawRange(0,p/3);
 t.group.position.copy(root);poseVRM(t,positions,root);
}
function drawPerson(t,cards){
 const {xy,conf,rect,z,raw}=t.lastPose,c=bufferCtx,threshold=+$('jointScore').value;
 c.save();c.strokeStyle='#'+t.color.toString(16).padStart(6,'0');c.fillStyle=c.strokeStyle;c.lineWidth=Math.max(2,video.videoWidth/500);
 const point=j=>[rect[0]+xy[j*2]*rect[2],rect[1]+xy[j*2+1]*rect[3]];
 for(const [a,b]of edges)if(conf[a]>=threshold&&conf[b]>=threshold){c.beginPath();c.moveTo(...point(a));c.lineTo(...point(b));c.stroke();}
 const card=document.createElement('div');card.className='person';card.style.borderColor=c.strokeStyle;
 const zs=raw.filter((_,i)=>i%3===2);card.textContent=`腰からの奥行き ${Math.min(...zs).toFixed(2)}〜${Math.max(...zs).toFixed(2)}m`;cards.append(card);c.restore();
}
function commitFrame(r,started,frameTime){
 const now=performance.now(),cards=document.createDocumentFragment();
 if(overlayBuffer.width!==overlay.width||overlayBuffer.height!==overlay.height){overlayBuffer.width=overlay.width;overlayBuffer.height=overlay.height;}
 bufferCtx.clearRect(0,0,overlayBuffer.width,overlayBuffer.height);
 const t=r.t||makeTrack();t.history=r.history;t.lastPose=r.pose;updateGeometry(t);t.group.visible=true;if(t.vrm)t.vrm.scene.visible=true;drawPerson(t,cards);
 // There is no await between clearing the visible canvas and painting the
 // complete frame. Browsers never present an empty or partially rebuilt pose.
 ctx.clearRect(0,0,overlay.width,overlay.height);ctx.drawImage(overlayBuffer,0,0);$('people').replaceChildren(cards);
 $('modelInput').getContext('2d').drawImage(modelCanvas,0,0);
 $('status').textContent=`${(now-started).toFixed(0)} ms / frame`;lastTime=frameTime;
}
async function infer(){
 if(busy||!session||video.readyState<2||video.seeking)return;
 busy=true;const now=performance.now(),generation=frameGeneration;
 try{
  $('error').textContent='';const frameTime=video.currentTime;
  frame.width=video.videoWidth;frame.height=video.videoHeight;frameCtx.drawImage(video,0,0);
  const result=await predictFrame(tracks[0]);
  if(generation===frameGeneration)commitFrame(result,now,frameTime);
 }catch(e){stop();fail(e);}finally{busy=false;}
}
async function loop(generation){if(!running||generation!==loopGeneration)return;if(video.ended){stop();return;}if(video.currentTime!==lastTime)await infer();if(running&&generation===loopGeneration)requestAnimationFrame(()=>loop(generation));}
$('run').onclick=async()=>{if(running){stop();return;}const generation=++loopGeneration;try{await video.play();if(generation!==loopGeneration)return;running=true;$('run').textContent='推論停止';loop(generation);}catch(e){fail(e);}};
$('single').onclick=async()=>{stop();while(busy)await new Promise(r=>setTimeout(r,20));await infer();};
