import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {VRMLoaderPlugin,VRMUtils} from '@pixiv/three-vrm';
import {convertCheckpoint} from './checkpoint.mjs';
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
let tracks=[],nextId=1,session=null,ort=null,detector=null,modelBytes=null,temporal=false,running=false,busy=false,videoURL=null,lastTime=-1,vrmBytes=null,vrmGeneration=0,loadGeneration=0;
let lastRender=performance.now();renderer.setAnimationLoop(now=>{const dt=Math.min((now-lastRender)/1000,.1);lastRender=now;controls.update();for(const t of tracks)if(t.vrm)t.vrm.update(dt);renderer.render(scene,camera);});
function fail(e){$('error').textContent=e?.stack||String(e);$('status').textContent='エラー';}
function script(src){return new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=src;s.onload=resolve;s.onerror=()=>reject(Error('実行ライブラリを取得できません: '+src));document.head.append(s);});}
async function dependencies(){
 await script('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.js');ort=globalThis.ort;ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';ort.env.wasm.numThreads=1;
 await script('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');await globalThis.tf.setBackend('webgl');await globalThis.tf.ready();
 await script('https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js');
 $('status').textContent='人物検出モデルを取得中';detector=await globalThis.cocoSsd.load({base:'lite_mobilenet_v2'});$('status').textContent='人物検出準備完了';ready();
}
function ready(){const ok=!!session&&!!detector&&video.readyState>=2;$('run').disabled=!ok;$('single').disabled=!ok;}
const deps=dependencies().catch(e=>{fail(e);throw e;});deps.catch(()=>{});
function stop(){running=false;video.pause();$('run').textContent='推論開始';}
function clearTracks(){for(const t of tracks)disposeTrack(t);tracks=[];$('people').replaceChildren();ctx.clearRect(0,0,overlay.width,overlay.height);}
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
 $('modelStatus').textContent=result.summary+' · '+backend;$('download').disabled=false;$('status').textContent='ONNX推論セッション準備完了';
 }catch(err){fail(err);$('modelStatus').textContent='読み込み失敗';}finally{$('checkpoint').disabled=false;ready();}};
$('download').onclick=()=>{if(!modelBytes)return;const url=URL.createObjectURL(new Blob([modelBytes],{type:'application/octet-stream'})),a=document.createElement('a');a.href=url;a.download='wholebody73.onnx';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('videoFile').onchange=async e=>{const file=e.target.files[0];if(!file)return;stop();while(busy)await new Promise(r=>setTimeout(r,20));clearTracks();if(videoURL)URL.revokeObjectURL(videoURL);videoURL=URL.createObjectURL(file);video.src=videoURL;lastTime=-1;video.load();ready();};
video.onloadeddata=()=>{overlay.width=video.videoWidth;overlay.height=video.videoHeight;$('empty').hidden=true;ready();};video.onerror=()=>fail(Error('この動画をブラウザで再生できません'));video.onseeking=()=>{lastTime=-1;for(const t of tracks)t.history=[];};video.onended=()=>{running=false;$('run').textContent='推論開始';};
// A VRM is parsed locally for every tracked person; no file or texture is sent.
async function attachVRM(t){if(!vrmBytes||t.vrmLoading)return;const gen=vrmGeneration;t.vrmLoading=true;try{
 const loader=new GLTFLoader();loader.register(p=>new VRMLoaderPlugin(p));
 // Reject external URI resources. Embedded GLB images/buffers use blob/data URLs.
 loader.manager.setURLModifier(url=>{if(/^(blob:|data:)/.test(url))return url;throw Error('VRM内の外部リソース参照は読み込みできません');});
 const gltf=await loader.parseAsync(vrmBytes.slice(0),'');const vrm=gltf.userData.vrm;if(!vrm)throw Error('VRM拡張がありません');
 if(gen!==vrmGeneration||!tracks.includes(t)){VRMUtils.deepDispose(vrm.scene);return;}disposeVRM(t);VRMUtils.rotateVRM0(vrm);t.vrm=vrm;scene.add(vrm.scene);$('vrmStatus').textContent='VRM読み込み済み · 人物ごとに主要ボーンを駆動';
 }catch(e){$('vrmStatus').textContent='VRM読み込み失敗: '+e.message;}finally{t.vrmLoading=false;}}
$('vrmFile').onchange=async e=>{if(!e.target.files[0])return;try{vrmBytes=await e.target.files[0].arrayBuffer();vrmGeneration++;for(const t of tracks)disposeVRM(t);$('vrmStatus').textContent='VRMを選択済み · 人物検出後に表示';for(const t of tracks)await attachVRM(t);}catch(e){fail(e);}};
function makeTrack(box){const color=palette[(nextId-1)%palette.length],group=new THREE.Group();const lines=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color,transparent:true,opacity:.9})),points=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({color,size:.035}));group.add(lines,points);scene.add(group);const t={id:nextId++,box,group,lines,points,color,history:[],missed:0,vrm:null};tracks.push(t);attachVRM(t);return t;}
function iou(a,b){const w=Math.max(0,Math.min(a[0]+a[2],b[0]+b[2])-Math.max(a[0],b[0])),h=Math.max(0,Math.min(a[1]+a[3],b[1]+b[3])-Math.max(a[1],b[1])),i=w*h;return i/(a[2]*a[3]+b[2]*b[3]-i||1);}
const frame=document.createElement('canvas'),frameCtx=frame.getContext('2d');
const crop=document.createElement('canvas');crop.width=192;crop.height=256;const cropCtx=crop.getContext('2d',{willReadFrequently:true});
function input(box){let [x,y,w,h]=box;h=Math.max(h,w/.75)*1.25;w=h*.75;const cx=x+box[2]/2,cy=y+box[3]/2;x=cx-w/2;y=cy-h/2;cropCtx.fillStyle='black';cropCtx.fillRect(0,0,192,256);cropCtx.drawImage(frame,x,y,w,h,0,0,192,256);const rgba=cropCtx.getImageData(0,0,192,256).data,n=192*256,data=new Float32Array(n*3);for(let i=0;i<n;i++)for(let c=0;c<3;c++)data[c*n+i]=rgba[i*4+c]/255;return {data,rect:[x,y,w,h]};}
function poseVRM(t,positions,root){if(!t.vrm)return;const h=t.vrm.humanoid;if(!h)return;h.resetNormalizedPose();t.vrm.scene.position.copy(root);
 const v=i=>new THREE.Vector3(...positions.slice(i*3,i*3+3));
 // Solve world-space bone directions, then bring them into each parent's frame.
 const chains=[['leftUpperArm','leftLowerArm',0,2],['rightUpperArm','rightLowerArm',1,3],['leftLowerArm','leftHand',2,4],['rightLowerArm','rightHand',3,5],['leftUpperLeg','leftLowerLeg',6,8],['rightUpperLeg','rightLowerLeg',7,9],['leftLowerLeg','leftFoot',8,10],['rightLowerLeg','rightFoot',9,11]];
 const hips=h.getNormalizedBoneNode('hips'),spine=h.getNormalizedBoneNode('spine');
 if(hips&&spine){t.vrm.scene.updateMatrixWorld(true);const rest=spine.getWorldPosition(new THREE.Vector3()).sub(hips.getWorldPosition(new THREE.Vector3())).normalize(),desired=v(0).add(v(1)).multiplyScalar(.5).sub(v(6).add(v(7)).multiplyScalar(.5)).normalize();if(desired.lengthSq()>.1){const delta=new THREE.Quaternion().setFromUnitVectors(rest,desired),parent=hips.parent.getWorldQuaternion(new THREE.Quaternion()),world=hips.getWorldQuaternion(new THREE.Quaternion());hips.quaternion.copy(parent.invert().multiply(delta.multiply(world)));}}
 for(const [a,b,i,j]of chains){const bone=h.getNormalizedBoneNode(a),child=h.getNormalizedBoneNode(b);if(!bone||!child)continue;t.vrm.scene.updateMatrixWorld(true);const rest=child.getWorldPosition(new THREE.Vector3()).sub(bone.getWorldPosition(new THREE.Vector3())).normalize(),desired=v(j).sub(v(i));if(desired.lengthSq()<1e-8)continue;desired.normalize();const delta=new THREE.Quaternion().setFromUnitVectors(rest,desired),parent=bone.parent.getWorldQuaternion(new THREE.Quaternion()),world=bone.getWorldQuaternion(new THREE.Quaternion());bone.quaternion.copy(parent.invert().multiply(delta.multiply(world)));}
 if(hips){t.vrm.scene.updateMatrixWorld(true);const hp=hips.getWorldPosition(new THREE.Vector3());t.vrm.scene.position.add(root.clone().sub(hp));}}
async function infer(){if(busy||!session||!detector||video.readyState<2)return;busy=true;const now=performance.now();try{
 $('error').textContent='';const frameTime=video.currentTime;frame.width=video.videoWidth;frame.height=video.videoHeight;frameCtx.drawImage(video,0,0);const boxes=(await detector.detect(frame,20,+$('score').value)).filter(d=>d.class==='person').sort((a,b)=>b.score-a.score).slice(0,Math.min(20,Math.max(1,+$('maxPeople').value||6)));
 ctx.clearRect(0,0,overlay.width,overlay.height);const available=new Set(tracks),active=[];
 for(const d of boxes){let best=null,score=.1;for(const t of available){const s=iou(t.box,d.bbox);if(s>score){best=t;score=s;}}const t=best||makeTrack(d.bbox);available.delete(t);t.box=d.bbox;t.missed=0;active.push({t,d});}
 for(const t of available){t.missed++;t.group.visible=false;if(t.vrm)t.vrm.scene.visible=false;if(t.missed>12){disposeTrack(t);tracks=tracks.filter(x=>x!==t);}}
 $('people').replaceChildren();
 for(const {t,d}of active){const {data,rect}=input(d.bbox),n=data.length;t.history.push(data);if(t.history.length>3)t.history.shift();while(t.history.length<3)t.history.unshift(data);
 const image=temporal?new Float32Array(n*3):data;if(temporal)t.history.forEach((f,i)=>image.set(f,n*i));
 const feeds={[temporal?'frames':'image']:new ort.Tensor('float32',image,temporal?[1,3,3,256,192]:[1,3,256,192]),tracking_rotation:new ort.Tensor('float32',Float32Array.of(1,0,0,0,1,0,0,0,1),[3,3]),tracking_origin:new ort.Tensor('float32',new Float32Array(3),[1,1,3])};
 let out;try{out=await session.run(feeds);const xy=Array.from(out.keypoints2d.data),conf=Array.from(out.confidence2d.data),raw=Array.from(out.vrc_relative.data);if([...xy,...conf,...raw].some(v=>!Number.isFinite(v)))throw Error('ONNX出力に非有限値があります');
 const focal=video.videoWidth/(2*Math.tan((+$('fov').value||60)*Math.PI/360)),height=+$('height').value||1.7,z=focal*height/Math.max(1,d.bbox[3]),hipx=rect[0]+(xy[12]+xy[14])*.5*rect[2],hipy=rect[1]+(xy[13]+xy[15])*.5*rect[3];
 const root=new THREE.Vector3((hipx-video.videoWidth/2)*z/focal,height*.55-(hipy-(d.bbox[1]+d.bbox[3]*.55))*z/focal,-z),positions=[];for(let j=0;j<73;j++)positions.push(raw[j*3],raw[j*3+1],-raw[j*3+2]);
 t.group.position.copy(root);t.group.visible=true;const threshold=+$('jointScore').value,segments=[],dots=[];for(let j=0;j<73;j++)if(conf[j]>=threshold)dots.push(...positions.slice(j*3,j*3+3));for(const [a,b]of edges)if(conf[a]>=threshold&&conf[b]>=threshold)segments.push(...positions.slice(a*3,a*3+3),...positions.slice(b*3,b*3+3));
 t.lines.geometry.dispose();t.points.geometry.dispose();t.lines.geometry=new THREE.BufferGeometry();t.lines.geometry.setAttribute('position',new THREE.Float32BufferAttribute(segments,3));t.points.geometry=new THREE.BufferGeometry();t.points.geometry.setAttribute('position',new THREE.Float32BufferAttribute(dots,3));if(t.vrm){t.vrm.scene.visible=true;poseVRM(t,positions,root);}
 ctx.strokeStyle='#'+t.color.toString(16).padStart(6,'0');ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=Math.max(2,video.videoWidth/500);ctx.strokeRect(...d.bbox);ctx.font=`${Math.max(16,video.videoWidth/55)}px system-ui`;ctx.fillText('ID '+t.id,d.bbox[0],Math.max(20,d.bbox[1]-6));const point=j=>[rect[0]+xy[j*2]*rect[2],rect[1]+xy[j*2+1]*rect[3]];for(const [a,b]of edges)if(conf[a]>=threshold&&conf[b]>=threshold){ctx.beginPath();ctx.moveTo(...point(a));ctx.lineTo(...point(b));ctx.stroke();}
 const card=document.createElement('div');card.className='person';card.style.borderColor=ctx.strokeStyle;const zs=raw.filter((_,i)=>i%3===2);card.textContent=`ID ${t.id} · 配置距離 ≈${z.toFixed(2)}m · 腰基準Z ${Math.min(...zs).toFixed(2)}〜${Math.max(...zs).toFixed(2)}m`; $('people').append(card);
 }finally{Object.values(feeds).forEach(x=>x.dispose());if(out)Object.values(out).forEach(x=>x.dispose());}}
 $('status').textContent=`${active.length}人 · ${(performance.now()-now).toFixed(0)} ms / frame`;lastTime=frameTime;
 }catch(e){stop();fail(e);}finally{busy=false;}}
async function loop(){if(!running)return;if(video.ended){stop();return;}if(video.currentTime!==lastTime)await infer();if(running)requestAnimationFrame(loop);}
$('run').onclick=async()=>{if(running){stop();return;}try{await video.play();running=true;$('run').textContent='推論停止';loop();}catch(e){fail(e);}};
$('single').onclick=async()=>{stop();await infer();};
