import {sourceArchive} from './source-cache.mjs?revision=10';
import {Zip} from './zip.mjs?revision=12';
import {readNumpyPickle} from './numpy-pickle.mjs?revision=12';
export function rotationMatrix(v){
 const theta=Math.hypot(...v);if(theta<1e-12)return[1,0,0,0,1,0,0,0,1];const[x,y,z]=v.map(a=>a/theta),c=Math.cos(theta),s=Math.sin(theta),d=1-c;
 return[c+x*x*d,x*y*d-z*s,x*z*d+y*s,y*x*d+z*s,c+y*y*d,y*z*d-x*s,z*x*d-y*s,z*y*d+x*s,c+z*z*d];
}
export function projectFrame(world,offset,camera,scale){
 const xyz=new Float64Array(51),uv=new Float64Array(34),r=camera.R||rotationMatrix(camera.rotation),t=camera.translation,K=camera.matrix,[k1=0,k2=0,p1=0,p2=0,k3=0]=camera.distortions;let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
 for(let j=0;j<17;j++){
  const p=offset+j*3,x=world[p],y=world[p+1],z=world[p+2],cx=r[0]*x+r[1]*y+r[2]*z+t[0],cy=r[3]*x+r[4]*y+r[5]*z+t[1],cz=r[6]*x+r[7]*y+r[8]*z+t[2];
  if(!Number.isFinite(cx+cy+cz)||cz<=1e-6)return null;xyz.set([cx,cy,cz],j*3);
  const nx=cx/cz,ny=cy/cz,q=nx*nx+ny*ny,radial=1+k1*q+k2*q*q+k3*q*q*q,dx=nx*radial+2*p1*nx*ny+p2*(q+2*nx*nx),dy=ny*radial+p1*(q+2*ny*ny)+2*p2*nx*ny;
  const u=K[0][0]*dx+K[0][1]*dy+K[0][2],v=K[1][0]*dx+K[1][1]*dy+K[1][2];if(!Number.isFinite(u+v))return null;uv.set([u,v],j*2);minX=Math.min(minX,u);maxX=Math.max(maxX,u);minY=Math.min(minY,v);maxY=Math.max(maxY,v);
 }
 const height=Math.max(maxY-minY,(maxX-minX)/.75)*1.25;if(!(height>1e-6))return null;
 const width=height*.75,ox=(minX+maxX-width)/2,oy=(minY+maxY-height)/2,hip=[0,1,2].map(k=>(xyz[11*3+k]+xyz[12*3+k])/2),row=new Float32Array(511);let onScreen=0;
 for(let j=0;j<12;j++){const source=j+5,u=uv[source*2],v=uv[source*2+1];if(u>=0&&v>=0&&u<camera.size[0]&&v<camera.size[1])onScreen++;
  row.set([(u-ox)/width,(v-oy)/height,1,(xyz[source*3]-hip[0])/scale,-(xyz[source*3+1]-hip[1])/scale,(xyz[source*3+2]-hip[2])/scale,1],j*7);
 }return onScreen>=6?row:null;
}
export class AistData{
 constructor(meta,progress=()=>{}){this.meta=meta;this.progress=progress;this.ranges={train:[],val:[]};this.counts={train:0,val:0};this.summary={dataset:'AIST++',source:meta.source,license:meta.license,coordinates:meta.coordinates,train_clips:0,val_clips:0,train_frames:0,val_frames:0,excluded_missing_scale:{train:0,val:0},excluded_views:0,sha256:Object.fromEntries(Object.entries(meta.sources).map(([k,v])=>[k,v.sha256]))};}
 async load(signal){
  if(this.loaded)return;Object.assign(this,new AistData(this.meta,this.progress));const get=async name=>Zip.open(await sourceArchive(name,this.meta.sources[name],signal,(n,total,detail)=>this.progress('AIST++の事前取得',n,total,detail)));
  const motions=await get('aist-motions'),poses=await get('aist-keypoints3d'),ignored=new Set(this.meta.ignore),selected=[];
  for(const split of ['train','val'])for(const name of this.meta.splits[split])if(!ignored.has(name)){
   if(!motions.entries.has(`motions/${name}.pkl`)){this.summary.excluded_missing_scale[split]++;continue;}
   selected.push({split,name});
  }
  const seen=new Set();let done=0;
  for(const{split,name}of selected){
   signal?.throwIfAborted();if(seen.has(name))throw Error('AIST++の学習・検証が重複しています');seen.add(name);
   const motion=readNumpyPickle(await motions.bytes(`motions/${name}.pkl`)),pose=readNumpyPickle(await poses.bytes(`keypoints3d/${name}.pkl`)),a=pose.keypoints3d_optim,scale=motion.smpl_scaling?.values?.[0];
   if(!a||a.shape.length!==3||a.shape[1]!==17||a.shape[2]!==3||motion.smpl_poses?.shape[0]!==a.shape[0]||!Number.isFinite(scale)||scale<=0)throw Error(`AIST++の形状・フレーム・尺度が不正です: ${name}`);
   const world=Float32Array.from(a.values),setting=this.meta.cameras[this.meta.mapping[name]];if(!setting)throw Error('AIST++カメラがありません');
   let acceptedFrames=new Set();
   for(const original of setting){
    if((name==='gBR_sBM_cAll_d04_mBR0_ch01'&&original.name==='c05')||(name==='gJB_sBM_cAll_d07_mJB3_ch05'&&original.name==='c07')){this.summary.excluded_views++;continue;}
    const camera={...original,R:rotationMatrix(original.rotation)},frames=[];
    for(let i=0;i<a.shape[0];i++)if(projectFrame(world,i*51,camera,scale)){frames.push(i);acceptedFrames.add(i);}
    if(!frames.length)continue;const start=this.counts[split];this.counts[split]+=frames.length;
    this.ranges[split].push({name,camera,scale,world,frames:Uint32Array.from(frames),start,end:this.counts[split]});
   }
   if(acceptedFrames.size){this.summary[split+'_clips']++;this.summary[split+'_frames']+=acceptedFrames.size;}
   this.progress('AIST++の座標・分割を照合',++done,selected.length,`${this.counts.train.toLocaleString()}学習例 / ${this.counts.val.toLocaleString()}検証例`);
   await new Promise(resolve=>setTimeout(resolve,0));
  }
  if(!this.counts.train||!this.counts.val)throw Error('AIST++に有効な学習・検証教師がありません');Object.assign(this.summary,{train_count:this.counts.train,val_count:this.counts.val});this.loaded=true;
 }
 row(split,index){const rows=this.ranges[split];let lo=0,hi=rows.length;while(lo<hi){const mid=(lo+hi)>>1;if(rows[mid].end<=index)lo=mid+1;else hi=mid;}const r=rows[lo];if(!r||index<r.start)throw Error('AIST++教師の範囲外');const out=projectFrame(r.world,r.frames[index-r.start]*51,r.camera,r.scale);if(!out)throw Error('AIST++の検証済み座標が無効です');return out;}
 references(){return Object.fromEntries(['train','val'].map(split=>[split,this.ranges[split].map(r=>({sequence:r.name,camera:r.camera.name,frames:Array.from(r.frames),scale:r.scale}))]));}
}
