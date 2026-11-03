import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {stableStep} from '../public/gradients.mjs';
import {WB73} from '../public/model.mjs';

const require=createRequire(import.meta.url);
if(!process.env.TFJS_BUNDLE)throw Error('Set TFJS_BUNDLE to the existing TensorFlow.js 4.22.0 CommonJS bundle');
const tf=require(process.env.TFJS_BUNDLE);
await tf.setBackend('cpu');await tf.ready();
assert.equal(tf.version.tfjs,'4.22.0');
const tests=[],tolerance=3e-6;
const close=(a,b,label)=>{
 assert.equal(a.length,b.length,label+' shape');let max=0;
 for(let i=0;i<a.length;i++){assert.ok(Number.isFinite(a[i])&&Number.isFinite(b[i]),label+' finite');const error=Math.abs(a[i]-b[i]);max=Math.max(max,error);assert.ok(error<=tolerance*Math.max(1,Math.abs(b[i])),`${label}[${i}]: ${a[i]} != ${b[i]}`);}
 return max;
};
const variable=values=>tf.tidy(()=>tf.variable(tf.tensor1d(values)));
async function snapshot(v,opt){
 const weights=await opt.getWeights(),adam=weights.map(w=>({name:w.name,shape:w.tensor.shape,data:Array.from(w.tensor.dataSync())}));
 weights.filter(w=>w.name==='iter').forEach(w=>w.tensor.dispose());
 return{value:Array.from(v.dataSync()),adam};
}
async function counted(fn){
 const original=tf.Tensor.prototype.data;let reads=0;
 tf.Tensor.prototype.data=function(...args){reads++;return original.apply(this,args);};
 try{return{value:await fn(),reads};}finally{tf.Tensor.prototype.data=original;}
}
// Independent host-side clipping reference, using double-precision hypot.
async function reference(fn,vars,opt){
 const result=tf.variableGrads(fn,vars),entries=Object.entries(result.grads),chunks=entries.map(([,g])=>g.dataSync());
 let norm=0;for(const chunk of chunks)for(const value of chunk)norm=Math.hypot(norm,value);
 const factor=norm>1?1/norm:1,loss=result.value.dataSync()[0];
 tf.tidy(()=>opt.applyGradients(Object.fromEntries(entries.map(([name,g])=>[name,g.mul(factor)]))));
 result.value.dispose();entries.forEach(([,g])=>g.dispose());return loss;
}

{
 const a=variable([2,-3,.25]),b=variable([2,-3,.25]),oa=tf.train.adam(.001),ob=tf.train.adam(.001);
 let max=0;
 try{
  for(let i=0;i<20;i++){
   const loss=v=>v.sub(tf.tensor1d([.1,.2,-.3])).square().sum();
   const actual=await counted(()=>stableStep(tf,()=>loss(a),[a],oa));
   assert.equal(actual.reads,2);const expected=await reference(()=>loss(b),[b],ob);
   close([actual.value],[expected],'loss');max=Math.max(max,close(a.dataSync(),b.dataSync(),'20-step weights'));
  }
  const aa=await snapshot(a,oa),bb=await snapshot(b,ob);
  aa.adam.forEach((w,i)=>close(w.data,bb.adam[i].data,'Adam moments'));
  tests.push({name:'20 Adam updates vs independent clipping reference',max_abs_weight_error:max,async_reads_per_update:2});
 }finally{a.dispose();b.dispose();oa.dispose();ob.dispose();}
}
for(const [name,values,loss]of [
 ['zero gradients',[0,0],v=>v.square().sum()],
 ['large finite gradients',[0,0],v=>v.mul(1e30).sum()]
]){
 const v=variable(values),opt=tf.train.adam(.001);
 try{const result=await counted(()=>stableStep(tf,()=>loss(v),[v],opt));assert.equal(result.reads,2);assert.ok([...v.dataSync()].every(Number.isFinite));tests.push({name,async_reads:result.reads});}
 finally{v.dispose();opt.dispose();}
}
{
 const v=variable([1,2]),opt=tf.train.adam(.001);
 try{
  const before=await snapshot(v,opt),tensors=tf.memory().numTensors;
  for(let i=0;i<5;i++){const result=await counted(()=>stableStep(tf,()=>v.square().sum(),[v],opt,false));assert.equal(result.reads,1);}
  assert.deepEqual(await snapshot(v,opt),before);assert.equal(tf.memory().numTensors,tensors);
  const badGradient=tf.customGrad(x=>({value:tf.scalar(0),gradFunc:()=>tf.fill(x.shape,NaN)}));
  await assert.rejects(()=>stableStep(tf,()=>badGradient(v),[v],opt),/非有限の勾配/);
  await assert.rejects(()=>stableStep(tf,()=>v.mul(Infinity).sum(),[v],opt),/損失が非有限値/);
  assert.deepEqual(await snapshot(v,opt),before);assert.equal(tf.memory().numTensors,tensors);
  tests.push({name:'dry run, invalid loss and invalid gradients preserve weights/Adam; no tensor leak',async_reads_dry_run:1});
 }finally{v.dispose();opt.dispose();}
}
{
 const v=variable([1,2]),opt=tf.train.adam(.001);
 try{
  await stableStep(tf,()=>v.square().sum(),[v],opt);
  const before=await snapshot(v,opt),tensors=tf.memory().numTensors,apply=opt.applyGradients;
  opt.applyGradients=function(grads){apply.call(this,grads);tf.tidy(()=>v.assign(tf.fill(v.shape,Infinity)));};
  for(let i=0;i<3;i++){
   await assert.rejects(()=>stableStep(tf,()=>v.square().sum(),[v],opt),/更新後の重み/);
   assert.deepEqual(await snapshot(v,opt),before);assert.equal(tf.memory().numTensors,tensors);
  }
  opt.applyGradients=apply;
  await stableStep(tf,()=>v.square().sum(),[v],opt);
  assert.ok([...v.dataSync()].every(Number.isFinite));
  tests.push({name:'three failed updates roll back weights and Adam without leaking; next update succeeds'});
 }finally{v.dispose();opt.dispose();}
}
{
 const a=new WB73(tf,20261103),b=new WB73(tf,20261103),oa=tf.train.adam(.001),ob=tf.train.adam(.001);
 const xy=tf.randomUniform([2,73,2],.2,.8,'float32',42),mask=tf.ones([2,73]),target=tf.zeros([2,73,3]);
 try{
  const loss=model=>model.loss3d(xy,mask,target,mask),actual=await counted(()=>stableStep(tf,()=>loss(a),a.liftVars,oa));
  const expected=await reference(()=>loss(b),b.liftVars,ob);close([actual.value],[expected],'WholeBody3D loss');
  let max=0;for(let i=0;i<a.liftVars.length;i++)max=Math.max(max,close(a.liftVars[i].dataSync(),b.liftVars[i].dataSync(),'WholeBody3D weights'));
  assert.equal(actual.reads,2);tests.push({name:'actual WholeBody73 3D loss and Adam update vs independent reference',max_abs_weight_error:max,async_reads:actual.reads});
 }finally{a.dispose();b.dispose();oa.dispose();ob.dispose();tf.dispose([xy,mask,target]);}
}
{
 const a=new WB73(tf,20261103),b=new WB73(tf,20261103),oa=tf.train.adam(.001),ob=tf.train.adam(.001);
 const image=tf.randomUniform([1,256,192,3],0,1,'float32',17),target=tf.fill([1,73,2],.5),mask=tf.ones([1,73]);
 try{
  const loss=model=>model.loss2d(image,target,mask),actual=await counted(()=>stableStep(tf,()=>loss(a),a.poseVars,oa));
  const expected=await reference(()=>loss(b),b.poseVars,ob);close([actual.value],[expected],'WholeBody2D loss');
  let max=0;for(let i=0;i<a.poseVars.length;i++)max=Math.max(max,close(a.poseVars[i].dataSync(),b.poseVars[i].dataSync(),'WholeBody2D weights'));
  assert.equal(actual.reads,2);tests.push({name:'actual WholeBody73 image CNN seeded-input loss and all 100 parameter-tensor updates',max_abs_weight_error:max,async_reads:actual.reads});
 }finally{a.dispose();b.dispose();oa.dispose();ob.dispose();tf.dispose([image,target,mask]);}
}
assert.equal(tf.memory().numTensors,0,'all validation tensors disposed');
console.log(JSON.stringify({backend:tf.getBackend(),tfjs:tf.version.tfjs,tests,speed_measured:false},null,2));
