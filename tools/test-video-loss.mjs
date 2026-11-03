import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {Video73} from '../public/video-model.mjs';
const require=createRequire(import.meta.url),tf=require(process.env.TFJS_CPU_MODULE||'@tensorflow/tfjs');
await tf.setBackend('cpu');await tf.ready();const m=new Video73(tf,20261103);
const input=tf.zeros([1,3,256,192,3]),target=tf.fill([1,73,3],.02),mask=tf.concat([tf.ones([1,12]),tf.zeros([1,61])],1),xy=tf.fill([1,73,2],.5),sequence=tf.stack([target.mul(.8),target.mul(.9),target],1),sequenceMask=tf.stack([mask,mask,mask],1);
const a=m.predictVideo(input),b=m.trainingSequence(input),last=b.relative.slice([0,2,0,0],[-1,1,73,3]).reshape([1,73,3]);
const difference=(await a.relative.sub(last).abs().max().data())[0];assert(difference<1e-5);tf.dispose([a,b,last]);
const g=tf.variableGrads(()=>m.lossVideo(input,target,mask,xy,mask,sequence,sequenceMask),m.videoVars);assert(Number.isFinite((await g.value.data())[0]));
let count=0,nonzero=0;for(const t of Object.values(g.grads))for(const v of await t.data()){assert(Number.isFinite(v));count++;if(v!==0)nonzero++;}
assert(nonzero>0);console.log(JSON.stringify({pass:true,blackInputFiniteGradients:count,nonzero,trainingInferenceMaxDifference:difference,loss:(await g.value.data())[0]}));
tf.dispose([input,target,mask,xy,sequence,sequenceMask,g]);m.dispose();
