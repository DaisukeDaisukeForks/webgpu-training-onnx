import {WB73,TFOps,ExportOps,smoothL1} from './model.mjs?revision=7';
import {ONNX} from './onnx.mjs?revision=4';
export class Video73 extends WB73{
 constructor(tf,seed){super(tf,seed);this.weight('video.in.w',[480,256],Math.sqrt(2/480));this.weight('video.in.b',[256],0);this.weight('video.hidden.w',[256,256],Math.sqrt(2/256));this.weight('video.hidden.b',[256],0);this.weight('video.out.w',[256,36],.001);this.weight('video.out.b',[36],0);this.videoVars=[...this.poseVars,...[...this.weights].filter(([n])=>n.startsWith('video.')).map(([,v])=>v)];}
 head(o,features){let x=o.act(o.dense(features,'video.in'));x=o.add(x,o.act(o.dense(x,'video.hidden')));return o.reshape(o.dense(x,'video.out'),[-1,12,3]);}
 predictVideo(frames,rotation=null,origin=null){const tf=this.tf;return tf.tidy(()=>{const b=frames.shape[0],p=this.pose(frames.reshape([b*3,256,192,3])),xy=p.xy.reshape([b,3,73,2]).slice([0,2,0,0],[-1,1,73,2]).reshape([b,73,2]),confidence=p.confidence.reshape([b,3,73]).slice([0,2,0],[-1,1,73]).reshape([b,73]);const frozen=tf.customGrad((x,c)=>({value:this.lift(x,c),gradFunc:()=>[tf.zerosLike(x),tf.zerosLike(c)]})),base=frozen(xy,confidence),body=this.head(new TFOps(this),p.features.reshape([b,480])),correction=tf.concat([body,tf.zeros([b,61,3])],1),relative=new TFOps(this).root(base.add(correction));let positions=relative;if(rotation)positions=relative.reshape([-1,3]).matMul(rotation).reshape([b,73,3]);if(origin)positions=positions.add(origin);return{xy,confidence,relative,positions};});}
 // Unavailable 3D joints are masked, never fabricated from a different image.
 trainingSequence(frames){const tf=this.tf;return tf.tidy(()=>{
  const b=frames.shape[0],p=this.pose(frames.reshape([b*3,256,192,3])),features=p.features.reshape([b,3,160]),at=i=>features.slice([0,i,0],[-1,1,160]).reshape([b,160]);
  const f0=at(0),f1=at(1),f2=at(2),contexts=tf.stack([tf.concat([f0,f0,f0],1),tf.concat([f0,f0,f1],1),tf.concat([f0,f1,f2],1)],1).reshape([b*3,480]);
  const frozen=tf.customGrad((x,c)=>({value:this.lift(x,c),gradFunc:()=>[tf.zerosLike(x),tf.zerosLike(c)]})),body=this.head(new TFOps(this),contexts),relative=new TFOps(this).root(frozen(p.xy,p.confidence).add(tf.concat([body,tf.zeros([b*3,61,3])],1))).reshape([b,3,73,3]);
  return{relative,xy:p.xy.reshape([b,3,73,2]).slice([0,2,0,0],[-1,1,73,2]).reshape([b,73,2])};
 });}
 lossVideo(frames,target,mask,xy,valid2d,sequence=null,sequenceMask=null){const tf=this.tf;return tf.tidy(()=>{
  if(!sequence||!sequenceMask){const p=this.predictVideo(frames),w=mask.expandDims(-1),l3=smoothL1(p.relative.sub(target).abs()).mul(w).sum().div(w.sum().mul(3).maximum(1)),l2=p.xy.sub(xy).square().sum(-1).mul(valid2d).sum().div(valid2d.sum().maximum(1));return l3.add(l2.mul(.25));}
  const p=this.trainingSequence(frames),w=sequenceMask.expandDims(-1),l3=smoothL1(p.relative.sub(sequence).abs()).mul(w).sum().div(w.sum().mul(3).maximum(1)),l2=p.xy.sub(xy).square().sum(-1).mul(valid2d).sum().div(valid2d.sum().maximum(1));
  const delta=x=>x.slice([0,1,0,0],[-1,2,73,3]).sub(x.slice([0,0,0,0],[-1,2,73,3])),motionMask=sequenceMask.slice([0,1,0],[-1,2,73]).mul(sequenceMask.slice([0,0,0],[-1,2,73])).expandDims(-1);
  // Match real displacement, rather than rewarding a frozen pose.
  const velocity=smoothL1(delta(p.relative).sub(delta(sequence)).abs()).mul(motionMask).sum().div(motionMask.sum().mul(3).maximum(1));
  let structure=tf.scalar(0);const edges=[[0,2],[2,4],[1,3],[3,5],[6,8],[8,10],[7,9],[9,11]],joint=(x,j)=>x.slice([0,0,j,0],[-1,3,1,3]),present=j=>sequenceMask.slice([0,0,j],[-1,3,1]);
  for(const[a,b]of edges){const m=present(a).mul(present(b)),length=x=>joint(x,a).sub(joint(x,b)).square().sum(-1).add(1e-8).sqrt();structure=structure.add(smoothL1(length(p.relative).sub(length(sequence)).abs()).mul(m).sum().div(m.sum().maximum(1)));}
  const m=present(10).mul(present(11)).expandDims(-1),separation=x=>joint(x,10).sub(joint(x,11));
  const legs=smoothL1(separation(p.relative).sub(separation(sequence)).abs()).mul(m).sum().div(m.sum().mul(3).maximum(1));
  return l3.add(l2.mul(.25)).add(velocity.mul(.2)).add(structure.div(edges.length).mul(.1)).add(legs.mul(.1));
 });}
 async export(){const params=new Map();for(const[n,w]of this.weights)params.set(n,{shape:w.shape,data:await w.data()});const g=new ONNX(params),o=new ExportOps(g,params),frames=g.input('frames',[1,3,3,256,192]),rotation=g.input('tracking_rotation',[3,3]),origin=g.input('tracking_origin',[1,1,3]),features=[];let latest;
 // Share convolution initializers across the three image branches.
 const constant=g.constant.bind(g),named=new Map();g.constant=(data,shape,type,name)=>{if(name&&named.has(name))return named.get(name);const id=constant(data,shape,type,name);if(name)named.set(name,id);return id;};
 for(let i=0;i<3;i++){const frame=o.reshape(o.slice(frames,i,i+1,1),[1,3,256,192]);latest=this.forward(o,g.op('Transpose',[frame],{perm:[0,2,3,1]}));features.push(latest.features);}
 const body=this.head(o,g.op('Concat',features,{axis:1})),correction=g.op('Concat',[body,g.constant(new Float32Array(183),[1,61,3])],{axis:1}),relative=o.root(o.add(this.lifter(o,latest.xy,latest.confidence),correction)),positions=o.add(o.reshape(g.op('MatMul',[o.reshape(relative,[-1,3]),rotation]),[1,73,3]),origin);g.output('keypoints2d',latest.xy,[1,73,2]);g.output('confidence2d',latest.confidence,[1,73]);g.output('vrc_relative',relative,[1,73,3]);g.output('vrc_positions',positions,[1,73,3]);return g.build();}
}
