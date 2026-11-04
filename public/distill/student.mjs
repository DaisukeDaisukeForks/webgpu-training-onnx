import {ONNX} from '../onnx.mjs';
import {TFOps,ExportOps,smoothL1} from '../model.mjs?revision=7';
import {K,BINS,Z_RANGE,EDGES} from './schema.mjs';
let instance=0;
export class Student37 {
  constructor(tf,seed=20261104){
    this.tf=tf;this.weights=new Map();this.seed=seed;this.serial=0;this.prefix=`wb${instance++}_`;
    const weight=(name,shape,std=0,fill=0)=>{const t=std?tf.randomNormal(shape,0,std,'float32',seed+this.serial++):tf.fill(shape,fill);this.weights.set(name,tf.variable(t,true,this.prefix+name));t.dispose();};
    const conv=(n,ci,co,k=1,depth=false)=>{weight(n+'.w',depth?[k,k,ci,1]:[k,k,ci,co],Math.sqrt(2/(k*k*(depth?1:ci))));weight(n+'.b',[co]);};
    const norm=(n,c)=>{weight(n+'.g',[c],0,1);weight(n+'.b',[c]);};
    const dense=(n,ci,co)=>{weight(n+'.w',[ci,co],Math.sqrt(1/ci));weight(n+'.b',[co]);};
    conv('d37.stem',3,32,3);norm('d37.stemn',32);
    this.blocks=[];let channels=32;
    for(const [out,stride,expand] of [[64,2,2],[64,1,2],[64,1,2],[128,2,3],[128,1,3],[128,1,3],[192,2,3],[192,1,3],[192,1,3]]){
      const n=`d37.b${this.blocks.length}`,hidden=channels*expand;
      conv(n+'.e',channels,hidden);norm(n+'.en',hidden);conv(n+'.d',hidden,hidden,3,true);norm(n+'.dn',hidden);conv(n+'.p',hidden,out);norm(n+'.pn',out);
      this.blocks.push({n,stride,residual:channels===out&&stride===1});channels=out;
    }
    conv('d37.map',192,K);dense('d37.spatial',16*12,128);dense('d37.context',192,128);dense('d37.mix',128,128);
    for(let a=0;a<3;a++)dense(`d37.axis${a}`,128,BINS[a]);
    this.vars=[...this.weights.values()];
  }
  forward(o,image){
    let x=o.act(o.norm(o.conv(image,'d37.stem',2),'d37.stemn'));
    for(const {n,stride,residual} of this.blocks){let y=o.act(o.norm(o.conv(x,n+'.e'),n+'.en'));y=o.act(o.norm(o.conv(y,n+'.d',stride),n+'.dn'));y=o.norm(o.conv(y,n+'.p'),n+'.pn');x=residual?o.add(x,y):y;}
    const global=o.reshape(o.dense(o.pool(x),'d37.context'),[-1,1,128]);
    const spatial=o.reshape(o.transpose(o.conv(x,'d37.map'),[0,3,1,2]),[-1,K,192]);
    let tokens=o.act(o.add(o.dense3(spatial,'d37.spatial',192,128),global));
    tokens=o.add(tokens,o.act(o.dense3(tokens,'d37.mix',128,128)));
    return BINS.map((n,a)=>o.dense3(tokens,`d37.axis${a}`,128,n));
  }
  logits(image){return this.forward(new StudentTF(this),image);}
  loss(batch){const tf=this.tf;return tf.tidy(()=>{
    const logits=this.logits(batch.image),probs=logits.map(t=>tf.softmax(t,-1));
    // Every body/foot/hand group has its own denominator. Feet cannot be hidden by other points.
    const groups=[[0,17,1],[17,23,2],[23,30,1],[30,37,1]];
    let loss=tf.scalar(0);
    for(let a=0;a<3;a++){
      const ce=batch.axes[a].mul(tf.logSoftmax(logits[a],-1)).sum(-1).neg();
      const entropy=batch.axes[a].mul(batch.axes[a].maximum(1e-8).log()).sum(-1).neg();
      for(const [lo,hi,scale] of groups){const w=batch.mask.slice([0,lo],[-1,hi-lo]);loss=loss.add(ce.sub(entropy).slice([0,lo],[-1,hi-lo]).mul(w).sum().div(w.sum().maximum(1)).mul(scale/15));}
    }
    const expectation=tf.stack(probs.map((p,a)=>p.mul(tf.range(0,BINS[a]).add(.5).div(BINS[a])).sum(-1)),-1);
    const d=smoothL1(expectation.sub(batch.position).abs(),.02).mean(-1);
    loss=loss.add(d.mul(batch.mask).sum().div(batch.mask.sum().maximum(1)));
    // Relative leg and heel/toe vectors retain lifted/forward feet; no floor or crossing constraint.
    const starts=EDGES.map(e=>e[0]),ends=EDGES.map(e=>e[1]);
    const delta=t=>tf.gather(t,ends,1).sub(tf.gather(t,starts,1));
    const edgeMask=tf.gather(batch.mask,starts,1).minimum(tf.gather(batch.mask,ends,1));
    const error=smoothL1(delta(expectation).sub(delta(batch.position)).mul(tf.tensor1d([1,1,2])).abs(),.02).mean(-1);
    loss=loss.add(error.mul(edgeMask).sum().div(edgeMask.sum().maximum(1)).mul(2));
    // Real COCO/CMU annotations remain stronger than incorrect pseudo-labels for XY.
    if(batch.gtMask){const xy=expectation.slice([0,0,0],[-1,K,2]);const gt=smoothL1(xy.sub(batch.gtXY).abs(),.02).mean(-1);loss=loss.add(gt.mul(batch.gtMask).sum().div(batch.gtMask.sum().maximum(1)).mul(2));}
    return loss;
  });}
  predict(image){const tf=this.tf;return tf.tidy(()=>{const logits=this.logits(image),axes=logits.map((t,a)=>t.argMax(-1).toFloat().add(.5).div(BINS[a]));return{keypoints:tf.stack([axes[0].mul(192),axes[1].mul(256),axes[2].mul(2).sub(1).mul(Z_RANGE)],-1),scores:tf.softmax(logits[0],-1).max(-1).minimum(tf.softmax(logits[1],-1).max(-1))};});}
  async export(){
    const params=new Map();for(const[n,t]of this.weights)params.set(n,{shape:t.shape,data:await t.data()});
    const g=new ONNX(params),o=new StudentExport(g,params),image=g.input('image',[1,3,256,192]);
    const logits=this.forward(o,g.op('Transpose',[image],{perm:[0,2,3,1]}));
    const axis=logits.map((t,a)=>g.op('Div',[g.op('Add',[g.op('Cast',[g.op('ArgMax',[t],{axis:2,keepdims:1})],{to:1}),o.c(.5)]),o.c(BINS[a])]));
    const xyz=g.op('Concat',[o.mul(axis[0],o.c(192)),o.mul(axis[1],o.c(256)),o.mul(o.sub(o.mul(axis[2],o.c(2)),o.c(1)),o.c(Z_RANGE))],{axis:2});
    const peaks=logits.slice(0,2).map(t=>g.op('ReduceMax',[g.op('Softmax',[t],{axis:-1})],{axes:[2],keepdims:0}));
    g.output('keypoints',xyz,[1,K,3]);g.output('scores',g.op('Min',peaks),[1,K]);return g.build();
  }
  dispose(){this.vars.forEach(t=>t.dispose());}
}
class StudentTF extends TFOps {
  transpose(x,p){return x.transpose(p);}
  dense3(x,n,ci,co){return this.dense(x.reshape([-1,ci]),n).reshape([-1,K,co]);}
}
class StudentExport extends ExportOps {
  transpose(x,p){return this.g.op('Transpose',[x],{perm:p});}
  dense3(x,n,ci,co){return this.reshape(this.dense(this.reshape(x,[-1,ci]),n),[-1,K,co]);}
}
