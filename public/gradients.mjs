// Keep reductions and the clipping coefficient on the GPU. A normal update
// reads the numeric summary and boolean flags concurrently, then checks updated weights.
export async function stableStep(tf,fn,vars,optimizer,apply=true){
 let result,summary,validity,factor;
 try{
  result=tf.variableGrads(fn,vars);
  const entries=Object.entries(result.grads);
  ({summary,validity,factor}=tf.tidy(()=>{
   const peak=tf.stack(entries.map(([,v])=>v.abs().max())).max();
   const divisor=tf.where(peak.greater(0),peak,tf.scalar(1));
   // WebGPU AddN binds every input at once (100 grads + output > limit 8).
   // Stack/Concat splits inputs by the device binding limit; Sum reads one buffer.
   const squared=tf.stack(entries.map(([,v])=>v.div(divisor).square().sum())).sum();
   // For a nonzero gradient, the scaled squared norm is at least one.
   // Divide in two stages: peak * sqrt(squared) can overflow for finite grads.
   const factor=tf.scalar(1).div(divisor).div(squared.maximum(1).sqrt()).minimum(1);
   // Keep boolean storage separate: cast() may retain the underlying bool buffer.
   // Submit both readbacks together below, without an intermediate CPU barrier.
   const validity=tf.stack(entries.map(([,v])=>tf.isFinite(v).all()));
   return{factor,validity,summary:tf.stack([result.value,peak,squared,factor])};
  }));
  const [values,flags]=await Promise.all([summary.data(),validity.data()]);
  const [loss,peak,squared,coefficient]=values;
  if(!Number.isFinite(loss))throw Error('損失が非有限値です。重みは更新していません。');
  const bad=entries.filter((_,i)=>!flags[i]).map(([name])=>name);
  if(bad.length)throw Error('非有限の勾配: '+bad.slice(0,6).join(', ')+'。重みは更新していません。');
  if(!Number.isFinite(peak)||!Number.isFinite(squared)||!Number.isFinite(coefficient))throw Error('勾配ノルム計算が非有限値です。重みは更新していません。');
  if(!apply)return loss;
  const before=vars.map(v=>v.clone());let adam=[];
  try{
   const old=await optimizer.getWeights();
   try{adam=old.map(w=>({name:w.name,tensor:w.tensor.clone()}));}
   finally{old.filter(w=>w.name==='iter').forEach(w=>w.tensor.dispose());}
   try{
    tf.tidy(()=>{const clipped={};for(const[n,v]of entries)clipped[n]=v.mul(factor);optimizer.applyGradients(clipped);});
    const check=tf.tidy(()=>tf.stack(vars.map(v=>tf.isFinite(v).all())).all());
    let finite;try{finite=Boolean((await check.data())[0]);}finally{check.dispose();}
    if(!finite)throw Error('更新後の重みが非有限値になったため、更新前へ戻しました。');
   }catch(e){
    vars.forEach((v,i)=>v.assign(before[i]));
    // Adam.setWeights replaces its moment variables without disposing them.
    const replaced=await optimizer.getWeights();
    try{await optimizer.setWeights(adam);}finally{replaced.forEach(w=>w.tensor.dispose());}
    throw e;
   }
  }finally{before.forEach(v=>v.dispose());adam.forEach(w=>w.tensor.dispose());}
  return loss;
 }finally{
  summary?.dispose();validity?.dispose();factor?.dispose();
  if(result){result.value.dispose();Object.values(result.grads).forEach(t=>t.dispose());}
 }
}
