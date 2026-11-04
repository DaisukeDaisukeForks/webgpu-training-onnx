// One packed diagnostic readback per update; no per-weight CPU copies.
export async function step(tf,model,optimizer,batch){
  let result,diagnostic,factor;
  try{
    result=tf.variableGrads(()=>model.loss(batch),model.vars);
    const grads=Object.values(result.grads);
    [factor,diagnostic]=tf.tidy(()=>{
      const peak=tf.stack(grads.map(g=>g.abs().max())).max(),divisor=peak.maximum(1e-12);
      const square=tf.stack(grads.map(g=>g.div(divisor).square().sum())).sum();
      const factor=tf.scalar(1).div(divisor).div(square.maximum(1).sqrt()).minimum(1);
      return[factor,tf.stack([result.value,peak,square,factor])];
    });
    const values=await diagnostic.data();if(!values.every(Number.isFinite))throw Error('損失または勾配が非有限です。この更新は適用していません。');
    tf.tidy(()=>optimizer.applyGradients(Object.fromEntries(Object.entries(result.grads).map(([name,g])=>[name,g.mul(factor)]))));
    return values[0];
  }finally{factor?.dispose();diagnostic?.dispose();result?.value.dispose();if(result)Object.values(result.grads).forEach(t=>t.dispose());}
}
export function seeded(state){return()=>{state.rng=(Math.imul(1664525,state.rng)+1013904223)>>>0;return state.rng/4294967296;};}
export function shuffled(keys,random){const out=[...keys];for(let i=out.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[out[i],out[j]]=[out[j],out[i]];}return out;}
