const sameShape=(a,b)=>Array.isArray(a)&&a.length===b.length&&a.every((v,i)=>v===b[i]);
const logicalName=name=>name.replace(/^wb\d+_/,'');
export function validateTensor(item,shape,name=item?.name){
 if(!item||!sameShape(item.shape,shape)||!Array.isArray(item.data)||item.data.length!==shape.reduce((a,b)=>a*b,1)||item.data.some(v=>typeof v!=='number'||!Number.isFinite(v)))throw Error(`不正な保存テンソル: ${name}`);
}
export function validateWeights(weights,model){
 if(!Array.isArray(weights)||weights.length!==model.weights.size||new Set(weights.map(w=>w.name)).size!==weights.length)throw Error('保存されたモデルの重み数が一致しません');
 for(const w of weights){const v=model.weights.get(w.name);if(!v)throw Error(`未対応の重み: ${w.name}`);validateTensor(w,v.shape);}
}
export async function readWeights(model){
 const out=[];
 for(const[name,tensor]of model.weights){const item={name,shape:tensor.shape,dtype:tensor.dtype,data:Array.from(await tensor.data())};validateTensor(item,tensor.shape);out.push(item);}
 return out;
}
export function assignWeights(tf,model,weights){
 validateWeights(weights,model);
 for(const w of weights)tf.tidy(()=>model.weights.get(w.name).assign(tf.tensor(w.data,w.shape,w.dtype||'float32')));
}
export async function readAdam(optimizer){
 const items=await optimizer.getWeights(),weights=[];
 try{for(const w of items){const item={name:w.name,shape:w.tensor.shape,dtype:w.tensor.dtype,data:Array.from(await w.tensor.data())};validateTensor(item,w.tensor.shape);weights.push(item);}}
 finally{items.filter(w=>w.name==='iter').forEach(w=>w.tensor.dispose());}
 // getWeights() omits these two accumulated float32 values. Recomputing beta^t
 // on restore need not equal the original sequence of float32 multiplications.
 const powers={accBeta1:(await optimizer.accBeta1.data())[0],accBeta2:(await optimizer.accBeta2.data())[0]};
 return{weights,metadata:{config:optimizer.getConfig(),powers}};
}
export async function restoreAdam(tf,entries,metadata,variables,learningRate){
 const config=metadata?.config||{learningRate,beta1:.9,beta2:.999,epsilon:1e-7};
 if(!Number.isFinite(config.learningRate)||config.learningRate<=0||!Number.isFinite(config.epsilon)||config.epsilon<=0||!['beta1','beta2'].every(k=>Number.isFinite(config[k])&&config[k]>=0&&config[k]<1))throw Error('Adamの保存設定が不正です');
 if(!Array.isArray(entries)||entries[0]?.name!=='iter')throw Error('Adamの更新回数がありません');
 validateTensor(entries[0],[]);
 const iterations=entries[0].data[0];
 if(!Number.isInteger(iterations)||iterations<0||!(entries.length===1&&iterations===0||entries.length===1+2*variables.length))throw Error('Adamの保存状態が不完全です');
 for(let i=1;i<entries.length;i++){
  const v=variables[(i-1)%variables.length],suffix=i<=variables.length?'/m':'/v';
  validateTensor(entries[i],v.shape);
  if(logicalName(entries[i].name)!==logicalName(v.name)+suffix)throw Error(`Adamの重み順が一致しません: ${entries[i].name}`);
 }
 const powers=metadata?.powers;
 if(powers&&!['accBeta1','accBeta2'].every(k=>Number.isFinite(powers[k])&&powers[k]>=0&&powers[k]<=1))throw Error('Adamの累積係数が不正です');
 const opt=tf.train.adam(config.learningRate,config.beta1,config.beta2,config.epsilon),tensors=[];
 try{
  for(const w of entries)tensors.push({name:w.name,tensor:tf.tensor(w.data,w.shape,w.dtype||'float32')});
  await opt.setWeights(tensors);
  if(powers)tf.tidy(()=>{opt.accBeta1.assign(tf.scalar(powers.accBeta1));opt.accBeta2.assign(tf.scalar(powers.accBeta2));});
  return opt;
 }catch(e){opt.dispose();throw e;}finally{tensors.forEach(w=>w.tensor.dispose());}
}
