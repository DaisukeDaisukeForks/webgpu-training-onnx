// Expose existing SimCC tensors in a decoded RTMW3D ONNX. Weights are unchanged.
const text=new TextDecoder('utf-8',{fatal:true}),enc=new TextEncoder();
function fields(a){const out=[];let p=0;function read(){let v=0,s=0,b;do{if(p>=a.length||s>49)throw Error('不正なONNX varint');b=a[p++];v+=(b&127)*2**s;s+=7;}while(b&128);return v;}while(p<a.length){const start=p,tag=read(),id=Math.floor(tag/8),wire=tag%8;let value;if(wire===2){const n=read();if(n<0||p+n>a.length)throw Error('ONNXが途中で切れています');value=a.subarray(p,p+n);p+=n;}else if(wire===0){read();}else if(wire===1)p+=8;else if(wire===5)p+=4;else throw Error('未対応のONNX field');if(p>a.length)throw Error('ONNXが途中で切れています');out.push({id,wire,value,raw:a.subarray(start,p)});}return out;}
function vi(v){const a=[];do{const b=v%128;v=Math.floor(v/128);a.push(b+(v?128:0));}while(v);return Uint8Array.from(a);}
function cat(parts){const out=new Uint8Array(parts.reduce((n,a)=>n+a.length,0));let p=0;for(const a of parts){out.set(a,p);p+=a.length;}return out;}
const bytes=(n,v)=>cat([vi(n*8+2),vi(v.length),v]),num=(n,v)=>cat([vi(n*8),vi(v)]);
function info(name,shape){return cat([bytes(1,enc.encode(name)),bytes(2,bytes(1,cat([num(1,1),bytes(2,cat(shape.map(n=>bytes(1,num(1,n)))))])))]);}
export function exposeSimCC(model){
  const outer=fields(model),graphField=outer.find(f=>f.id===7);if(!graphField)throw Error('ONNX graphがありません');const graph=fields(graphField.value);
  const names=['simcc_x','simcc_y','simcc_z'],outputs=graph.filter(f=>f.id===12).map(f=>text.decode(fields(f.value).find(g=>g.id===1).value));if(names.every(n=>outputs.includes(n)))return model;
  const produced=new Set();for(const field of graph.filter(f=>f.id===1))for(const n of fields(field.value).filter(f=>f.id===2))produced.add(text.decode(n.value));
  if(!names.every(n=>produced.has(n)))throw Error('このONNXにはRTMW3DのSimCC出力がありません。公式モデルのraw FP32 ONNXを選んでください。');
  const rebuilt=cat([...graph.filter(f=>f.id!==12&&f.id!==13).map(f=>f.raw),...names.map((name,a)=>bytes(12,info(name,[1,133,[576,768,576][a]])))]);
  return cat(outer.map(f=>f.id===7?bytes(7,rebuilt):f.raw));
}
