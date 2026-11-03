const enc=new TextEncoder();
function vi(n){let v=BigInt(n);if(v<0)v=BigInt.asUintN(64,v);const a=[];do{let b=Number(v&127n);v>>=7n;a.push(b|(v?128:0));}while(v);return Uint8Array.from(a);}
const cat=a=>{const b=new Uint8Array(a.reduce((s,x)=>s+x.length,0));let p=0;for(const x of a){b.set(x,p);p+=x.length;}return b;};
const bytes=(id,b)=>cat([vi(id*8+2),vi(b.length),b]),s=(id,t)=>bytes(id,enc.encode(t)),num=(id,n)=>cat([vi(id*8),vi(n)]);
function attr(name,value){const a=[s(1,name)];if(typeof value==='string')return cat([...a,s(4,value),num(20,3)]);if(Array.isArray(value))return cat([...a,bytes(8,cat(value.map(vi))),num(20,7)]);if(Number.isInteger(value))return cat([...a,num(3,value),num(20,2)]);const f=new Float32Array([value]);return cat([...a,cat([vi(2*8+5),new Uint8Array(f.buffer)]),num(20,1)]);}
function tensor(name,shape,data,type=1){let raw;if(type===7){const b=new ArrayBuffer(data.length*8),d=new DataView(b);data.forEach((v,i)=>d.setBigInt64(i*8,BigInt(v),true));raw=new Uint8Array(b);}else raw=new Uint8Array(Float32Array.from(data).buffer);return cat([...shape.map(v=>num(1,v)),num(2,type),s(8,name),bytes(9,raw)]);}
function valueInfo(name,shape){const dims=cat(shape.map(v=>bytes(1,typeof v==='number'?num(1,v):s(2,v))));return cat([s(1,name),bytes(2,bytes(1,cat([num(1,1),bytes(2,dims)])))]);}
export class ONNX{
 constructor(weights){this.nodes=[];this.initializers=[];this.weights=weights;this.next=0;this.inputs=[];this.outputs=[];this.known=new Map();}
 input(name,shape){this.inputs.push(valueInfo(name,shape));return name;}
 constant(data,shape=[data.length],type=1,name=`c${this.next++}`){this.initializers.push(tensor(name,shape,data,type));return name;}
 param(name){if(!this.known.has(name)){const w=this.weights.get(name);if(!w)throw Error(`Missing parameter ${name}`);this.initializers.push(tensor(name,w.shape,w.data));this.known.set(name,name);}return name;}
 op(op,inputs,attrs={},name=`v${this.next++}`){this.nodes.push(cat([...inputs.map(v=>s(1,v)),s(2,name),s(3,name),s(4,op),...Object.entries(attrs).map(([k,v])=>bytes(5,attr(k,v)))]));return name;}
 output(name,value,shape){this.op('Identity',[value],{},name);this.outputs.push(valueInfo(name,shape));}
 build(){const g=cat([...this.nodes.map(x=>bytes(1,x)),s(2,'WB73-HourglassLite-RandomInit'),...this.initializers.map(x=>bytes(5,x)),...this.inputs.map(x=>bytes(11,x)),...this.outputs.map(x=>bytes(12,x))]);return cat([num(1,8),s(2,'WholeBody73 WebGPU'),s(3,'1'),bytes(7,g),bytes(8,cat([s(1,''),num(2,13)]))]);}
}
