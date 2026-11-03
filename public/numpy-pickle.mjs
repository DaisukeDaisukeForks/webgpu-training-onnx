// Data-only NumPy pickle reader. No imports, eval, or arbitrary reducers execute.
const decoder=new TextDecoder('utf-8',{fatal:true}),MARK=Symbol();
const globals=new Set(['numpy.ndarray','numpy.dtype','numpy.core.multiarray._reconstruct','numpy._core.multiarray._reconstruct','numpy.core.multiarray.scalar','numpy._core.multiarray.scalar']);
function numbers(dtype,bytes){
 const m=/^([fi])(4|8)$/.exec(dtype?.name);if(!m||!(bytes instanceof Uint8Array))throw Error('Unsupported NumPy dtype');
 const size=Number(m[2]),little=dtype.order!=='>';if(bytes.length%size)throw Error('Invalid NumPy bytes');
 const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),out=new Float64Array(bytes.length/size);
 for(let i=0;i<out.length;i++){const p=i*size;out[i]=m[1]==='f'?(size===8?v.getFloat64(p,little):v.getFloat32(p,little)):(size===8?Number(v.getBigInt64(p,little)):v.getInt32(p,little));if(m[1]==='i'&&!Number.isSafeInteger(out[i]))throw Error('Unsafe integer');}return out;
}
export function readNumpyPickle(bytes){
 const v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),stack=[],memo=new Map();let p=0,next=0;
 const take=n=>{if(!Number.isSafeInteger(n)||n<0||p+n>bytes.length)throw Error('Truncated pickle');const b=bytes.subarray(p,p+n);p+=n;return b;};
 const int=(n,signed=false)=>{take(n);return n===1?v.getUint8(p-n):n===2?v.getUint16(p-n,true):n===4?(signed?v.getInt32(p-n,true):v.getUint32(p-n,true)):Number(v.getBigUint64(p-n,true));};
 const marked=()=>{const i=stack.lastIndexOf(MARK);if(i<0)throw Error('Missing MARK');return stack.splice(i).slice(1);};
 const global=(module,name)=>{const key=module+'.'+name;if(!globals.has(key))throw Error('Rejected pickle global: '+key);return{global:key};};
 const assign=items=>{const obj=stack.at(-1);if(!obj||Object.getPrototypeOf(obj)!==null||items.length%2)throw Error('Invalid dictionary');for(let i=0;i<items.length;i+=2){if(typeof items[i]!=='string')throw Error('Invalid key');obj[items[i]]=items[i+1];}};
 while(p<bytes.length){const op=int(1);switch(op){
 case 0x80:if(int(1)!==4)throw Error('Unsupported pickle protocol');break;
 case 0x95:if(p+8+int(8)>bytes.length)throw Error('Invalid frame');break;
 case 0x2e:if(stack.length!==1||p!==bytes.length)throw Error('Invalid pickle end');return stack[0];
 case 0x7d:stack.push(Object.create(null));break;
 case 0x28:stack.push(MARK);break;
 case 0x29:stack.push([]);break;
 case 0x4e:stack.push(null);break;
 case 0x88:stack.push(true);break;
 case 0x89:stack.push(false);break;
 case 0x4b:stack.push(int(1));break;
 case 0x4d:stack.push(int(2));break;
 case 0x4a:stack.push(int(4,true));break;
 case 0x47:take(8);stack.push(v.getFloat64(p-8));break;
 case 0x8c:stack.push(decoder.decode(take(int(1))));break;
 case 0x58:stack.push(decoder.decode(take(int(4))));break;
 case 0x43:stack.push(take(int(1)));break;
 case 0x42:stack.push(take(int(4)));break;
 case 0x94:memo.set(next++,stack.at(-1));break;
 case 0x68:case 0x6a:{const i=int(op===0x68?1:4);if(!memo.has(i))throw Error('Invalid memo');stack.push(memo.get(i));break;}
 case 0x74:stack.push(marked());break;
 case 0x85:case 0x86:case 0x87:{const n=op-0x84;if(stack.length<n)throw Error('Invalid tuple');stack.push(stack.splice(-n));break;}
 case 0x75:assign(marked());break;
 case 0x73:{const value=stack.pop(),key=stack.pop();assign([key,value]);break;}
 case 0x93:{const name=stack.pop(),module=stack.pop();stack.push(global(module,name));break;}
 case 0x52:{const args=stack.pop(),fn=stack.pop()?.global;if(!Array.isArray(args))throw Error('Invalid reducer');
  if(fn==='numpy.dtype')stack.push({kind:'dtype',name:args[0],order:'='});
  else if(fn?.endsWith('._reconstruct')&&args[0]?.global==='numpy.ndarray')stack.push({kind:'ndarray'});
  else if(fn?.endsWith('.scalar'))stack.push(numbers(args[0],args[1])[0]);else throw Error('Rejected reducer');break;}
 case 0x62:{const state=stack.pop(),obj=stack.at(-1);if(!Array.isArray(state))throw Error('Invalid NumPy state');
  if(obj?.kind==='dtype'){if(!['<','>','=','|'].includes(state[1]))throw Error('Invalid byte order');obj.order=state[1];}
  else if(obj?.kind==='ndarray'){const[version,shape,dtype,fortran,raw]=state;if(version!==1||!Array.isArray(shape)||shape.some(n=>!Number.isSafeInteger(n)||n<0)||fortran)throw Error('Unsupported NumPy layout');obj.shape=shape;obj.values=numbers(dtype,raw);if(shape.reduce((a,b)=>a*b,1)!==obj.values.length)throw Error('NumPy shape mismatch');}
  else throw Error('Rejected BUILD');break;}
 default:throw Error('Unsupported pickle opcode '+op.toString(16));
 }}throw Error('Unterminated pickle');
}
