// Read ZIP members without expanding an entire dataset archive in memory.
const decoder=new TextDecoder('utf-8',{fatal:true});
const table=Uint32Array.from({length:256},(_,n)=>{for(let k=0;k<8;k++)n=n&1?0xedb88320^(n>>>1):n>>>1;return n>>>0;});
function crc32(bytes){let crc=0xffffffff;for(const b of bytes)crc=table[(crc^b)&255]^(crc>>>8);return(crc^0xffffffff)>>>0;}
export class Zip{
  static async open(blob){
    const zip=new Zip();zip.blob=blob;zip.entries=new Map();
    const tail=new Uint8Array(await blob.slice(Math.max(0,blob.size-65557)).arrayBuffer()),v=new DataView(tail.buffer);let end=-1;
    for(let i=tail.length-22;i>=0;i--)if(v.getUint32(i,true)===0x06054b50&&i+22+v.getUint16(i+20,true)===tail.length){end=i;break;}
    if(end<0||v.getUint16(end+4,true)||v.getUint16(end+6,true))throw Error('Unsupported ZIP directory');
    const count=v.getUint16(end+10,true),size=v.getUint32(end+12,true),offset=v.getUint32(end+16,true);
    if(count===65535||size===0xffffffff||offset+size>blob.size)throw Error('Unsupported ZIP64/directory');
    const bytes=new Uint8Array(await blob.slice(offset,offset+size).arrayBuffer()),d=new DataView(bytes.buffer);let p=0;
    for(let i=0;i<count;i++){
      if(p+46>size||d.getUint32(p,true)!==0x02014b50)throw Error('Invalid ZIP central entry');
      const flags=d.getUint16(p+8,true),method=d.getUint16(p+10,true),crc=d.getUint32(p+16,true),compressed=d.getUint32(p+20,true),length=d.getUint32(p+24,true),n=d.getUint16(p+28,true),extra=d.getUint16(p+30,true),comment=d.getUint16(p+32,true),start=d.getUint32(p+42,true);
      if(flags&1||![0,8].includes(method)||p+46+n+extra+comment>size)throw Error('Unsupported ZIP member');
      const name=decoder.decode(bytes.subarray(p+46,p+46+n));
      if(zip.entries.has(name))throw Error('Duplicate ZIP member');
      zip.entries.set(name,{name,method,crc,compressed,length,start});p+=46+n+extra+comment;
    }
    if(p!==size)throw Error('ZIP directory length mismatch');return zip;
  }
  async bytes(name){
    const e=this.entries.get(name);if(!e)throw Error(`Missing archive member: ${name}`);
    const h=new DataView(await this.blob.slice(e.start,e.start+30).arrayBuffer());
    if(h.byteLength!==30||h.getUint32(0,true)!==0x04034b50||h.getUint16(8,true)!==e.method)throw Error('Invalid ZIP local header');
    const start=e.start+30+h.getUint16(26,true)+h.getUint16(28,true),end=start+e.compressed;
    if(end>this.blob.size)throw Error('Truncated ZIP member');
    const stream=this.blob.slice(start,end).stream(),decoded=e.method===8?stream.pipeThrough(new DecompressionStream('deflate-raw')):stream;
    const bytes=new Uint8Array(await new Response(decoded).arrayBuffer());
    if(bytes.length!==e.length||crc32(bytes)!==e.crc)throw Error(`ZIP CRC mismatch: ${name}`);
    return bytes;
  }
}
