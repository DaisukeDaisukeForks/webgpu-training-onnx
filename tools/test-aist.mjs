import assert from 'node:assert/strict';
import {rotationMatrix,projectFrame} from '../public/aist.mjs';
import {readNumpyPickle} from '../public/numpy-pickle.mjs';
import {Zip} from '../public/zip.mjs';
import fs from 'node:fs';
assert.deepEqual(rotationMatrix([0,0,0]),[1,0,0,0,1,0,0,0,1]);
const r=rotationMatrix([0,0,Math.PI/2]);assert(Math.abs(r[0])<1e-12&&Math.abs(r[1]+1)<1e-12&&Math.abs(r[3]-1)<1e-12);
const world=new Float32Array(51);for(let i=0;i<17;i++)world.set([(i%2?1:-1)*20,(i-8)*10,0],i*3);
const cam={rotation:[0,0,0],translation:[0,0,500],matrix:[[1000,0,960],[0,1000,540],[0,0,1]],distortions:[0,0,0,0,0],size:[1920,1080]};
const row=projectFrame(world,0,cam,100);assert(row&&row.every(Number.isFinite));assert(row.slice(84).every(x=>x===0));
assert(Math.abs(row[6*7+3]+row[7*7+3])<1e-6);assert(Math.abs(row[6*7+4]+row[7*7+4])<1e-6);
assert(Math.abs(row[0*7+3]-.2)<1e-6);assert(Math.abs(row[0*7+4]-.65)<1e-6);
assert.equal(projectFrame(world,0,{...cam,translation:[0,0,-500]},100),null);
// Reject a pickle asking to call arbitrary Python code.
const malicious=Buffer.concat([Buffer.from([0x80,4,0x8c,2]),Buffer.from('os'),Buffer.from([0x8c,6]),Buffer.from('system'),Buffer.from([0x93,0x2e])]);
assert.throws(()=>readNumpyPickle(malicious),/Rejected pickle global/);
if(process.env.AIST_CACHE){
 for(const name of ['aist-keypoints3d','aist-motions']){const decoded=readNumpyPickle(fs.readFileSync(process.env.AIST_CACHE+'/'+name+'-sample.pkl'));assert(Object.keys(decoded).length>0);}
 const zip=await Zip.open(new Blob([fs.readFileSync(process.env.AIST_CACHE+'/cameras.zip')]));assert(zip.entries.has('cameras/mapping.txt'));assert((await zip.bytes('cameras/mapping.txt')).length>1000);
}
console.log('PASS: camera projection; metric scale; Unity Y axis; hip center; absent joints masked; invalid geometry; rejected executable pickle; actual NumPy/ZIP samples when provided.');
