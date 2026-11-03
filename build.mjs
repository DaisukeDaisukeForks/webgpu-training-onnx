import './tools/verify-data.mjs';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {gzipSync} from 'node:zlib';
const root=path.dirname(new URL(import.meta.url).pathname);
const publicDir=path.join(root,'public'),dist=path.join(root,'dist'),teachers={};
const teacherPattern=/^coco-.*\.json$|^cmu-foot-.*\.json$|^aist\.json$|^h3wb\.json$|^(h3wb|aspset3d)-.*\.(bin|json)$|^paired\.json$|^paired\/.*\.jpg$/;
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
for(const name of fs.readdirSync(publicDir,{recursive:true}).filter(n=>teacherPattern.test(n))){
  const b=fs.readFileSync(path.join(publicDir,name));teachers[name]={bytes:b.length,sha256:sha(b)};
  if(/\.(bin|json)$/.test(name)){const packed=gzipSync(b,{level:6});teachers[name].transport={file:name+'.packed',bytes:packed.length,sha256:sha(packed)};}
}
fs.writeFileSync(path.join(publicDir,'teacher-index.mjs'),`export const TEACHERS=${JSON.stringify(teachers)};\n`);
if(!fs.existsSync(path.join(publicDir,'index.html')))throw Error('Missing source; build preserved');
fs.rmSync(dist,{recursive:true,force:true});fs.mkdirSync(path.join(dist,'server'),{recursive:true});
for(const entry of fs.readdirSync(publicDir,{recursive:true,withFileTypes:true})){
 if(!entry.isFile())continue;
 const from=path.join(entry.parentPath,entry.name),name=path.relative(publicDir,from),item=teachers[name],to=path.join(dist,'client',item?.transport?.file||name);
 fs.mkdirSync(path.dirname(to),{recursive:true});
 if(item?.transport)fs.writeFileSync(to,gzipSync(fs.readFileSync(from),{level:6}));
 else if(name.startsWith('paired/')&&name.endsWith('.jpg')){try{fs.linkSync(from,to);}catch{fs.copyFileSync(from,to);}}
 else fs.copyFileSync(from,to);
}
fs.mkdirSync(path.join(dist,'.openai'),{recursive:true});fs.copyFileSync(path.join(root,'.openai/hosting.json'),path.join(dist,'.openai/hosting.json'));
const rows=['coco-train','coco-val','cmu-foot-train','cmu-foot-val'].flatMap(name=>JSON.parse(fs.readFileSync(path.join(publicDir,name+'.json'),'utf8')).rows),allowed=[...new Set(rows.map(r=>`${r.split}/${r.id}`))];
const sourceAssets=JSON.parse(fs.readFileSync(path.join(root,'config/source-assets.json'),'utf8'));
fs.writeFileSync(path.join(dist,'server/index.js'),fs.readFileSync(path.join(root,'worker/index.mjs'),'utf8')+`\nexport default handler(new Set(${JSON.stringify(allowed)}),${JSON.stringify(sourceAssets)});\n`);
console.log(JSON.stringify({images:allowed.length,annotations:rows.length,teacherFiles:Object.keys(teachers).length}));
