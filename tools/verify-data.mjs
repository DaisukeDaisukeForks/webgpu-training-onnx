import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const root=path.dirname(path.dirname(new URL(import.meta.url).pathname));
const lock=JSON.parse(fs.readFileSync(path.join(root,'config/data-lock.json'),'utf8'));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const read=name=>fs.readFileSync(path.join(root,'public',name));
const json=name=>JSON.parse(read(name).toString('utf8'));
for(const[name,hash]of Object.entries(lock.files)){
 if(!fs.existsSync(path.join(root,'public',name)))throw Error(`Missing teacher ${name}; run npm run data:prepare first`);
 if(sha(read(name))!==hash)throw Error(`Teacher bytes differ from the reference: ${name}`);
}
for(const split of ['train','val']){
 const coco=json(`coco-${split}.json`).rows.map(r=>[r.id,r.bbox,r.points,r.width,r.height]);
 if(sha(JSON.stringify(coco))!==lock.coco[split])throw Error(`COCO selection/labels differ: ${split}`);
}
const paired=json('paired.json');
for(const split of ['train','val'])if(sha(JSON.stringify(paired[split]))!==lock.paired[split])throw Error(`Video/3D pairing differs: ${split}`);
const manifest=json('h3wb.json');
if(manifest.train_count!==302262||manifest.val_count!==66317)throw Error('Unexpected 3D teacher count');
console.log('Teacher hashes, selected images, video alignment and split counts verified');
