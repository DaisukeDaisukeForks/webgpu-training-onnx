import {Dataset} from '../cache.mjs?revision=8';
import {teachers} from '../teacher-cache.mjs?revision=17';
import {IDS as OLD_IDS} from '../schema.mjs';
import {IDS,K,BINS,FLIP} from './schema.mjs';
import {unpackAxis} from './teacher.mjs';
export function personCrop(image,box,width=288,height=384){
  const [x,y,w,h]=box;if(![x,y,w,h].every(Number.isFinite)||w<=0||h<=0)throw Error('人物枠が不正です');
  let ew=w*1.25,eh=h*1.25;if(ew/eh>.75)eh=ew/.75;else ew=eh*.75;
  const left=x+w/2-ew/2,top=y+h/2-eh/2,canvas=new OffscreenCanvas(width,height),c=canvas.getContext('2d',{willReadFrequently:true});
  c.fillStyle='#000';c.fillRect(0,0,width,height);c.setTransform(width/ew,0,0,height/eh,-left*width/ew,-top*height/eh);c.drawImage(image,0,0);c.resetTransform();
  return{canvas,rect:[left,top,ew,eh]};
}
export function realLabels(row,rect){const xy=new Float32Array(K*2),mask=new Float32Array(K);for(let j=0;j<K;j++){const i=OLD_IDS.indexOf(IDS[j]),p=i<0?null:row.points[i];if(!p)continue;const x=(p[0]-rect[0])/rect[2],y=(p[1]-rect[1])/rect[3];xy[j*2]=x;xy[j*2+1]=y;mask[j]=p[2]>0&&x>=0&&x<1&&y>=0&&y<1?1:0;}return{gtXY:xy,gtMask:mask};}
export async function sourceRows(split){const files=split==='train'?['coco-train.json','coco-extra-train-0.json','coco-extra-train-1.json','cmu-foot-train.json']:['coco-val.json','cmu-foot-val.json'];const out=[];for(const name of files){const m=await teachers.json(name);out.push(...m.rows.map(r=>({...r,origin:name})));}return out;}
export async function imageSource(log){const ds=new Dataset(log);await ds.init();return ds;}
export async function batch(tf,records,random,augment=true){
  const n=records.length,images=new Float32Array(n*256*192*3),position=new Float32Array(n*K*3),mask=new Float32Array(n*K),gtXY=new Float32Array(n*K*2),gtMask=new Float32Array(n*K),axes=BINS.map(b=>new Float32Array(n*K*b));
  for(let s=0;s<n;s++){
    const r=records[s],flip=augment&&random()<.5,canvas=new OffscreenCanvas(192,256),c=canvas.getContext('2d',{willReadFrequently:true}),im=await createImageBitmap(r.image);
    if(flip){c.translate(192,0);c.scale(-1,1);}c.drawImage(im,0,0,192,256);im.close();c.resetTransform();
    if(augment&&random()<.2){const w=12+random()*24,h=16+random()*32;c.fillStyle=`rgb(${Math.floor(random()*255)} ${Math.floor(random()*255)} ${Math.floor(random()*255)})`;c.fillRect(random()*(192-w),random()*(256-h),w,h);}
    const rgba=c.getImageData(0,0,192,256).data,gain=augment?.85+random()*.3:1;for(let i=0;i<192*256;i++)for(let ch=0;ch<3;ch++)images[(s*192*256+i)*3+ch]=Math.min(1,rgba[i*4+ch]/255*gain);
    const dense=r.axes.map((v,a)=>unpackAxis(v,BINS[a]));
    for(let j=0;j<K;j++){
      const from=flip?FLIP[j]:j;mask[s*K+j]=r.mask[from];gtMask[s*K+j]=r.gtMask?.[from]||0;
      for(let a=0;a<3;a++){const v=r.position[from*3+a];position[(s*K+j)*3+a]=flip&&a===0?1-v:v;const b=BINS[a];for(let i=0;i<b;i++)axes[a][(s*K+j)*b+i]=dense[a][from*b+(flip&&a===0?b-1-i:i)];}
      for(let a=0;a<2;a++){const v=r.gtXY?.[from*2+a]||0;gtXY[(s*K+j)*2+a]=flip&&a===0?1-v:v;}
    }
  }
  return{image:tf.tensor4d(images,[n,256,192,3]),axes:axes.map((v,a)=>tf.tensor3d(v,[n,K,BINS[a]])),position:tf.tensor3d(position,[n,K,3]),mask:tf.tensor2d(mask,[n,K]),gtXY:tf.tensor3d(gtXY,[n,K,2]),gtMask:tf.tensor2d(gtMask,[n,K])};
}
export function disposeBatch(b){b.axes.forEach(t=>t.dispose());for(const[k,v]of Object.entries(b))if(k!=='axes')v.dispose();}
