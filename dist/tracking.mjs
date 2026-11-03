export const HOLD_MS=300;
export const FORGET_MS=1200;
export function iou(a,b){
 const w=Math.max(0,Math.min(a[0]+a[2],b[0]+b[2])-Math.max(a[0],b[0]));
 const h=Math.max(0,Math.min(a[1]+a[3],b[1]+b[3])-Math.max(a[1],b[1]));
 const area=w*h;return area/(a[2]*a[3]+b[2]*b[3]-area||1);
}
export function matchDetections(tracks,detections,now){
 const candidates=[];
 for(let d=0;d<detections.length;d++)for(const t of tracks){
  if(now-t.lastSeen>FORGET_MS)continue;
  const a=t.box,b=detections[d].bbox,overlap=iou(a,b);
  const distance=Math.hypot(a[0]+a[2]/2-b[0]-b[2]/2,a[1]+a[3]/2-b[1]-b[3]/2)/Math.max(1,Math.hypot(a[2],a[3]));
  const scale=Math.sqrt(b[2]*b[3]/Math.max(1,a[2]*a[3]));
  if(scale>.5&&scale<2&&(overlap>.08||distance<.35))candidates.push({d,t,score:overlap+.15*(1-distance)});
 }
 candidates.sort((a,b)=>b.score-a.score);
 const assigned=new Map(),used=new Set();
 for(const c of candidates)if(!assigned.has(c.d)&&!used.has(c.t)){assigned.set(c.d,c.t);used.add(c.t);}
 return detections.map((d,i)=>({d,t:assigned.get(i)||null}));
}
export function trackVisibility(track,now){
 const age=now-track.lastSeen;
 return {held:age>0,visible:!!track.lastPose&&age<=HOLD_MS,expired:age>FORGET_MS};
}
// Compute asynchronously into a batch; presentation commits exactly once, only
// when every person is ready and the originating video/seek is still current.
export async function collectFrame(items,compute,isCurrent,commit){
 const result=[];
 for(const item of items){result.push(await compute(item));if(!isCurrent())return false;}
 if(!isCurrent())return false;
 commit(result);return true;
}
