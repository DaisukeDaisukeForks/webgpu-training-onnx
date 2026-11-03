// Draw after geometry transforms; covered ground-truth joints retain their targets.
export function occlusionPlan(random){
  if(random()>.65)return[];
  return Array.from({length:1+Math.floor(random()*3)},()=>{
    const w=.08+random()*.28,h=.08+random()*.28,x=random()*(1-w),y=random()*(1-h),tile=.02+random()*.04,colors=[];
    for(let j=0;j<Math.ceil(h/tile);j++)for(let i=0;i<Math.ceil(w/tile);i++)colors.push([random(),random(),random()].map(v=>Math.floor(v*256)));
    return{x,y,w,h,tile,colors};
  });
}
export function drawOcclusion(context,plan,width,height){
  context.save();context.globalAlpha=1;context.globalCompositeOperation='source-over';
  for(const p of plan){let k=0;for(let y=0;y<p.h;y+=p.tile)for(let x=0;x<p.w;x+=p.tile){context.fillStyle=`rgb(${p.colors[k++].join(',')})`;context.fillRect((p.x+x)*width,(p.y+y)*height,Math.min(p.tile,p.w-x)*width,Math.min(p.tile,p.h-y)*height);}}
  context.restore();
}
