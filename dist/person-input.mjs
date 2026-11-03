export function personRect(box){
 const [left,top,width,height]=box;
 const h=Math.max(height,width/.75)*1.25,w=h*.75;
 return [left+width/2-w/2,top+height/2-h/2,w,h];
}
export function imageTensor(rgba){
 const n=192*256,data=new Float32Array(n*3);
 for(let i=0;i<n;i++)for(let c=0;c<3;c++)data[c*n+i]=rgba[i*4+c]/255;
 return data;
}
