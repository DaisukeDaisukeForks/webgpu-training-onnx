// Fit the entire frame in the model input. No detection box or person crop.
export function frameTransform(width,height){
 if(!(width>0&&height>0))throw Error('動画のサイズが不正です');
 const scale=Math.min(192/width,256/height),w=width*scale,h=height*scale;
 const padX=(192-w)/2,padY=(256-h)/2;
 // Inverse affine map: normalized model xy -> original video pixels.
 return {scale,padX,padY,width:w,height:h,rect:[-padX/scale,-padY/scale,192/scale,256/scale]};
}
export function frameInput(rgba){
 const n=192*256,data=new Float32Array(n*3);
 for(let i=0;i<n;i++)for(let c=0;c<3;c++)data[c*n+i]=rgba[i*4+c]/255;
 return data;
}
