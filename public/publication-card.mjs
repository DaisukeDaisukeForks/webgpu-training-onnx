export function modelCard(spec){
 const video=String(spec.format||'').includes('video'),training=spec.training||{},data=spec.data||{};
 return`---
license: other
license_name: research-and-noncommercial-dataset-conditions
license_link: https://github.com/jin-s13/COCO-WholeBody#terms-of-use
tags:
- onnx
- pose-estimation
- vrchat
---
# WholeBody73 VRC
Independent model trained in browser WebGPU from seeded random weights. No pretrained model or teacher weights. Noncommercial research use; the source datasets retain their own conditions.

## Input and output
${video?'Three causal RGB frames, float32 [1,3,3,256,192], oldest to current, at 50fps.':'RGB image, float32 [1,3,256,192].'} Values are 0..1. Use the documented person crop. The 73 outputs use Unity axes (+X right,+Y up,+Z forward), meters. vrc_relative is hip-relative; vrc_positions applies the supplied tracking_rotation and tracking_origin. Absolute camera/world position is not inferred.

## Training and validation
Preparation updates: 2D ${training.updates2??'unknown'}, 3D ${training.updates3??'unknown'}. Video updates: ${spec.position?.updates??0}. The exact datasets, counts, source hashes, exclusions and metrics are in model-spec.json. Annotation-2D-to-3D metrics are separate from video-to-3D validation. No claim of production accuracy or commercial clearance is made.
${video?'The direct RGB/3D video supervision covers ASPset body12. Hand/finger/face/sole 3D video supervision has not been supplied.':'Teacherless points are masked, not fabricated.'}

## Credits and source conditions
- COCO-WholeBody: Jin et al., ECCV 2020 / SenseTime Research. Research/noncommercial annotations. https://github.com/jin-s13/COCO-WholeBody . Original COCO image conditions apply: https://cocodataset.org/#termsofuse .
- CMU Human Foot Keypoint Dataset: CMU Perceptual Computing Lab. CC BY 4.0 annotations, original COCO image licenses. https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/ . Toe/heel 2D labels; no sole contact labels. Pinned mirror provenance is included in model-spec.json.
- H3WB annotations: Copyright (c) 2022 H3WB 3D WholeBody Dataset Annotations. Yue Zhu, Nermin Samet, David Picard, ICCV 2023. MIT annotation license is included. https://github.com/wholebody3d/wholebody3d . Human3.6M RGB images are not included.
- ASPset-510: Aiden Nibali, Joshua Millward, Zhen He, Stuart Morgan. CC0-1.0. https://archive.org/details/aspset510 .
${data.aist?'- AIST++: Ruilong Li et al., Learning to Generate Diverse Dance Motions with Transformer, ICCV 2021. Google annotations: CC BY 4.0. https://google.github.io/aistplusplus_dataset/ . Reconstructed body coordinates and camera metadata only; AIST Dance DB videos are not included. This application is for noncommercial research.\n':''}
Modifications include landmark selection, cropping, seeded occlusion/scale augmentation, hip centering and Unity-axis conversion. AIST++ labels, when present, use matched per-sequence SMPL scale and the official pose split and exclusion list. Synthetic occluders do not create new ground truth. See model-spec.json for the actual evaluation state.
`;
}
