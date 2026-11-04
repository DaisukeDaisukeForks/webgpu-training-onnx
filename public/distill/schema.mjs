// COCO-WholeBody order. Keep all six foot landmarks and five fingertips/hand.
export const IDS = [...Array.from({length: 23}, (_, i) => i), 95,96,99,103,107,108,111,116,117,120,124,128,129,132];
export const NAMES = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel', ...['left','right'].flatMap(side => ['thumb_tip','index_mcp','index_tip','middle_tip','ring_tip','little_mcp','little_tip'].map(n => `${side}_${n}`))];
export const K = IDS.length, BINS = [192,256,192], Z_RANGE = 2.1744869;
export const FLIP = IDS.map(i => IDS.indexOf(i < 17 ? [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15][i] : i < 23 ? i < 20 ? i+3 : i-3 : i < 112 ? i+21 : i-21));
export const EDGES = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[15,19],[19,17],[19,18],[17,18],[16,22],[22,20],[22,21],[20,21]];
export const SPEC = {format:'rtmw3d-distill37-v1', joints:NAMES, coco133_ids:IDS,
  input:{name:'image',shape:[1,3,256,192],range:[0,1],crop:'bbox expanded 1.25; aspect 0.75; black padding'},
  output:{keypoints:[1,K,3],units:['192-wide crop pixels','256-high crop pixels','pelvis-relative metres'],scores:'XY distribution concentration; not visibility probability'},
  teacher:{repository:'https://github.com/open-mmlab/mmpose',revision:'759b39c13fea6ba094afc1fa932f51dc1b11cbf9',input:[1,3,384,288],mean:[123.675,116.28,103.53],std:[58.395,57.12,57.375],rawBins:[576,768,576]},
  architecture:'inverted residual CNN, spatial joint tokens + global image context, shared SimCC XYZ heads', bins:BINS,zRange:Z_RANGE};
