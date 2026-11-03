import assert from 'node:assert/strict';
import {rng} from '../public/cache.mjs';
import {occlusionPlan,drawOcclusion} from '../public/augmentation.mjs';
assert.deepEqual(occlusionPlan(rng(771)),occlusionPlan(rng(771)));
assert.notDeepEqual(occlusionPlan(rng(771)),occlusionPlan(rng(772)));
let blocks=0;const c={save(){},restore(){},fillRect(x,y,w,h){assert.equal(this.globalAlpha,1);assert.equal(this.globalCompositeOperation,'source-over');assert(x>=0&&y>=0&&w>0&&h>0);assert(x+w<=192+1e-9&&y+h<=256+1e-9);assert.match(this.fillStyle,/^rgb\(\d+,\d+,\d+\)$/);blocks++;}};
for(let seed=0;seed<1000;seed++)drawOcclusion(c,occlusionPlan(rng(seed)),192,256);
assert(blocks>1000);
console.log('PASS: deterministic seed; opaque color blocks; in-bounds occlusion geometry.');
