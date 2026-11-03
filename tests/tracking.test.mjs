import test from 'node:test';
import assert from 'node:assert/strict';
import {collectFrame,matchDetections,trackVisibility} from '../dist/tracking.mjs';

test('multi-person frame remains intact until every asynchronous prediction completes',async()=>{
 const pending=[];let displayed=['previous frame'],commits=0;
 const task=collectFrame([1,2],x=>new Promise(resolve=>pending.push(()=>resolve(x))),()=>true,r=>{displayed=r;commits++;});
 assert.deepEqual(displayed,['previous frame']);
 pending.shift()();await new Promise(resolve=>setImmediate(resolve));
 assert.deepEqual(displayed,['previous frame']);assert.equal(commits,0);
 pending.shift()();assert.equal(await task,true);
 assert.deepEqual(displayed,[1,2]);assert.equal(commits,1);
});

test('failed or invalidated frames never replace the last complete frame; empty detections do commit',async()=>{
 let commits=0,current=true;
 await assert.rejects(collectFrame([1,2],async x=>{if(x===2)throw Error('inference failed');return x;},()=>true,()=>commits++));
 assert.equal(commits,0);
 assert.equal(await collectFrame([1],async x=>{current=false;return x;},()=>current,()=>commits++),false);
 assert.equal(commits,0);
 await collectFrame([],async()=>{},()=>true,r=>{assert.deepEqual(r,[]);commits++;});
 assert.equal(commits,1);
});

test('short detection gaps keep IDs with a bounded visible hold, and expired people are removed',()=>{
 const a={id:1,box:[0,0,100,200],lastSeen:1000,lastPose:{}},b={id:2,box:[300,0,100,200],lastSeen:1000,lastPose:{}};
 const match=matchDetections([a,b],[{bbox:[310,0,100,200]},{bbox:[15,0,100,200]}],1200);
 assert.deepEqual(match.map(x=>x.t.id),[2,1]);
 assert.deepEqual(trackVisibility(a,1000),{held:false,visible:true,expired:false});
 assert.equal(trackVisibility(a,1299).visible,true);
 assert.equal(trackVisibility(a,1301).visible,false);
 assert.equal(trackVisibility(a,2201).expired,true);
 assert.equal(matchDetections([a],[{bbox:[800,0,100,200]}],1200)[0].t,null);
 assert.equal(matchDetections([a],[{bbox:a.box}],2201)[0].t,null);
});
