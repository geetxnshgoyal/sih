import { playbackBounds, trackedHand } from '../src/lib/signGeometry.ts';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { textToGlosses } from '../src/lib/reverse.ts';
import { validateSignLibrary } from '../src/lib/signLibrary.ts';
import { SignSegmenter, segmentQuality } from '../src/lib/segment.ts';
import { StabilityGate } from '../src/lib/gate.ts';
import { UtteranceBuilder } from '../src/lib/sentence.ts';
import type { PointFrame } from '../src/lib/features.ts';
const library = validateSignLibrary(JSON.parse(readFileSync(new URL('../public/model/_signs.json', import.meta.url), 'utf8')));

test('all bundled playback clips contain complete finite frames and distinct animation data', () => {
  const clips = Object.values(library).map(x => JSON.stringify(x));
  assert.equal(new Set(clips).size, clips.length);
  assert.throws(() => validateSignLibrary({ Hello: [] }));
});
test('longest phrase matching consumes each word once and keeps intentional repeats', () => {
  assert.deepEqual(textToGlosses('How are you? Hello Hello. Thank you', library), {
    matched: ['How are you','Hello','Hello','Thank you'], skipped: [],
  });
});
test('aliases support complete phrases, case-insensitive targets, and Indic combining marks', () => {
  assert.deepEqual(textToGlosses('नमस्ते thank you very much', library, {'thank you very much':'THANK YOU'}), {matched:['Hello','Thank you'], skipped:[]});
});
test('unsupported signs remain visible as text and never become a different concept', () => {
  assert.deepEqual(textToGlosses('unknownword missingword', library), { matched: [], skipped:['unknownword','missingword'] });
  assert.deepEqual(textToGlosses('test', library, {test:'Not a real sign'}), {matched:[],skipped:['test']});
});
function frame(shift=0): PointFrame {
  const f = Array.from({length:65}, (_,i) => ({x:.45+(i%5)*.008,y:.3+(i%7)*.01,z:.001*i}));
  f[11]={x:.4,y:.3,z:0}; f[12]={x:.6,y:.3,z:0};
  for(let i=23;i<65;i++) f[i].x += shift;
  return f;
}
for(const fps of [10,15,30,60]) test(`one movement produces one segment at ${fps} unique FPS`, () => {
  const segmenter = new SignSegmenter(); const segments: PointFrame[][]=[];
  for(let tick=0;tick<=fps*3;tick++) {
    const t=tick*1000/fps;
    const shift=Math.min(.15, Math.max(0,(t-500)/900*.15));
    const result=segmenter.push(frame(shift), true, t);
    if(result) segments.push(result);
  }
  assert.equal(segments.length,1); assert.equal(segmentQuality(segments[0]),null);
});
test('a held pose and duplicate capture timestamps produce no detections', () => {
  const s=new SignSegmenter();
  for(let t=0;t<3000;t+=33) {assert.equal(s.push(frame(),true,t),null);assert.equal(s.push(frame(.4),true,t),null);}
});
test('missing hands, shoulders and invalid frames cannot become confident labels', () => {
  const noHand=frame();for(let i=23;i<44;i++)noHand[i]={x:0,y:0,z:0};
  assert.equal(segmentQuality(Array(20).fill(noHand)),'one-hand');
  const noPose=frame(); noPose[11]={x:0,y:0,z:0};noPose[12]={x:0,y:0,z:0};
  assert.equal(segmentQuality(Array(20).fill(noPose)),'no-pose');
});
test('an unbroken gesture times out without firing repeatedly', () => {
  const s=new SignSegmenter();let count=0;let timedOut=false;
  for(let t=0;t<15000;t+=33) {if(s.push(frame((t/33)%2 ? .3:0),true,t))count++; if(s.lastReject==='too-long') timedOut=true;}
  assert.equal(count,0);assert.equal(timedOut,true);
});
test('cooldown rejects duplicate detections but preserves different and later repeated signs', () => {
  const g=new StabilityGate();assert.equal(g.once({gloss:'Hello',conf:.99},0).fire,'Hello');
  assert.equal(g.once({gloss:'Hello',conf:.99},500).fire,null);
  assert.equal(g.once({gloss:'Doctor',conf:.99},600).fire,'Doctor');
  assert.equal(g.once({gloss:'Hello',conf:.99},700).fire,'Hello');
  assert.equal(g.once({gloss:'Hello',conf:.99},2600).fire,'Hello');
});
test('explicit flush delivers the last utterance exactly once', () => {
  const utterance=new UtteranceBuilder();utterance.add('Doctor',1000);
  assert.deepEqual(utterance.flush()?.glosses,['Doctor']);assert.equal(utterance.flush(),null);
});
test('one held gesture cannot duplicate a word inside an utterance', () => {
  const utterance=new UtteranceBuilder();
  utterance.add('healthy',1000,.91);utterance.add('Healthy',2900,.88);
  assert.deepEqual(utterance.flush()?.glosses,['healthy']);
});

test('anchored missing hands never shrink the player viewport or draw a collapsed hand', () => {
  const f = Array.from({length:65},(_,i)=>[i/100,i/100,0]);
  for(let i=23;i<44;i++) f[i]=[-50,-50,0];
  assert.equal(trackedHand(f,23),false);
  assert.ok(playbackBounds([f])!.minX >= 0);
});
