import { playbackBounds, trackedHand } from '../src/lib/signGeometry.ts';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { textToGlosses } from '../src/lib/reverse.ts';
import { validateSignLibrary } from '../src/lib/signLibrary.ts';
import { parseRecordings } from '../src/lib/modelChecks.ts';
import { FACE_SUBSET, selectFace } from '../src/lib/face.ts';
import { SignSegmenter, segmentQuality, noticeForReject } from '../src/lib/segment.ts';
import { StabilityGate } from '../src/lib/gate.ts';
import { UtteranceBuilder } from '../src/lib/sentence.ts';
import type { PointFrame } from '../src/lib/features.ts';
const library = validateSignLibrary(JSON.parse(readFileSync(new URL('../public/model/_signs.json', import.meta.url), 'utf8')));

test('all bundled playback clips contain complete finite frames and distinct animation data', () => {
  const clips = Object.values(library).map(x => JSON.stringify(x));
  assert.equal(new Set(clips).size, clips.length);
  assert.throws(() => validateSignLibrary({ Hello: [] }));
});
test('sharded playback index covers every legacy sign exactly once', () => {
  const index = JSON.parse(readFileSync(new URL('../public/signs/index.json', import.meta.url), 'utf8'));
  assert.equal(index.version, 1);
  assert.equal(index.count, Object.keys(library).length);
  assert.deepEqual(new Set(Object.keys(index.glosses)), new Set(Object.keys(library)));
  assert.ok(index.shards.length > 1);
});
test('v3 camera recordings retain body compatibility and the curated face subset', () => {
  const face = Array.from({length:468}, (_, i) => ({x:i/1000,y:i/900,z:i/800}));
  assert.equal(selectFace(face)?.length, FACE_SUBSET.length);
  const body = Array.from({length:4}, () => frame().map(p => [p.x,p.y,p.z]));
  assert.deepEqual(parseRecordings({format:'setu-recordings-v3',takes:[{
    gloss:'please', body, face:Array.from({length:4},()=>[]), faceAvailable:true,
  }]}), [{true:'please',frames:body}]);
});
test('please remains a confirmable dictionary sign until it has enough training data', () => {
  const labels = JSON.parse(readFileSync(new URL('../public/model/labels.json', import.meta.url), 'utf8'));
  const bank = JSON.parse(readFileSync(new URL('../public/model/_bank.json', import.meta.url), 'utf8'));
  const index = bank.words.indexOf('please');
  assert.equal(labels.some((label:string) => label.toLowerCase() === 'please'), false);
  assert.ok(index >= 0);
  assert.ok(bank.refs[index] < 10);
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
test('every rejection reason says something, and a real reason is never silently overwritten by "hands are visible"', () => {
  // no-pose reachable by leaning in close: hands stay visible while shoulders
  // leave the frame. Before noticeForReject existed, SignBridge's if/else
  // chain had no branch for it, so hasHands === true fell into the final
  // `else if (hasHands) notify(null)` and erased the one useful signal.
  for (const reject of ['no-hands', 'one-hand', 'no-pose', 'too-long', 'too-short'] as const) {
    const withHands = noticeForReject(reject, true);
    const withoutHands = noticeForReject(reject, false);
    assert.notEqual(withHands, 'unchanged', `${reject} with hands visible must say something`);
    assert.notEqual(withoutHands, 'unchanged', `${reject} without hands visible must say something`);
    assert.ok(typeof withHands === 'object' && withHands.message, `${reject} message must be non-empty`);
  }
  // No rejection, hands visible: clear whatever was showing.
  assert.deepEqual(noticeForReject(null, true), { message: null, holdMs: 0 });
  // No rejection, no hands yet: this frame is not news, leave the last notice.
  assert.equal(noticeForReject(null, false), 'unchanged');
});
test('missing hands, shoulders and invalid frames cannot become confident labels', () => {
  const noHand=frame();for(let i=23;i<44;i++)noHand[i]={x:0,y:0,z:0};
  assert.equal(segmentQuality(Array(20).fill(noHand)),'one-hand');
  const noPose=frame(); noPose[11]={x:0,y:0,z:0};noPose[12]={x:0,y:0,z:0};
  assert.equal(segmentQuality(Array(20).fill(noPose)),'no-pose');
});
/** Rest as a camera delivers it: still, but never twice the same.
 *
 *  Real Holistic rest, measured on motionless stretches of islgov clips, runs
 *  p50 0.0025 / p90 0.0085 / p95 0.0128 against STOP 0.006 -- so a frozen frame
 *  (energy exactly 0) is the one kind of stillness a camera NEVER produces, and
 *  it was the only kind these tests used. jit .0028 puts this at 0.011, inside
 *  the real p90-p95 band: plainly a person holding still, and plainly above the
 *  absolute floor. */
function restFrame(k: number, jit = .0028): PointFrame {
  const f = frame();
  for (let i = 23; i < 65; i++) { f[i].x += jit * Math.sin(k * 1.7 + i); f[i].y += jit * Math.cos(k * 2.3 + i); }
  return f;
}
/** A sign with the back-and-forth energy of a real one (median ~0.06, against
 *  a measured real-clip median of 0.055), not a slow one-way drift. */
const signFrame = (k: number) => frame(.05 * Math.sin(k * .5));

test('a sign ending in real jittery rest still completes, and is not held until it times out', () => {
  const s = new SignSegmenter(); const segments: PointFrame[][] = []; let i = 0;
  const t = () => i * 1000 / 15;
  for (let k = 0; k < 30; k++, i++) { const r = s.push(signFrame(k), true, t()); if (r) segments.push(r); }
  for (let k = 0; k < 45; k++, i++) { const r = s.push(restFrame(k), true, t()); if (r) segments.push(r); }
  assert.equal(segments.length, 1);
  assert.equal(segmentQuality(segments[0]), null);
  assert.notEqual(s.lastReject, 'too-long');
});
test('the settling cooldown always ends, so one over-long sign cannot kill the camera', () => {
  const s = new SignSegmenter(); let i = 0;
  const t = () => i * 1000 / 15;
  for (let k = 0; k < 15 * 12; k++, i++) s.push(signFrame(k), true, t());
  assert.equal(s.current, 'settling');        // the sign was discarded as too-long
  for (let k = 0; k < 15 * 4; k++, i++) s.push(restFrame(k), true, t());
  assert.equal(s.current, 'idle');            // and the cooldown released it
  const segments: PointFrame[][] = [];        // so the NEXT sign is readable
  for (let k = 0; k < 30; k++, i++) { const r = s.push(signFrame(k), true, t()); if (r) segments.push(r); }
  for (let k = 0; k < 45; k++, i++) { const r = s.push(restFrame(k), true, t()); if (r) segments.push(r); }
  assert.equal(segments.length, 1);
});
test('an unbroken gesture times out without firing repeatedly', () => {
  const s=new SignSegmenter();let count=0;let timedOut=false;
  for(let t=0;t<15000;t+=33) {if(s.push(frame((t/33)%2 ? .3:0),true,t))count++; if(s.lastReject==='too-long') timedOut=true;}
  assert.equal(count,0);assert.equal(timedOut,true);
});
test('manual finish preserves a moving face-adjacent sign without waiting for quiet', () => {
  const s = new SignSegmenter();
  for(let t=0;t<1200;t+=67) {
    const f=frame(Math.max(0,(t-200)/1000*.12));
    for(let i=23;i<44;i++) f[i].y=.2+t/1200*.16+((i%2)*.004);
    s.push(f,true,t);
  }
  const result=s.flush();
  assert.ok(result && result.length >= 4);
  assert.equal(segmentQuality(result),null);
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
