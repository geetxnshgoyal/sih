import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFileSync, readdirSync } from 'node:fs';
import { GlossClassifier } from '../src/lib/classifier.ts';
import { SignSegmenter } from '../src/lib/segment.ts';

test('How are you remains one complete sign across internal holds, never its healthy suffix', async () => {
  const root = new URL('../public/model/', import.meta.url);
  const allowed = new Set(readdirSync(root));
  const server = createServer((req, res) => {
    const name = (req.url ?? '').slice(1);
    if (!allowed.has(name)) { res.writeHead(404).end(); return; }
    res.end(readFileSync(new URL(name, root)));
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const base = `http://127.0.0.1:${address.port}`;
  const classifier = new GlossClassifier();
  try {
    await classifier.load(`${base}/model.json`, `${base}/labels.json`);
    const clips = JSON.parse(readFileSync(new URL('_demo.json', root), 'utf8'));
    const frames = clips.find((c: {true: string}) => c.true === 'How are you').frames
      .map((f: number[][]) => f.map(([x,y,z]) => ({x,y,z})));
    // These are raw, full-depth unit-coordinate clips, not playback assets.
    // Test every common camera rate and multiple locations of a 500ms hold.
    for (const fps of [10, 15, 30, 60]) for (const pauseAt of [30, 40, 50, 60]) {
      const padded = [...Array(30).fill(frames[0]), ...frames.slice(0,pauseAt),
        ...Array(15).fill(frames[pauseAt-1]), ...frames.slice(pauseAt),
        ...Array(60).fill(frames.at(-1))];
      const segmenter = new SignSegmenter();
      const segments = [];
      for (let t=0; t < padded.length/30*1000; t+=1000/fps) {
        const index = Math.min(padded.length-1, Math.floor(t/1000*30));
        const segment = segmenter.push(padded[index], true, t);
        if (segment) segments.push(segment);
      }
      assert.equal(segments.length, 1, `one complete sign at ${fps} FPS, hold at ${pauseAt}`);
      const top = classifier.predictTop(segments[0], 16/9, 1)[0];
      assert.equal(top.gloss, 'How are you', `${fps} FPS, hold at ${pauseAt}: ${JSON.stringify(top)}`);
    }
  } finally {
    classifier.dispose();
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  }
});
