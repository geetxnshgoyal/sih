import { test } from "node:test";
import assert from "node:assert/strict";
import { mirrorAnchored } from "../src/lib/mirror.ts";
import { FEATURE_SIZE, N_DIMS, N_POINTS, POSE_KEEP } from "../src/lib/features.ts";

function random(seed = 7): Float64Array {
  const v = new Float64Array(FEATURE_SIZE);
  let s = seed;
  for (let i = 0; i < v.length; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; v[i] = s / 0x7fffffff - 0.5; }
  return v;
}

test("mirror is an involution", () => {
  const v = random();
  const back = mirrorAnchored(mirrorAnchored(v));
  for (let i = 0; i < v.length; i++) assert.equal(back[i], v[i]);
});

test("mirror swaps the hands, negates x, keeps y and z", () => {
  const v = random(3);
  const m = mirrorAnchored(v);
  const per = N_POINTS * N_DIMS;
  const L = POSE_KEEP, R = POSE_KEEP + 21;
  for (let t = 0; t < 2; t++) for (let k = 0; k < 21; k++) {
    const l = t * per + (L + k) * 3, r = t * per + (R + k) * 3;
    assert.equal(m[l], -v[r]); assert.equal(m[l + 1], v[r + 1]); assert.equal(m[l + 2], v[r + 2]);
    assert.equal(m[r], -v[l]);
  }
  // nose (0) has no pair: only x flips
  assert.equal(m[0], -v[0]); assert.equal(m[1], v[1]); assert.equal(m[2], v[2]);
  // shoulders (11,12) swap
  assert.equal(m[11 * 3], -v[12 * 3]); assert.equal(m[12 * 3 + 1], v[11 * 3 + 1]);
});
