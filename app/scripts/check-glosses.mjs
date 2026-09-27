/**
 * Every gloss a domain can play must have a recording.
 *
 * domains.ts writes glosses as bare strings, so a typo or a word that only
 * exists in labels.json (the 38 trained classes, a different vocabulary from
 * the 179 playback recordings) fails silently: SignPlayback just lists it
 * under "Text only" and the phrase plays short. That is how ["I","Deaf","Sign"]
 * shipped broken, and nothing caught it because nothing was looking.
 *
 * Run in CI and before a build.
 */
import { readFileSync } from "node:fs";

const signs = new Set(
  Object.keys(JSON.parse(readFileSync("public/model/_signs.json", "utf8")))
);
const src = readFileSync("src/lib/domains.ts", "utf8");

const bad = [];

for (const m of src.matchAll(/glosses:\s*\[([^\]]*)\]/g)) {
  for (const g of [...m[1].matchAll(/"([^"]+)"/g)].map((x) => x[1])) {
    if (!signs.has(g)) bad.push(`gloss ${JSON.stringify(g)}`);
  }
}

for (const block of src.matchAll(/synonyms:\s*\{([\s\S]*?)\n {4}\},/g)) {
  for (const m of block[1].matchAll(/:\s*"([^"]+)"/g)) {
    if (!signs.has(m[1])) bad.push(`synonym target ${JSON.stringify(m[1])}`);
  }
}

if (bad.length) {
  const uniq = [...new Set(bad)].sort();
  console.error(`check-glosses: ${uniq.length} reference(s) with no recording in _signs.json\n`);
  for (const b of uniq) console.error(`  ${b}`);
  console.error(`\nUse a key of public/model/_signs.json, or record the sign.`);
  process.exit(1);
}

console.log("check-glosses: every domain gloss and synonym target has a recording");
