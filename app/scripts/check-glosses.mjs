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
    if (!signs.has(g)) bad.push(["missing", `gloss ${JSON.stringify(g)}`]);
  }
}

for (const block of src.matchAll(/synonyms:\s*\{([\s\S]*?)\n {4}\},/g)) {
  for (const m of block[1].matchAll(/:\s*"([^"]+)"/g)) {
    if (!signs.has(m[1])) bad.push(["missing", `synonym target ${JSON.stringify(m[1])}`]);
  }
}

// Two quick phrases in one domain must not share a gloss sequence.
//
// QuickPhrases keys its buttons on glosses.join("|"), so a repeat is a
// duplicate React key. It is also a real defect on screen: two buttons that
// play the identical sign, which tells a Deaf user they are saying two
// different things when they are saying one. Spreading SHARED_QUICK and
// COUNTER_QUICK into a domain that already has the same phrase is the easy
// way to cause it, and that is exactly how bank and cinema acquired three.
const block = (name) => {
  const i = src.indexOf(`const ${name}`);
  return i === -1 ? [] : glossesIn(src.slice(i, src.indexOf("];", i)));
};
const glossesIn = (text) =>
  [...text.matchAll(/glosses:\s*\[([^\]]*)\]/g)]
    .map((m) => (m[1].match(/"[^"]+"/g) || []).join("|"));

const SHARED = block("SHARED_QUICK");
const COUNTER = block("COUNTER_QUICK");

for (const m of src.matchAll(/^  (\w+): \{\n    id: "(\w+)"/gm)) {
  const seg = src.slice(m.index, src.indexOf("\n  },", m.index));
  let list = glossesIn(seg);
  if (/\.\.\.SHARED_QUICK/.test(seg)) list = [...SHARED, ...list];
  if (/\.\.\.COUNTER_QUICK/.test(seg)) list = [...list, ...COUNTER];
  const seen = new Map();
  for (const g of list) seen.set(g, (seen.get(g) || 0) + 1);
  for (const [g, n] of seen) {
    if (n > 1) bad.push(["dupe", `domain ${m[2]}: ${n} quick phrases share the glosses [${g.replace(/"/g, "")}]`]);
  }
}

const sliceOf = (name) => {
  const i = src.indexOf(`const ${name}`);
  return i === -1 ? "" : src.slice(i, src.indexOf("];", i));
};
const sharedSrc = sliceOf("SHARED_QUICK");
const counterSrc = sliceOf("COUNTER_QUICK");

// Every setting needs at least one priority quick phrase.
//
// QuickPhrases only renders the "Priority needs" group when something is
// urgent, so a domain with none silently loses the whole strip. Retail
// shipped that way: its Deaf user got one flat alphabetical-ish list with
// nothing surfaced, while every other counter had a priority group.
for (const m of src.matchAll(/^  (\w+): \{\n    id: "(\w+)"/gm)) {
  const seg = src.slice(m.index, src.indexOf("\n  },", m.index));
  const urgentIn = (t) => /urgent:\s*true/.test(t);
  const shared = /\.\.\.SHARED_QUICK/.test(seg) && urgentIn(sharedSrc);
  const counter = /\.\.\.COUNTER_QUICK/.test(seg) && urgentIn(counterSrc);
  if (!urgentIn(seg) && !shared && !counter) {
    bad.push(["priority", `domain ${m[2]}: no quick phrase marked urgent, so it renders no priority group`]);
  }
}

if (bad.length) {
  const of = (kind) => [...new Set(bad.filter(([k]) => k === kind).map(([, m]) => m))].sort();
  const missing = of("missing");
  const dupes = of("dupe");
  const priority = of("priority");

  if (missing.length) {
    console.error(`check-glosses: ${missing.length} reference(s) with no recording in _signs.json\n`);
    for (const b of missing) console.error(`  ${b}`);
    console.error(`\n  Fix: use a key of public/model/_signs.json, or record the sign.\n`);
  }
  if (dupes.length) {
    console.error(`check-glosses: ${dupes.length} duplicate quick-phrase gloss sequence(s)\n`);
    for (const b of dupes) console.error(`  ${b}`);
    console.error(`\n  Two buttons that play the identical sign tell a Deaf user they are`);
    console.error(`  saying different things when they are saying one, and collide on the`);
    console.error(`  React key. Fix: give one of them distinct glosses, or drop it if a`);
    console.error(`  spread of SHARED_QUICK or COUNTER_QUICK already supplies it.\n`);
  }
  if (priority.length) {
    console.error(`check-glosses: ${priority.length} setting(s) with no priority quick phrase\n`);
    for (const b of priority) console.error(`  ${b}`);
    console.error(`\n  QuickPhrases only renders "Priority needs" when something is urgent,`);
    console.error(`  so the whole strip disappears and nothing is surfaced. Fix: mark the`);
    console.error(`  phrase that matters most at that counter urgent: true.\n`);
  }
  process.exit(1);
}
