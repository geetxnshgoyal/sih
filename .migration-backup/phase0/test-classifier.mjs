/* Regression test for classify(), no camera, no browser.
   Extracts the geometry + classifier straight out of index.html so the
   test can never drift from the shipped code.

   Run:  node phase0/test-classifier.mjs
*/
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const src  = readFileSync(join(here, "index.html"), "utf8");
const body = src.slice(
  src.indexOf("const d = (a,b)"),
  src.indexOf("/* ================================================================\n   3 ·")
);
const { classify } = await import(
  "data:text/javascript," + encodeURIComponent(body + "\nexport { classify };")
);

/* ---- synthetic hands, MediaPipe topology, y grows downward ---- */
const mk = p => p.map(([x, y]) => ({ x, y, z: 0 }));
const finger = (x, ext, yW) => {
  const m = yW - 0.20;
  return ext ? [[x,m],[x,m-0.09],[x,m-0.16],[x,m-0.22]]
             : [[x,m],[x,m-0.07],[x,m-0.04],[x,m+0.02]];
};
function hand({ open=true, thumbUp=false, fist=false, xc=0.5, yW=0.90, spread=0.055 } = {}){
  const thUp  = [[xc-0.05,yW-0.05],[xc-0.07,yW-0.12],[xc-0.08,yW-0.22],[xc-0.08,yW-0.32]];
  const thOut = [[xc-0.06,yW-0.04],[xc-0.11,yW-0.07],[xc-0.14,yW-0.09],[xc-0.16,yW-0.10]];
  const thIn  = [[xc-0.05,yW-0.05],[xc-0.06,yW-0.10],[xc-0.04,yW-0.14],[xc-0.02,yW-0.16]];
  const th    = thumbUp ? thUp : (fist || !open ? thIn : thOut);
  const cols  = [xc-1.5*spread, xc-0.5*spread, xc+0.5*spread, xc+1.5*spread];
  return mk([[xc,yW], ...th, ...cols.flatMap(x => finger(x, open, yW))]);
}

const FLOOR = 0.75;               // must match the gate in index.html
const cases = [
  ["one open palm",      [hand({open:true})],                                  "HELLO",     true ],
  ["thumbs up",          [hand({open:false, thumbUp:true})],                   "HELP",      true ],
  ["two palms together", [hand({open:true,xc:0.44,spread:0.030}),
                          hand({open:true,xc:0.56,spread:0.030})],             "THANK_YOU", true ],
  ["plain fist",         [hand({open:false, fist:true})],                       null,       false],
  ["no hands",           [],                                                    null,       false],
];

let pass = 0;
for (const [name, hands, want, shouldFire] of cases){
  const { gloss, conf } = classify(hands);
  const fires = gloss !== null && conf >= FLOOR;
  const ok = shouldFire ? (gloss === want && fires) : !fires;
  if (ok) pass++;
  console.log(
    `${ok ? "PASS" : "FAIL"}  ${name.padEnd(20)} -> ${String(gloss).padEnd(10)} ` +
    `conf=${conf.toFixed(2)} ${fires ? "FIRES" : "silent"}`
  );
}
console.log(`\n${pass}/${cases.length} passed`);
process.exit(pass === cases.length ? 0 : 1);
