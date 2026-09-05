/**
 * Validate the domain definitions against the trained label set.
 *
 *   node --experimental-strip-types app/tools/checkDomains.ts
 *
 * Why this is a script and not a comment
 * --------------------------------------
 * Every quick phrase and every synonym target in `domains.ts` names a gloss the
 * classifier is supposed to be able to produce. Nothing at runtime checks that.
 * A typo, or a retrain that drops a class, turns a demo button into a silent
 * no-op: it renders, it is tappable, and it plays nothing.
 *
 * `glossCorpus.ts` is already protected this way by the generator, which refuses
 * to run on an unknown gloss. This gives `domains.ts` the same guarantee, and
 * makes "does the Travel domain still work after a retrain?" a command rather
 * than a hope.
 *
 * Exits non-zero on any unknown gloss, so it can gate a build.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { DOMAIN_LIST } from "../src/lib/domains.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..", "..");
const LABELS = path.join(ROOT, "models", "labels.json");
const SIGNS = path.join(ROOT, "app", "public", "model", "_signs.json");

const labels: string[] = JSON.parse(fs.readFileSync(LABELS, "utf8"));
const known = new Set(labels);

// The reverse direction can only PLAY a gloss that has pose frames bundled.
// A synonym pointing at a recognisable-but-unplayable gloss is still a dead end
// in Conversation mode, so check that set separately rather than assuming.
let playable: Set<string> | null = null;
try {
  playable = new Set(Object.keys(JSON.parse(fs.readFileSync(SIGNS, "utf8"))));
} catch {
  console.log("note: _signs.json unreadable, skipping playability checks\n");
}

let errors = 0;
let warnings = 0;

for (const domain of DOMAIN_LIST) {
  console.log(`${domain.label}  (${domain.quick.length} quick phrases, ` +
    `${Object.keys(domain.synonyms).length} synonyms)`);

  for (const q of domain.quick) {
    const unknown = q.glosses.filter((g) => !known.has(g));
    if (unknown.length) {
      console.log(`  ERROR  quick "${q.caption}" -> unknown gloss: ${unknown.join(", ")}`);
      errors++;
      continue;
    }
    if (playable) {
      const unplayable = q.glosses.filter((g) => !playable!.has(g));
      if (unplayable.length) {
        console.log(`  warn   quick "${q.caption}" -> not playable in reverse: ${unplayable.join(", ")}`);
        warnings++;
      }
    }
  }

  for (const [word, target] of Object.entries(domain.synonyms)) {
    if (!known.has(target)) {
      console.log(`  ERROR  synonym "${word}" -> unknown gloss "${target}"`);
      errors++;
    } else if (playable && !playable.has(target)) {
      console.log(`  warn   synonym "${word}" -> "${target}" is recognisable but not playable`);
      warnings++;
    }
  }
  console.log();
}

console.log(`${errors} error(s), ${warnings} warning(s)`);
if (errors) {
  console.log("\nUnknown glosses cannot be produced by the classifier. Fix them or");
  console.log("remove the entry: a phrase naming a gloss that does not exist is a");
  console.log("button that silently does nothing.");
}
process.exit(errors ? 1 : 0);
