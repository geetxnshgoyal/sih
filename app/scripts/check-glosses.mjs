/** Validate the actual domain data, including shared and generated quick phrases. */
import { readFileSync } from "node:fs";
import { DOMAIN_LIST } from "../src/lib/domains.ts";

const signs = new Set(Object.keys(JSON.parse(readFileSync(
  new URL("../public/model/_signs.json", import.meta.url), "utf8",
))));
const errors = [];
if (!DOMAIN_LIST.length) errors.push("No service domains are configured");
for (const domain of DOMAIN_LIST) {
  const seen = new Set();
  for (const phrase of domain.quick) {
    const key = phrase.glosses.join("|");
    if (!key) errors.push(`${domain.id}: empty quick phrase`);
    if (seen.has(key)) errors.push(`${domain.id}: duplicate quick phrase ${key}`);
    seen.add(key);
    for (const gloss of phrase.glosses) {
      if (!signs.has(gloss)) errors.push(`${domain.id}: missing recording ${gloss}`);
    }
  }
  for (const gloss of Object.values(domain.synonyms)) {
    if (!signs.has(gloss)) errors.push(`${domain.id}: missing synonym recording ${gloss}`);
  }
  if (!domain.quick.some((phrase) => phrase.urgent)) {
    errors.push(`${domain.id}: no priority quick phrase`);
  }
}
if (errors.length) {
  console.error(errors.join("\n"));
  process.exit(1);
}
console.log(`Validated quick signs for ${DOMAIN_LIST.length} service settings.`);
