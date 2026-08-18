/**
 * Offline gloss reordering — ISL gloss order -> natural spoken sentence.
 *
 *   ANTHROPIC_API_KEY=... node --experimental-strip-types app/tools/buildGlossTable.ts
 *
 * Why this runs offline and not in the browser
 * --------------------------------------------
 * Two independent reasons, both hard requirements:
 *
 *  1. A browser call would ship the API key to every viewer. There is no safe
 *     way to hold an Anthropic key in client-side code.
 *  2. The stage demo must never depend on a live network call. `speech.ts`
 *     already states this policy for Bhashini and it applies with equal force
 *     here — a 500 ms round trip is fine while a user finishes signing, but a
 *     timeout in front of judges is not recoverable.
 *
 * So translation happens here, once, and ships as a static table the app reads
 * with zero network. Re-running is cheap: entries already present are skipped,
 * so this is resumable and incremental as the corpus grows.
 *
 * What the model is actually asked to do
 * --------------------------------------
 * Not word substitution — that is what `reverse.ts` already does and correctly
 * refuses to call translation. Here the model is given the gloss sequence in
 * SIGNED order plus the domain, and asked to produce the sentence a fluent
 * speaker of each language would actually say. "STATION GO WHERE" becomes
 * "स्टेशन कहाँ है", not "स्टेशन जाना कहाँ".
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";

import { GLOSS_CORPUS, type GlossSeq } from "./glossCorpus.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP = path.resolve(HERE, "..");
const ROOT = path.resolve(APP, "..");

const LABELS_PATH = path.join(ROOT, "models", "labels.json");
const OUT_PATH = path.join(APP, "public", "model", "_utterances.json");

const LANGS = ["hi-IN", "ta-IN", "te-IN", "bn-IN", "mr-IN", "en-IN"] as const;
type Lang = (typeof LANGS)[number];

const LANG_NAMES: Record<Lang, string> = {
  "hi-IN": "Hindi (Devanagari script)",
  "ta-IN": "Tamil (Tamil script)",
  "te-IN": "Telugu (Telugu script)",
  "bn-IN": "Bengali (Bengali script)",
  "mr-IN": "Marathi (Devanagari script)",
  "en-IN": "Indian English",
};

/** How many sequences go into one request. Keeps requests small enough to stay reliable. */
const BATCH_SIZE = 8;

const RenderingSchema = z.object({
  key: z.string().describe("the gloss key exactly as given, pipe-separated"),
  "hi-IN": z.string(),
  "ta-IN": z.string(),
  "te-IN": z.string(),
  "bn-IN": z.string(),
  "mr-IN": z.string(),
  "en-IN": z.string(),
  reordered: z
    .boolean()
    .describe("true if the spoken order differs from the signed gloss order"),
  register: z
    .enum(["statement", "question", "request"])
    .describe("what the utterance actually does"),
});

const BatchSchema = z.object({
  renderings: z.array(RenderingSchema),
});

const SYSTEM = `You render Indian Sign Language gloss sequences into natural spoken sentences.

You are given signs in the order a Deaf signer produced them. ISL word order is not
spoken word order: ISL is topic-comment, drops copulas and articles, places
question markers at the end, and has no separate word for many function words.

Your job is to write what a fluent speaker of each target language would ACTUALLY
SAY to mean the same thing — not a word-by-word substitution of the glosses.

Rules:
- Produce ONE natural sentence per language. No alternatives, no notes, no glosses.
- Preserve the speech act. A gloss sequence ending in a question marker, or one
  whose note says it is a request, must come out as a question or request — not a
  flat statement. This is the single most common failure: rendering a question as
  a statement changes the meaning.
- The vocabulary is a fixed lexicon of 264 isolated signs with no YES, NO, PLEASE,
  HELP, WHERE, WHAT or NOT. Signers therefore use "Location" for WHERE, "Time" for
  WHEN, "Price" for HOW MUCH, and "Mean" for WHAT DOES X MEAN. Read these as the
  question words they stand in for.
- These are real patients and travellers. Use plain, respectful, everyday register
  — the way a person speaks at a hospital reception desk, not formal written prose.
- Use the correct script for each language. Never transliterate into Latin.
- Where a language marks politeness (Hindi आप, Tamil நீங்கள்), use the polite form.
- Set "reordered" true when the natural spoken order differs from the signed order.`;

function buildUserMessage(batch: GlossSeq[]): string {
  const items = batch.map((seq) => {
    const key = seq.glosses.join("|");
    const note = seq.note ? `\n  context: ${seq.note}` : "";
    return `- key: ${key}\n  signed order: ${seq.glosses.join(" ")}\n  setting: ${seq.domain}${note}`;
  });
  return [
    `Render each of these ${batch.length} gloss sequences into the six languages.`,
    "",
    ...items,
    "",
    `Return one rendering per key, with "key" copied back exactly as given.`,
    `Target languages: ${LANGS.map((l) => `${l} = ${LANG_NAMES[l]}`).join(", ")}.`,
  ].join("\n");
}

type Entry = {
  glosses: string[];
  domain: string;
  reordered: boolean;
  register: string;
} & Record<Lang, string>;

function loadExisting(): Record<string, Entry> {
  try {
    const raw = fs.readFileSync(OUT_PATH, "utf8");
    const parsed = JSON.parse(raw);
    return parsed?.entries ?? {};
  } catch {
    return {};
  }
}

function validateCorpus(labels: Set<string>): { usable: GlossSeq[]; dropped: GlossSeq[] } {
  const usable: GlossSeq[] = [];
  const dropped: GlossSeq[] = [];
  for (const seq of GLOSS_CORPUS) {
    // "Water" and "Help" live in speech.ts PHRASES but are NOT INCLUDE classes;
    // anything the classifier cannot emit is not a reachable utterance.
    if (seq.glosses.every((g) => labels.has(g))) usable.push(seq);
    else dropped.push(seq);
  }
  return { usable, dropped };
}

async function main() {
  const labels = new Set<string>(JSON.parse(fs.readFileSync(LABELS_PATH, "utf8")));
  const { usable, dropped } = validateCorpus(labels);

  console.log(`corpus: ${GLOSS_CORPUS.length} sequences, ${usable.length} usable`);
  if (dropped.length) {
    console.log(`dropped ${dropped.length} using glosses the model cannot emit:`);
    for (const d of dropped) {
      const bad = d.glosses.filter((g) => !labels.has(g));
      console.log(`  ${d.glosses.join("|")}  (unknown: ${bad.join(", ")})`);
    }
  }

  const existing = loadExisting();
  const todo = usable.filter((s) => !existing[s.glosses.join("|")]);
  console.log(`${Object.keys(existing).length} already cached, ${todo.length} to generate`);

  if (todo.length === 0) {
    console.log("nothing to do");
    return;
  }

  // Check credentials BEFORE the first batch. Without this the run prints
  // "batch 1/10" and then dies in an SDK stack trace, which reads like the
  // generator is broken rather than like a key is missing.
  if (!process.env.ANTHROPIC_API_KEY && !process.env.ANTHROPIC_AUTH_TOKEN) {
    console.error(`
No API credentials found.

This generator runs OFFLINE and on purpose: the table it writes is committed and
served statically, so no key ever reaches the browser and the demo never depends
on a live API call.

  ANTHROPIC_API_KEY=sk-ant-... node --experimental-strip-types tools/buildGlossTable.ts

${todo.length} sequences to generate, in ${Math.ceil(todo.length / BATCH_SIZE)} batches.
Progress is cached to ${path.relative(APP, OUT_PATH)}, so an interrupted run resumes where it stopped.
`);
    process.exitCode = 1;
    return;
  }

  const client = new Anthropic();
  const entries: Record<string, Entry> = { ...existing };

  for (let i = 0; i < todo.length; i += BATCH_SIZE) {
    const batch = todo.slice(i, i + BATCH_SIZE);
    const n = Math.floor(i / BATCH_SIZE) + 1;
    const total = Math.ceil(todo.length / BATCH_SIZE);
    console.log(`batch ${n}/${total} (${batch.length} sequences)...`);

    const response = await client.messages.parse({
      model: "claude-opus-5",
      max_tokens: 16000,
      thinking: { type: "adaptive" },
      system: SYSTEM,
      messages: [{ role: "user", content: buildUserMessage(batch) }],
      output_config: { format: zodOutputFormat(BatchSchema) },
    });

    if (response.stop_reason === "refusal") {
      console.error(`  refused: ${response.stop_details?.explanation ?? "no explanation"}`);
      continue;
    }

    const parsed = response.parsed_output;
    if (!parsed) {
      console.error("  could not parse response, skipping batch");
      continue;
    }

    const byKey = new Map(parsed.renderings.map((r) => [r.key, r]));
    for (const seq of batch) {
      const key = seq.glosses.join("|");
      const r = byKey.get(key);
      if (!r) {
        console.error(`  missing rendering for ${key}`);
        continue;
      }
      entries[key] = {
        glosses: seq.glosses,
        domain: seq.domain,
        reordered: r.reordered,
        register: r.register,
        "hi-IN": r["hi-IN"],
        "ta-IN": r["ta-IN"],
        "te-IN": r["te-IN"],
        "bn-IN": r["bn-IN"],
        "mr-IN": r["mr-IN"],
        "en-IN": r["en-IN"],
      };
      console.log(`  ${key}  ->  ${r["en-IN"]}`);
    }

    // Write after every batch so an interrupted run keeps its progress.
    writeTable(entries);
  }

  console.log(`\nwrote ${Object.keys(entries).length} entries to ${path.relative(ROOT, OUT_PATH)}`);
}

function writeTable(entries: Record<string, Entry>) {
  const payload = {
    generated: new Date().toISOString(),
    model: "claude-opus-5",
    note: "Generated offline by app/tools/buildGlossTable.ts. The app reads this with no network call.",
    entries,
  };
  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(OUT_PATH, JSON.stringify(payload, null, 2) + "\n");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
