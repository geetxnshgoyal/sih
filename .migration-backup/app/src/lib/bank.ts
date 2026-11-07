/**
 * Dictionary lookup: recognising signs the classifier has no output for.
 *
 * The model ships with 83 answers. Pain, please, help, water, blood, bandage
 * and injection are not among them, and cannot be: every source that has those
 * words has exactly one clip of each, and a class with one example cannot be
 * both taught and examined.
 *
 * So they are not classified, they are looked up. Each word in the bank is one
 * 256-d vector, the average of every reference clip of that sign, taken from
 * the same layer of the same model that produces the live embedding. Matching
 * is cosine similarity, which for unit vectors is a dot product.
 *
 * Measured, leaving each query clip out of its own reference vector:
 *
 *     all queries              56.6% top-1   82.4% top-5   (n=244)
 *     query from a corpus
 *     the reference is not in  46.0% top-1   72.0% top-5   (n=100)
 *
 * over a 116-word bank, against a 0.86% chance rate. The second row is the
 * honest one, and it is why this surfaces a SHORTLIST and calls itself a
 * dictionary match rather than announcing an answer. 53 of the 116 words exist
 * as a single clip in the world and could not be tested at all.
 */

export type BankMatch = { word: string; score: number; refs: number };

type BankFile = {
  dim: number;
  scale: number;
  words: string[];
  refs: number[];
  sources: string[][];
  data: string;
};

export class SignBank {
  private vectors: Float32Array | null = null;   // words x dim, row-major
  private words: string[] = [];
  private refs: number[] = [];
  private dim = 0;

  get ready() { return this.vectors !== null; }
  get size() { return this.words.length; }
  get vocabulary() { return [...this.words]; }

  async load(url: string) {
    const spec: BankFile = await fetch(url).then(r => r.json());
    const raw = atob(spec.data);
    const n = spec.words.length;
    const v = new Float32Array(n * spec.dim);
    // int8 with one global scale, then renormalised: quantisation moves each
    // vector slightly off the unit sphere and cosine assumes it is on it
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < spec.dim; j++) {
        const b = raw.charCodeAt(i * spec.dim + j);
        const s = (b > 127 ? b - 256 : b) * spec.scale;
        v[i * spec.dim + j] = s;
        sum += s * s;
      }
      const norm = Math.sqrt(sum) || 1;
      for (let j = 0; j < spec.dim; j++) v[i * spec.dim + j] /= norm;
    }
    this.vectors = v;
    this.words = spec.words;
    this.refs = spec.refs;
    this.dim = spec.dim;
  }

  /** Nearest words to a live embedding, best first. */
  lookup(embedding: Float32Array, k = 5): BankMatch[] {
    if (!this.vectors || embedding.length !== this.dim) return [];
    const out: BankMatch[] = [];
    for (let i = 0; i < this.words.length; i++) {
      let dot = 0;
      const off = i * this.dim;
      for (let j = 0; j < this.dim; j++) dot += this.vectors[off + j] * embedding[j];
      out.push({ word: this.words[i], score: dot, refs: this.refs[i] });
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, k);
  }
}
