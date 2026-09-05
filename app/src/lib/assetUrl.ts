/**
 * Resolve a bundled asset against the deployment's base path.
 *
 * The app fetches its model, labels and sign library at runtime with paths like
 * "/model/labels.json". That absolute leading slash is correct on a dev server
 * and on any root-domain host, and WRONG the moment the app is served from a
 * subpath: GitHub Pages puts it at /<repo>/, so "/model/labels.json" resolves
 * to the domain root and 404s. The page loads, the model never arrives, and the
 * failure looks like "the model is broken" rather than "the path is wrong".
 *
 * Vite substitutes BASE_URL at build time from `base` in vite.config.ts, so one
 * build works at the root and another at a subpath with no code change.
 */
export function asset(path: string): string {
  // import.meta.env is substituted by Vite at build time and is always present
  // in the browser. It is NOT present under plain Node, so guard it, otherwise
  // this module cannot be unit-tested without a bundler, and a helper that can
  // only run inside the app is a helper nobody tests.
  const base = import.meta.env?.BASE_URL || "/";
  return `${base.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
}
