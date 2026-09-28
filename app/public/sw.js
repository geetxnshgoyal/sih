/**
 * Service worker: what makes Setu an app rather than a page.
 *
 * The point is not the home-screen icon. It is that a clinic's wifi drops and
 * the consultation has to carry on. Everything Setu needs to recognise a sign
 * and speak a phrase is static: a 2.1 MB model, a label list, and phrase tables
 * generated once with NLLB-200 and committed. None of it needs a network, so
 * after the first visit none of it should ask for one.
 *
 * Hand-rolled rather than generated. A precache manifest would have to be
 * rebuilt on every deploy and would go stale silently; runtime caching adapts
 * to Vite's hashed filenames on its own.
 */
// Bump on every model change. Forgetting to is not cosmetic: the first
// release cached the 264-class model cache-first and never bumped, so every
// returning visitor kept being served a superseded model while the site
// advertised a better one.
const VERSION = "setu-v19";
const SHELL = `${VERSION}-shell`;
const MODEL = `${VERSION}-model`;

// Resolve against the worker's own location so this works both at a domain root
// and under the /sih/ subpath GitHub Pages serves from.
const BASE = new URL("./", self.location).pathname;

// Worth having before the network disappears. Kept small and non-fatal: one
// missing file must not fail the whole install and leave the app uncached.
const PRECACHE = [
  BASE,
  `${BASE}manifest.webmanifest`,
  `${BASE}icon-192.png`,
  `${BASE}model/model.json`,
  `${BASE}model/labels.json`,
  `${BASE}model/_phrasebook.json`,
  `${BASE}model/_bank.json`,
  `${BASE}signs/index.json`,
];

self.addEventListener("install", (e) => {
  e.waitUntil((async () => {
    const c = await caches.open(SHELL);
    await Promise.allSettled(PRECACHE.map((u) => c.add(u)));
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((k) => !k.startsWith(VERSION)).map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  const sameOrigin = url.origin === self.location.origin;

  // Navigations: network first so a deploy is picked up, falling back to the
  // cached shell. Without the fallback, opening the app offline shows the
  // browser's dinosaur rather than the phrase board.
  if (req.mode === "navigate") {
    e.respondWith((async () => {
      try {
        const fresh = await fetch(req);
        (await caches.open(SHELL)).put(BASE, fresh.clone());
        return fresh;
      } catch {
        return (await caches.match(BASE)) || Response.error();
      }
    })());
    return;
  }

  // The model and its tables: serve the cached copy immediately, then refresh
  // it in the background. Re-downloading 2 MB on every load is the difference
  // between usable and not on a clinic connection, so the cache has to answer
  // first -- but PURE cache-first never updates, which is exactly how a
  // retrained model failed to reach anyone who had already visited. Stale
  // content is served once and heals itself by the next load, without needing
  // a human to remember to bump a version string.
  //
  // EXCEPT the small text tables. Stale-while-revalidate means the first load
  // after a deploy is answered by the PREVIOUS service worker out of its own
  // cache, so a returning user gets the old table for that whole visit and
  // only heals on the next reload. For a 2 MB weights file that is the right
  // trade. For the phrase board it is not: the board is the path we tell
  // people is exact, and a stale table silently drops every phrase back to
  // English, which is the one failure the board exists to prevent. These are
  // tens to low hundreds of KB, so network-first costs a round trip and
  // falls back to cache when offline.
  const SMALL_TABLES = [
    "_phrasebook.json", "_utterances.json", "_bank.json",
    "labels.json", "metrics.json", "_framing.json", "model.json",
  ];
  if (sameOrigin && SMALL_TABLES.some((f) => url.pathname.endsWith(f))) {
    e.respondWith((async () => {
      const cache = await caches.open(MODEL);
      try {
        const fresh = await fetch(req);
        if (fresh.ok) cache.put(req, fresh.clone());
        return fresh;
      } catch {
        return (await cache.match(req)) || Response.error();
      }
    })());
    return;
  }

  const isModel = sameOrigin && (url.pathname.includes("/model/") || url.pathname.includes("/signs/"));
  if (isModel) {
    e.respondWith((async () => {
      const cache = await caches.open(MODEL);
      const hit = await cache.match(req);
      const fresh = fetch(req).then((res) => {
        if (res.ok) cache.put(req, res.clone());
        return res;
      }).catch(() => null);
      // Offline with nothing cached is the only case that can fail here.
      return hit || (await fresh) || Response.error();
    })());
    return;
  }

  // Built assets are content-hashed, so a cached copy can never be stale.
  if (sameOrigin && url.pathname.startsWith(`${BASE}assets/`)) {
    e.respondWith((async () => {
      const hit = await caches.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) (await caches.open(SHELL)).put(req, res.clone());
      return res;
    })());
    return;
  }

  // Everything else, including Google Fonts: try the network, fall back to any
  // cached copy. Fonts failing offline costs typography, not function.
  e.respondWith(
    fetch(req).then((res) => {
      if (res.ok && sameOrigin) {
        caches.open(SHELL).then((c) => c.put(req, res.clone()));
      }
      return res;
    }).catch(async () => (await caches.match(req)) || Response.error())
  );
});
