import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

if (import.meta.env.DEV) {
  import("./devParity").then((m) => { m.checkModelParity(); m.checkClipParity(); });
}

// Register the service worker in production only.
//
// Not in dev: it would cache modules between edits and serve stale code, which
// is a miserable way to lose an hour. See public/sw.js for what it caches and
// why -- a clinic's wifi drops, and the model and phrase tables are static, so
// after the first visit neither should need a network.
//
// BASE_URL, not "/": the app is served from /sih/ on GitHub Pages and from the
// root elsewhere. A hardcoded path registers a worker whose scope does not
// cover the app, and it silently controls nothing.
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    const base = import.meta.env.BASE_URL || "/";
    navigator.serviceWorker
      .register(`${base}sw.js`, { scope: base })
      .catch((err) => console.warn("[sw] registration failed", err));
  });
}
