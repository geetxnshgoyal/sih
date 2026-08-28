/**
 * Install state, captured before React exists.
 *
 * `beforeinstallprompt` fires very early -- often before the first render. A
 * listener registered inside a component therefore MISSES it on a fast load,
 * and the install offer can never appear no matter how correct the manifest is.
 * That was the bug: the PWA was installable, the app just never knew.
 *
 * So this module registers at import time (main.tsx imports it first) and holds
 * the event for whoever wants it later.
 */

type Choice = { outcome: "accepted" | "dismissed" };
interface InstallEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<Choice>;
}

let deferred: InstallEvent | null = null;
let installedFlag = false;
const listeners = new Set<() => void>();

const notify = () => listeners.forEach((fn) => fn());

if (typeof window !== "undefined") {
  window.addEventListener("beforeinstallprompt", (e) => {
    // Must preventDefault here or prompt() is not allowed to be called later.
    e.preventDefault();
    deferred = e as InstallEvent;
    notify();
  });
  window.addEventListener("appinstalled", () => {
    deferred = null;
    installedFlag = true;
    notify();
  });
}

export function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** True once the browser has offered us an install we can trigger from script. */
export const canPrompt = (): boolean => deferred !== null;

export function isInstalled(): boolean {
  if (installedFlag) return true;
  if (typeof window === "undefined") return false;
  return window.matchMedia?.("(display-mode: standalone)").matches
    // iOS marks an installed web app here rather than through display-mode
    || (navigator as { standalone?: boolean }).standalone === true;
}

export async function promptInstall(): Promise<"accepted" | "dismissed" | "unavailable"> {
  if (!deferred) return "unavailable";
  await deferred.prompt();
  const { outcome } = await deferred.userChoice;
  deferred = null;
  notify();
  return outcome;
}

export type Platform = "ios" | "android" | "desktop";

export function platform(): Platform {
  if (typeof navigator === "undefined") return "desktop";
  const ua = navigator.userAgent;
  // iPadOS reports itself as a Mac; the touch points give it away
  if (/iPad|iPhone|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1))
    return "ios";
  if (/Android/.test(ua)) return "android";
  return "desktop";
}

/**
 * What to tell someone whose browser will not hand us a prompt.
 *
 * Chrome only fires beforeinstallprompt once its own engagement heuristics are
 * satisfied, and Safari never fires it at all. Without these instructions the
 * app is installable and the person has no way to discover it.
 */
export function manualSteps(p: Platform = platform()): string {
  switch (p) {
    case "ios":
      return "In Safari, tap the Share button, then Add to Home Screen.";
    case "android":
      return "In Chrome, open the ⋮ menu, then tap Install app or Add to Home screen.";
    default:
      return "In Chrome or Edge, click the install icon at the right of the address bar, or open the ⋮ menu and choose Install.";
  }
}
