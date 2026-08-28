import { useEffect, useRef, useState } from "react";
import { Download, X, Share } from "lucide-react";

/**
 * Offer to install Setu as an app.
 *
 * The reason this matters is not the home-screen icon. Installing is what makes
 * the service worker's cache reliable: a browser can evict a tab's storage under
 * pressure, an installed app's is far stickier. In a clinic where the wifi drops
 * mid-consultation, that is the difference between the phrase board being there
 * and not.
 *
 * Rules it follows, all of which exist because install prompts are usually
 * obnoxious:
 *   - never while already installed
 *   - not immediately. Asking someone to install something they have not looked
 *     at yet is how prompts get dismissed reflexively
 *   - a dismissal is remembered, so it is asked once and not on every visit
 *   - iOS gets instructions instead of a button, because Safari has no
 *     beforeinstallprompt and there is no way to trigger the install from script
 */

type Choice = { outcome: "accepted" | "dismissed" };
interface InstallEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<Choice>;
}

const SNOOZE_KEY = "setu.install.snoozed";
const SNOOZE_DAYS = 30;
const DELAY_MS = 12_000;

function installed(): boolean {
  return window.matchMedia?.("(display-mode: standalone)").matches
    // iOS marks an installed web app here rather than via display-mode
    || (navigator as { standalone?: boolean }).standalone === true;
}

function snoozed(): boolean {
  try {
    const at = Number(localStorage.getItem(SNOOZE_KEY) || 0);
    return at > 0 && Date.now() - at < SNOOZE_DAYS * 864e5;
  } catch {
    // Private windows throw on localStorage. Showing the prompt is the safe
    // failure: worst case someone sees it twice.
    return false;
  }
}

function isIOS(): boolean {
  const ua = navigator.userAgent;
  return /iPad|iPhone|iPod/.test(ua)
    // iPadOS reports itself as a Mac; the touch points give it away
    || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
}

export default function InstallPrompt() {
  const [show, setShow] = useState(false);
  // Derived once at init rather than set from the effect: it depends only on
  // the user agent, which cannot change while mounted, and setting it inside
  // the effect costs a second render for no reason.
  const [ios] = useState(isIOS);
  const deferred = useRef<InstallEvent | null>(null);

  useEffect(() => {
    if (installed() || snoozed()) return;

    let timer: number | undefined;
    const onPrompt = (e: Event) => {
      // Keep the event: calling prompt() later is only allowed if the default
      // mini-infobar was prevented here.
      e.preventDefault();
      deferred.current = e as InstallEvent;
      timer = window.setTimeout(() => setShow(true), DELAY_MS);
    };
    window.addEventListener("beforeinstallprompt", onPrompt);

    // Safari never fires the event, so offer instructions on the same delay.
    if (ios) timer = window.setTimeout(() => setShow(true), DELAY_MS);

    const onInstalled = () => setShow(false);
    window.addEventListener("appinstalled", onInstalled);
    return () => {
      window.removeEventListener("beforeinstallprompt", onPrompt);
      window.removeEventListener("appinstalled", onInstalled);
      if (timer) clearTimeout(timer);
    };
  }, [ios]);

  const snooze = () => {
    try { localStorage.setItem(SNOOZE_KEY, String(Date.now())); } catch { /* private window */ }
    setShow(false);
  };

  const install = async () => {
    const e = deferred.current;
    if (!e) return;
    await e.prompt();
    const { outcome } = await e.userChoice;
    deferred.current = null;
    // A refusal is remembered too. Asking again next visit is what makes these
    // prompts hated.
    if (outcome === "dismissed") snooze();
    setShow(false);
  };

  if (!show) return null;

  return (
    <div className="install-prompt" role="dialog" aria-labelledby="install-title">
      <div className="install-icon" aria-hidden="true">
        <img src={`${import.meta.env.BASE_URL || "/"}icon-192.png`} alt="" width={44} height={44} />
      </div>
      <div className="install-body">
        <p id="install-title" className="install-title">Install Setu</p>
        <p className="install-text">
          {ios
            ? <>Tap <Share size={14} aria-label="the Share button" /> then <strong>Add to Home Screen</strong>. Setu then opens full screen and keeps working with no network.</>
            : <>Keeps working when the network drops, and opens full screen without the browser bar.</>}
        </p>
      </div>
      <div className="install-actions">
        {!ios && (
          <button className="install-yes" onClick={install}>
            <Download size={16} aria-hidden="true" /> Install
          </button>
        )}
        <button className="install-no" onClick={snooze} aria-label="Not now">
          <X size={16} aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}
