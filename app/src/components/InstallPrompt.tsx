import { useEffect, useState, useSyncExternalStore } from "react";
import { Download, X, Share, Smartphone } from "lucide-react";
import {
  canPrompt, isInstalled, manualSteps, platform, promptInstall, subscribe,
} from "../lib/install";

/**
 * Offer to install Setu.
 *
 * Installing is not about the home-screen icon. It is what makes the service
 * worker's cache reliable: a browser can evict a tab's storage under pressure,
 * an installed app's is far stickier. In a clinic where the wifi drops mid
 * consultation, that is the difference between the phrase board being there and
 * not.
 *
 * Two surfaces, because relying on the browser's own event is not enough:
 *
 *   <InstallPrompt/>  a banner, once, after a delay, dismissible for 30 days
 *   <InstallButton/>  an always-available control, so someone who wants the app
 *                     can get it even when Chrome's engagement heuristics have
 *                     not fired and on Safari, which never fires at all
 */

const SNOOZE_KEY = "setu.install.snoozed";
const SNOOZE_DAYS = 30;
const DELAY_MS = 6_000;

function snoozed(): boolean {
  try {
    const at = Number(localStorage.getItem(SNOOZE_KEY) || 0);
    return at > 0 && Date.now() - at < SNOOZE_DAYS * 864e5;
  } catch {
    // Private windows throw. Showing the banner is the safe failure: at worst
    // someone sees it twice.
    return false;
  }
}

function snooze() {
  try { localStorage.setItem(SNOOZE_KEY, String(Date.now())); } catch { /* private window */ }
}

const useInstallState = () =>
  useSyncExternalStore(subscribe, canPrompt, () => false);

/** Always-available install control. Falls back to instructions. */
export function InstallButton({ className = "" }: { className?: string }) {
  const ready = useInstallState();
  const [steps, setSteps] = useState<string | null>(null);
  if (isInstalled()) return null;

  const click = async () => {
    const r = await promptInstall();
    if (r === "unavailable") setSteps(manualSteps());
    if (r === "dismissed") snooze();
  };

  return (
    <span className={`install-inline ${className}`}>
      <button className="install-yes" onClick={click}>
        <Download size={16} aria-hidden="true" /> Install app
      </button>
      {steps && <span className="install-steps" role="status">{steps}</span>}
      {!steps && !ready && (
        <span className="install-hint">{manualSteps()}</span>
      )}
    </span>
  );
}

export default function InstallPrompt() {
  const ready = useInstallState();
  const [show, setShow] = useState(false);
  const [steps, setSteps] = useState<string | null>(null);
  const ios = platform() === "ios";

  useEffect(() => {
    if (isInstalled() || snoozed()) return;
    // Show once the browser has handed us a prompt, or on iOS where it never
    // will and instructions are the only route.
    if (!ready && !ios) return;
    const t = window.setTimeout(() => setShow(true), DELAY_MS);
    return () => clearTimeout(t);
  }, [ready, ios]);

  if (!show) return null;

  const dismiss = () => { snooze(); setShow(false); };
  const install = async () => {
    const r = await promptInstall();
    if (r === "unavailable") { setSteps(manualSteps()); return; }
    if (r === "dismissed") snooze();
    setShow(false);
  };

  return (
    <div className="install-prompt" role="dialog" aria-labelledby="install-title">
      <div className="install-icon" aria-hidden="true">
        <img src={`${import.meta.env.BASE_URL || "/"}icon-192.png`} alt="" width={44} height={44} />
      </div>
      <div className="install-body">
        <p id="install-title" className="install-title">Install Setu</p>
        <p className="install-text">
          {steps ? steps
            : ios
              ? <>Tap <Share size={14} aria-label="the Share button" /> then <strong>Add to Home Screen</strong>. Setu then opens full screen and keeps working with no network.</>
              : <>Keeps working when the network drops, and opens full screen without the browser bar.</>}
        </p>
      </div>
      <div className="install-actions">
        {!ios && !steps && (
          <button className="install-yes" onClick={install}>
            <Download size={16} aria-hidden="true" /> Install
          </button>
        )}
        {ios && !steps && <Smartphone size={18} aria-hidden="true" className="install-glyph" />}
        <button className="install-no" onClick={dismiss} aria-label="Not now">
          <X size={16} aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}
