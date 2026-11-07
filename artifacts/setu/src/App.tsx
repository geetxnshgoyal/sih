import { useEffect } from "react";
import { SessionProvider, useSession, type ActiveView } from "./context/SessionContext";
import { StitchHeader } from "./components/stitch/StitchHeader";
import { StitchNav } from "./components/stitch/StitchNav";
import { HomeScreen } from "./components/stitch/HomeScreen";
import { SpokenLanguageScreen } from "./components/stitch/SpokenLanguageScreen";
import { StaffViewScreen } from "./components/stitch/StaffViewScreen";
import { ClientViewScreen } from "./components/stitch/ClientViewScreen";
import { TranscriptFeedScreen } from "./components/stitch/TranscriptFeedScreen";
import { PhraseLibraryScreen } from "./components/stitch/PhraseLibraryScreen";
import { SessionSummaryScreen } from "./components/stitch/SessionSummaryScreen";
import { DiagnosticsScreen } from "./components/stitch/DiagnosticsScreen";
import { loadGlossTable } from "./lib/glossTranslate";
import { DeviceReadinessScreen } from "./components/stitch/DeviceReadinessScreen";
import InstallPrompt from "./components/InstallPrompt";
import Recorder from "./components/Recorder";
import "./App.css";
import { ScreenErrorBoundary } from "./components/ScreenErrorBoundary";
import { useDomain } from "./lib/useDomain";

/**
 * Tab titles, written rather than derived from the URL slug.
 *
 * Capitalising the hash gave "Setu · Language" under a nav item labelled
 * "Languages", and would silently produce a new wrong title for every route
 * added later. The bridge is the exception and uses the active setting, so
 * someone with several counters open can tell the tabs apart.
 */
const VIEW_TITLE: Record<Exclude<ActiveView, "home" | "bridge">, string> = {
  capture: "Capture signs",
  language: "Languages",
  transcript: "Transcript",
  phrases: "Phrases",
  summary: "Summary",
  devices: "Device check",
  diagnostics: "System checks",
};

function ScreenRouter() {
  const { activeView, selectedRole } = useSession();
  const domain = useDomain();

  useEffect(() => {
    void loadGlossTable();

  }, []);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "instant" });
    const main = document.querySelector("main");
    main?.setAttribute("tabindex", "-1");
    main?.setAttribute("id", "main-content");
    main?.focus({ preventScroll: true });
    // The home view keeps the full descriptive title. Crawlers render this SPA
    // and index whatever document.title holds afterwards, so overwriting the
    // landing page with a bare "Setu · Home" throws away every keyword the
    // static <title> was written to carry.
    document.title = activeView === "home"
      ? "Setu: Indian Sign Language Bridge | Team Awaaz"
      : `Setu · ${activeView === "bridge" ? domain.label : VIEW_TITLE[activeView]}`;
  }, [activeView, selectedRole, domain]);

  const renderActiveView = () => {
    switch (activeView) {
      case "home":
        return <HomeScreen />;
      case "bridge":
        return selectedRole === "staff" ? <StaffViewScreen /> : <ClientViewScreen />;
      case "capture":
        return <Recorder />;
      case "language":
        return <SpokenLanguageScreen />;
      case "transcript":
        return <TranscriptFeedScreen />;
      case "phrases":
        return <PhraseLibraryScreen />;
      case "summary":
        return <SessionSummaryScreen />;
      case "devices":
        return <DeviceReadinessScreen />;
      case "diagnostics":
        return <DiagnosticsScreen />;
      default:
        return <HomeScreen />;
    }
  };

  return (
    <div className="setu-app min-h-screen bg-surface text-on-surface font-body-md antialiased">
      <a className="skip-link" href="#main-content" onClick={e => { e.preventDefault(); document.querySelector<HTMLElement>("main")?.focus(); }}>Skip to content</a>
      <StitchHeader />
      <div key={`${activeView}-${selectedRole}`} className="page-enter"><ScreenErrorBoundary>{renderActiveView()}</ScreenErrorBoundary></div>

      <StitchNav />
      <InstallPrompt />
    </div>
  );
}

export default function App() {
  return (
    <SessionProvider>
      <ScreenRouter />
    </SessionProvider>
  );
}
