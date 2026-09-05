import { useEffect } from "react";
import { SessionProvider, useSession } from "./context/SessionContext";
import { StitchHeader } from "./components/stitch/StitchHeader";
import { StitchNav } from "./components/stitch/StitchNav";
import { HomeScreen } from "./components/stitch/HomeScreen";
import { SpokenLanguageScreen } from "./components/stitch/SpokenLanguageScreen";
import { DoctorViewScreen } from "./components/stitch/DoctorViewScreen";
import { PatientViewScreen } from "./components/stitch/PatientViewScreen";
import { TranscriptFeedScreen } from "./components/stitch/TranscriptFeedScreen";
import { ClinicalPhraseLibraryScreen } from "./components/stitch/ClinicalPhraseLibraryScreen";
import { ConsultationSummaryScreen } from "./components/stitch/ConsultationSummaryScreen";
import { DiagnosticsScreen } from "./components/stitch/DiagnosticsScreen";
import { loadGlossTable } from "./lib/glossTranslate";
import { DeviceReadinessScreen } from "./components/stitch/DeviceReadinessScreen";
import InstallPrompt from "./components/InstallPrompt";
import "./App.css";
import { ScreenErrorBoundary } from "./components/ScreenErrorBoundary";

function ScreenRouter() {
  const { activeView, selectedRole } = useSession();

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
      ? "Setu — Indian Sign Language Bridge | Team Awaaz"
      : `Setu · ${activeView === "bridge" ? "Consultation" : activeView.charAt(0).toUpperCase() + activeView.slice(1)}`;
  }, [activeView, selectedRole]);

  const renderActiveView = () => {
    switch (activeView) {
      case "home":
        return <HomeScreen />;
      case "bridge":
        return selectedRole === "doctor" ? <DoctorViewScreen /> : <PatientViewScreen />;
      case "language":
        return <SpokenLanguageScreen />;
      case "transcript":
        return <TranscriptFeedScreen />;
      case "phrases":
        return <ClinicalPhraseLibraryScreen />;
      case "summary":
        return <ConsultationSummaryScreen />;
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
