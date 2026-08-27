import React, { createContext, useContext, useState, useCallback, useEffect } from "react";
import { LANGUAGES, type LangCode } from "../lib/speech";

export type ScreenId = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;
export type ActiveView = "home" | "bridge" | "language" | "transcript" | "phrases" | "summary" | "diagnostics" | "devices";

export interface TranscriptItem {
  id: string;
  speaker: "doctor" | "patient" | "prescription";
  speakerName: string;
  timestamp: string;
  text: string;
  textEn?: string;
  glosses?: string[];
  category?: "doctor" | "patient" | "prescription";
  medication?: {
    name: string;
    dose: string;
    duration: string;
    instructions: string;
  };
}

export interface ProjectionState {
  text: string;
  textEn?: string;
  glosses?: string[];
  timestamp: number;
}

interface SessionContextType {
  activeView: ActiveView;
  setActiveView: (view: ActiveView) => void;
  navigateToView: (view: ActiveView) => void;
  currentScreen: ScreenId;
  setCurrentScreen: (s: ScreenId) => void;
  maxVisitedStep: ScreenId;
  navigateToStep: (s: ScreenId) => void;
  isTransitioning: boolean;
  transitionMessage: string;
  selectedRole: "doctor" | "patient";
  setSelectedRole: (role: "doctor" | "patient") => void;
  selectedLang: LangCode;
  setSelectedLang: (lang: LangCode) => void;
  transcript: TranscriptItem[];
  addTranscriptItem: (item: Omit<TranscriptItem, "id" | "timestamp">) => void;
  activeProjection: ProjectionState | null;
  projectToPatient: (p: { text: string; textEn?: string; glosses?: string[] }) => void;
  isSlowMode: boolean;
  setIsSlowMode: (slow: boolean) => void;
  deviceReady: { camera: boolean; mic: boolean };
  setDeviceReady: React.Dispatch<React.SetStateAction<{ camera: boolean; mic: boolean }>>;
  humanInterpreterRequested: boolean;
  requestHumanInterpreter: () => void;
  resetSession: () => void;
  parityStatus: { checked: boolean; isParityValid: boolean; maxDiff: number; error: string | null };
}

function readView(): ActiveView {
  const view = window.location.hash.slice(1);
  return ["home", "bridge", "language", "transcript", "phrases", "summary", "diagnostics", "devices"].includes(view) ? view as ActiveView : "home";
}

const SESSION_KEY = "setu-session-v1";
function readSession(): { selectedRole?: "doctor" | "patient"; selectedLang?: LangCode; transcript?: TranscriptItem[]; activeProjection?: ProjectionState | null } {
  try {
    const saved = JSON.parse(sessionStorage.getItem(SESSION_KEY) || "null");
    if (!saved || !Array.isArray(saved.transcript) || !saved.transcript.every((item: TranscriptItem) => item && typeof item.id === "string" && typeof item.text === "string" && typeof item.timestamp === "string" && ["doctor", "patient", "prescription"].includes(item.speaker))) return {};
    return {
      selectedRole: saved.selectedRole === "patient" ? "patient" : "doctor",
      selectedLang: LANGUAGES.some(l => l.code === saved.selectedLang) ? saved.selectedLang : "hi-IN",
      transcript: saved.transcript,
      activeProjection: saved.activeProjection && typeof saved.activeProjection.text === "string" ? saved.activeProjection : null,
    };
  } catch { return {}; }
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

const STEP_TO_VIEW: Record<ScreenId, ActiveView> = {
  1: "home",
  2: "language",
  3: "devices",
  4: "bridge",
  5: "bridge",
  6: "transcript",
  7: "phrases",
  8: "summary",
};

const VIEW_TO_STEP: Record<ActiveView, ScreenId> = {
  home: 1,
  bridge: 4,
  language: 2,
  transcript: 6,
  phrases: 7,
  summary: 8,
  diagnostics: 3,
  devices: 3,
};

export const SessionProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [initialSession] = useState(readSession);
  const [activeView, setActiveViewState] = useState<ActiveView>(() => readView());
  const [currentScreen, setCurrentScreenState] = useState<ScreenId>(() => VIEW_TO_STEP[readView()]);
  const [maxVisitedStep, setMaxVisitedStep] = useState<ScreenId>(8); // Unlock full navigation
  const isTransitioning = false;
  const transitionMessage = "";

  const [selectedRole, setSelectedRole] = useState<"doctor" | "patient">(initialSession.selectedRole ?? "doctor");
  const [selectedLang, setSelectedLang] = useState<LangCode>(initialSession.selectedLang ?? "hi-IN");
  const [transcript, setTranscript] = useState<TranscriptItem[]>(initialSession.transcript ?? []);
  const [activeProjection, setActiveProjection] = useState<ProjectionState | null>(initialSession.activeProjection ?? null);
  const [isSlowMode, setIsSlowMode] = useState(false);
  const [deviceReady, setDeviceReady] = useState({ camera: false, mic: false });
  const [humanInterpreterRequested, setHumanInterpreterRequested] = useState(false);
  const [parityStatus] = useState({
    checked: false,
    isParityValid: false,
    maxDiff: 0.0,
    error: null,
  });

  useEffect(() => {
    try { sessionStorage.setItem(SESSION_KEY, JSON.stringify({selectedRole, selectedLang, transcript, activeProjection})); } catch { /* The current visit remains usable when browser storage is unavailable. */ }
  }, [selectedRole, selectedLang, transcript, activeProjection]);

  useEffect(() => {
    const sync = () => { const view = readView(); setActiveViewState(view); setCurrentScreenState(VIEW_TO_STEP[view]); };
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);

  const navigateToView = useCallback((targetView: ActiveView) => {
    window.location.hash = targetView;
    setActiveViewState(targetView);
    setCurrentScreenState(VIEW_TO_STEP[targetView]);
  }, []);

  const setActiveView = useCallback(
    (view: ActiveView) => {
      navigateToView(view);
    },
    [navigateToView]
  );

  const navigateToStep = useCallback(
    (targetStep: ScreenId) => {
      if (targetStep === 4 || targetStep === 5) setSelectedRole(targetStep === 4 ? "doctor" : "patient");
      const view = STEP_TO_VIEW[targetStep] || "home";
      navigateToView(view);
    },
    [navigateToView]
  );

  const setCurrentScreen = useCallback(
    (s: ScreenId) => {
      navigateToStep(s);
    },
    [navigateToStep]
  );

  const addTranscriptItem = useCallback(
    (item: Omit<TranscriptItem, "id" | "timestamp">) => {
      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const newItem: TranscriptItem = {
        ...item,
        id: `tx-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
        timestamp: timeStr,
      };
      setTranscript((prev) => [...prev, newItem]);
    },
    []
  );

  const projectToPatient = useCallback(
    (p: { text: string; textEn?: string; glosses?: string[] }) => {
      setActiveProjection({
        ...p,
        timestamp: Date.now(),
      });
      addTranscriptItem({
        speaker: "doctor",
        speakerName: "Doctor",
        text: p.text,
        textEn: p.textEn,
        glosses: p.glosses,
        category: "doctor",
      });
    },
    [addTranscriptItem]
  );

  const requestHumanInterpreter = useCallback(() => {
    setHumanInterpreterRequested(true);
    setTimeout(() => {
      setHumanInterpreterRequested(false);
    }, 6000);
  }, []);

  const resetSession = useCallback(() => {
    setTranscript([]);
    setActiveProjection(null);
    window.location.hash = "home";
    setActiveViewState("home");
    setCurrentScreenState(1);
    setMaxVisitedStep(8);
    setHumanInterpreterRequested(false);
    setIsSlowMode(false);
    setDeviceReady({camera: false, mic: false});
  }, []);

  return (
    <SessionContext.Provider
      value={{
        activeView,
        setActiveView,
        navigateToView,
        currentScreen,
        setCurrentScreen,
        maxVisitedStep,
        navigateToStep,
        isTransitioning,
        transitionMessage,
        selectedRole,
        setSelectedRole,
        selectedLang,
        setSelectedLang,
        transcript,
        addTranscriptItem,
        activeProjection,
        projectToPatient,
        isSlowMode,
        setIsSlowMode,
        deviceReady,
        setDeviceReady,
        humanInterpreterRequested,
        requestHumanInterpreter,
        resetSession,
        parityStatus,
      }}
    >
      {children}
    </SessionContext.Provider>
  );
};

export const useSession = () => {
  const ctx = useContext(SessionContext);
  if (!ctx) {
    throw new Error("useSession must be used within a SessionProvider");
  }
  return ctx;
};
