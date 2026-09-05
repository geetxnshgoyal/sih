import React from "react";
import { useSession } from "../../context/SessionContext";

export const RoleSelectionScreen: React.FC = () => {
  const { selectedRole, setSelectedRole, setCurrentScreen } = useSession();

  return (
    <main className="flex flex-col relative w-full pt-20 pb-24 bg-surface min-h-screen">
      <div className="flex flex-col w-full px-4 sm:px-6 lg:px-12 max-w-6xl mx-auto py-space-md space-y-space-lg">
        {/* Emergency Escalation Banner */}
        <div className="w-full bg-error-container text-on-error-container rounded-xl p-space-md flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-space-sm min-w-0">
            <div className="w-8 h-8 rounded-lg bg-error text-on-error flex items-center justify-center shrink-0">
              <span className="material-symbols-outlined text-[18px]">support_agent</span>
            </div>
            <div className="flex flex-col min-w-0">
              <span className="font-label-md text-label-md font-semibold tracking-wide uppercase">
                Emergency Escalation
              </span>
              <span className="font-label-sm text-label-sm text-on-error-container truncate">
                Live ISL interpreter on standby • Ext 108
              </span>
            </div>
          </div>
          <a
            className="shrink-0 px-space-sm py-1 bg-error text-on-error font-label-sm text-label-sm rounded-lg flex items-center gap-1 shadow-sm active:scale-95 transition-transform"
            href="tel:108"
          >
            <span>Dial</span>
            <span className="material-symbols-outlined text-[14px]">call</span>
          </a>
        </div>

        {/* Step Header */}
        <div className="flex flex-col space-y-space-xxs pt-space-xs">
          <div className="flex items-center gap-space-xs">
            <span className="w-2 h-2 rounded-full bg-primary-container"></span>
            <span className="font-label-sm text-label-sm uppercase tracking-wider text-on-surface-variant">
              Step 1 of 3 • System Calibration
            </span>
          </div>
          <h1 className="font-display-md text-display-md text-on-surface">
            Choose Consultation Workstation Role
          </h1>
          <p className="font-body-md text-body-md text-on-surface-variant max-w-3xl">
            Select which terminal side you are setting up for this clinical session to calibrate
            hardware input streams and optical layout.
          </p>
        </div>

        {/* Role Cards Grid — Side-by-side on md+ */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-space-lg">
          {/* Doctor Card */}
          <div
            className={`role-card relative bg-surface-container-lowest text-on-surface p-space-xl rounded-2xl shadow-sm cursor-pointer transition-all duration-200 flex flex-col justify-between ${
              selectedRole === "doctor" ? "ring-2 ring-primary shadow-md" : "hover:shadow-md"
            }`}
            onClick={() => setSelectedRole("doctor")}
          >
            <div>
              <div className="flex items-start justify-between gap-space-sm">
                <div className="w-16 h-16 rounded-2xl bg-primary text-on-primary flex items-center justify-center shrink-0 shadow-sm">
                  <span
                    className="material-symbols-outlined text-[32px]"
                    style={{ fontVariationSettings: "'FILL' 1" }}
                  >
                    stethoscope
                  </span>
                </div>
                {selectedRole === "doctor" && (
                  <div className="selection-pill flex items-center gap-1 px-space-md py-1 rounded-full bg-primary text-on-primary font-label-sm text-label-sm">
                    <span className="material-symbols-outlined text-[14px]">check_circle</span>
                    <span>Active Selection</span>
                  </div>
                )}
              </div>
              <div className="mt-space-lg flex flex-col space-y-space-xs">
                <span className="font-headline-lg text-headline-lg font-bold text-primary">
                  Doctor Side Workstation
                </span>
                <div className="flex items-center gap-space-xs text-on-surface-variant font-label-md text-label-md">
                  <span className="material-symbols-outlined text-[18px]">desktop_windows</span>
                  <span>OPD Workstation • Speech & Audio Stream</span>
                </div>
                <p className="font-body-md text-body-md text-on-surface-variant pt-space-xs leading-relaxed">
                  Configured for high-fidelity multi-dialect speech pickup, real-time medical
                  glossary verification, and structured clinical transcription.
                </p>
              </div>
            </div>
            <div className="mt-space-xl pt-space-md border-t border-surface-container-low flex items-center gap-space-lg text-on-surface-variant">
              <div className="flex items-center gap-space-xs font-label-sm text-label-sm">
                <span className="material-symbols-outlined text-[18px] text-primary">mic</span>
                <span>Clinician Push-to-Talk</span>
              </div>
              <div className="flex items-center gap-space-xs font-label-sm text-label-sm">
                <span className="material-symbols-outlined text-[18px] text-primary">
                  translate
                </span>
                <span>Regional Speech to ISL</span>
              </div>
            </div>
          </div>

          {/* Patient Card */}
          <div
            className={`role-card relative bg-surface-container-lowest text-on-surface p-space-xl rounded-2xl shadow-sm cursor-pointer transition-all duration-200 flex flex-col justify-between ${
              selectedRole === "patient" ? "ring-2 ring-primary shadow-md" : "hover:shadow-md"
            }`}
            onClick={() => setSelectedRole("patient")}
          >
            <div>
              <div className="flex items-start justify-between gap-space-sm">
                <div className="w-16 h-16 rounded-2xl bg-surface-container-high text-primary flex items-center justify-center shrink-0">
                  <span className="material-symbols-outlined text-[32px]">sign_language</span>
                </div>
                {selectedRole === "patient" && (
                  <div className="selection-pill flex items-center gap-1 px-space-md py-1 rounded-full bg-primary text-on-primary font-label-sm text-label-sm">
                    <span className="material-symbols-outlined text-[14px]">check_circle</span>
                    <span>Active Selection</span>
                  </div>
                )}
              </div>
              <div className="mt-space-lg flex flex-col space-y-space-xs">
                <span className="font-headline-lg text-headline-lg font-bold text-on-surface">
                  Patient Side Bedside Terminal
                </span>
                <div className="flex items-center gap-space-xs text-on-surface-variant font-label-md text-label-md">
                  <span className="material-symbols-outlined text-[18px]">tablet</span>
                  <span>Bedside Display • ISL Sign Language Stream</span>
                </div>
                <p className="font-body-md text-body-md text-on-surface-variant pt-space-xs leading-relaxed">
                  Optimized for continuous optical gesture tracking, high-legibility sign avatar
                  prompts, and visual triage quick-answers.
                </p>
              </div>
            </div>
            <div className="mt-space-xl pt-space-md border-t border-surface-container-low flex items-center gap-space-lg text-on-surface-variant">
              <div className="flex items-center gap-space-xs font-label-sm text-label-sm">
                <span className="material-symbols-outlined text-[18px] text-primary">
                  videocam
                </span>
                <span>ISL Boundary Reticle</span>
              </div>
              <div className="flex items-center gap-space-xs font-label-sm text-label-sm">
                <span className="material-symbols-outlined text-[18px] text-primary">
                  touch_app
                </span>
                <span>Visual Triage Chips</span>
              </div>
            </div>
          </div>
        </div>

        {/* Sensor Init Card */}
        <div className="bg-surface-container p-space-lg rounded-2xl flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-space-md">
            <div className="w-10 h-10 rounded-full bg-primary-container text-on-primary flex items-center justify-center shrink-0">
              <span className="material-symbols-outlined text-[20px]">verified</span>
            </div>
            <div className="flex flex-col">
              <span className="font-label-md text-label-md font-semibold text-on-surface">
                Sensors Initialized & Calibrated
              </span>
              <span className="font-label-sm text-label-sm text-on-surface-variant">
                Microphone array & optical landmark tracking camera ready for OPD 104
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2 text-primary font-label-sm text-label-sm bg-surface-container-lowest px-space-md py-1 rounded-full">
            <span className="w-2 h-2 rounded-full bg-primary-container animate-ping"></span>
            <span className="font-semibold">OK • SENSORS ACTIVE</span>
          </div>
        </div>

        {/* Continue Button */}
        <div className="pt-space-xs max-w-xl mx-auto w-full">
          <button
            className="w-full bg-primary hover:bg-primary-container text-on-primary font-headline-md text-headline-md h-14 rounded-2xl flex items-center justify-center gap-space-sm shadow-md active:scale-[0.99] transition-all cursor-pointer"
            onClick={() => setCurrentScreen(2)}
          >
            <span>Continue to Language Setup</span>
            <span className="material-symbols-outlined text-[22px]">arrow_forward</span>
          </button>
        </div>
      </div>
    </main>
  );
};
