---
name: Clinical Real-Time Medical Bridge
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#40484c'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#70787d'
  outline-variant: '#bfc8cd'
  surface-tint: '#1e667f'
  primary: '#004357'
  on-primary: '#ffffff'
  primary-container: '#0d5c75'
  on-primary-container: '#93d3ef'
  inverse-primary: '#90cfec'
  secondary: '#545f73'
  on-secondary: '#ffffff'
  secondary-container: '#d5e0f8'
  on-secondary-container: '#586377'
  tertiary: '#5d3400'
  on-tertiary: '#ffffff'
  tertiary-container: '#794a15'
  on-tertiary-container: '#febd7d'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#bde9ff'
  primary-fixed-dim: '#90cfec'
  on-primary-fixed: '#001f2a'
  on-primary-fixed-variant: '#004d64'
  secondary-fixed: '#d8e3fb'
  secondary-fixed-dim: '#bcc7de'
  on-secondary-fixed: '#111c2d'
  on-secondary-fixed-variant: '#3c475a'
  tertiary-fixed: '#ffdcbf'
  tertiary-fixed-dim: '#fbb97a'
  on-tertiary-fixed: '#2d1600'
  on-tertiary-fixed-variant: '#683c06'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  display-lg:
    fontFamily: IBM Plex Sans
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.01em
  display-md:
    fontFamily: IBM Plex Sans
    fontSize: 26px
    fontWeight: '600'
    lineHeight: 34px
    letterSpacing: -0.005em
  headline-lg:
    fontFamily: IBM Plex Sans
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 30px
    letterSpacing: 0em
  headline-md:
    fontFamily: IBM Plex Sans
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 26px
    letterSpacing: 0em
  body-xl:
    fontFamily: IBM Plex Sans
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 32px
    letterSpacing: 0.01em
  body-lg:
    fontFamily: IBM Plex Sans
    fontSize: 17px
    fontWeight: '400'
    lineHeight: 26px
    letterSpacing: 0.01em
  body-md:
    fontFamily: IBM Plex Sans
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
    letterSpacing: 0.005em
  label-lg:
    fontFamily: IBM Plex Sans
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 18px
    letterSpacing: 0.04em
  label-md:
    fontFamily: IBM Plex Sans
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.06em
  label-sm:
    fontFamily: IBM Plex Sans
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.08em
  code-transcript:
    fontFamily: IBM Plex Sans
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
    letterSpacing: 0.02em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  space-xxs: 0.125rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
  space-2xl: 2rem
  space-3xl: 3rem
  viewport-margin: 1.5rem
  panel-gutter: 1rem
---

## Brand & Style

This design system establishes a high-acuity, mission-critical communication environment bridging deaf or hard-of-hearing patients utilizing Indian Sign Language (ISL) with hearing clinicians across India’s multilingual hospital OPDs and bedside settings. 

### Visual Personality
- **Clinical & Authoritative:** Styled with the disciplined utility of diagnostic interfaces, bedside telemetry monitors, and hospital EHR systems (e.g., Epic, Cerner).
- **Zero Consumer Tropes:** Eliminates chat bubbles, decorative floating illustrations, emotional reactions, AI sparkle motifs, playful rounded micro-interactions, and neon accenting. The interface is a serious diagnostic and communication instrument.
- **Calm & Reassuring:** Prioritizes predictability, high-contrast readability at arm’s length under harsh fluorescent hospital lighting, and absolute sensory clarity to reduce cognitive load for fatigued clinicians and vulnerable patients.

### Design Style
The system employs **Structural Functionalism**: crisp 1px structural data grids, rigid structural dividers, explicit rectangular bounding boxes, monoline 2px visual telemetry indicators, and strict state-driven functional coloration. Contrast ratios strictly exceed WCAG 2.1 AAA for text (7:1) and AA for operational telemetry (4.5:1).

## Colors

The palette is derived from clinical medical instruments and diagnostic workstations. Contrast thresholds are calculated to remain legible in high-glare clinical wards and low-light examination suites.

### Core Swatches
- **Primary Clinical Teal (`#0D5C75` / `#146C87`):** Directs primary interactive workflows, patient-side targeting bounds, and active language engine selections. Conveys institutional trust and medical sobriety.
- **Deep Clinical Charcoal (`#0F172A` / `#1E293B`):** Applied to core transcript typography, vital medical status text, and high-priority clinical readouts.
- **Slate Borders & Dividers (`#CBD5E1` / `#E2E8F0`):** Defines structural container edges, framing bounds, and table cell divisions without visual noise.
- **Clinical Backdrops (`#F8FAFC` base, `#F1F5F9` sub-tier, `#FFFFFF` elevated panels):** Replicates sterile white paper charts and bedside monitor chassis.

### Functional Status Tokens
Functional color is strictly utilitarian and never decorative:
- **Confirmed / Recognized (`#15803D` on `#DCFCE7`):** Indicates verified ISL gesture capture or accurate regional speech recognition.
- **Uncertain / Re-sign (`#B45309` on `#FEF3C7`):** Indicates confidence score under threshold, gesture occlusion, or regional acoustic noise requiring restatement.
- **High-Risk Alert & Human Escalation (`#B91C1C` on `#FEE2E2`):** Reserved exclusively for Emergency Medical Interpreter summon triggers, rapid clinical degradation, or critical misinterpretation warnings.

## Typography

The typography uses **IBM Plex Sans** across all roles. Its mechanical terminals, open counters, and engineered distinct letterforms eliminate character ambiguity in clinical contexts (e.g., uppercase `I`, lowercase `l`, and numeral `1`).

### Reading Distance & Multilingual Rendering
- **Bedside & Countertop Distance:** Transcript copy uses `body-xl` (20px/32px) and `body-lg` (17px/26px) to allow effortless parsing from 1 to 1.5 meters away while a clinician is palpating or entering chart data.
- **Complex Indian Script Support:** The system relies on native Unicode shaping engines for Hindi, Tamil, Kannada, Telugu, Bengali, and Marathi, mandating standard line-height multipliers of 1.5x–1.6x to prevent glyph clipping of matras and sub-joined consonants.
- **Labeling Standard:** All operational metadata, status codes, and patient identifiers use uppercase tracking with `label-md` or `label-sm` for unambiguous scannability.

## Layout & Spacing

The layout is built on a rigid 4px/8px base spacing grid modeled after clinical diagnostic split-consoles. 

### Architectural Layout
1. **Primary Operational Split (50/50 or 60/40 Split Frame):**
   - **Visual Telemetry Zone (Left/Patient Facing):** Dedicated to high-framerate ISL computer-vision video ingest, boundary tracking guides, and recognition state telemetry.
   - **Clinical Medical Record & Transcript Stream (Right/Clinician Facing):** Timestamped, structured medical transcription split into Patient (ISL Translated) and Clinician (Audio Regional Transcribed) channels.
2. **Top Administrative Rail:** 56px fixed height housing active OPD room ID, current patient MRN, paired language selector, and emergency human interpreter escalation CTA.
3. **Bottom Diagnostic Command Strip:** 72px fixed height housing direct quick-phrases, re-sign triggers, mic push-to-talk, and optical recalibration controls.

### Breakpoints & Adaptation
- **Tablet / Bedside Workstation on Wheels (1024px – 1366px):** 50/50 dual-pane landscape view.
- **Clinical Desktop Displays (1440px+):** 40/60 split with expanded right-rail medical glossary and EHR export staging drawer.
- **Mobile Handheld (Field Triage):** Stacked view with sticky visual PIP camera viewport locked at the top 35vh, transcript scrolling underneath.

## Elevation & Depth

To emulate sterile hospital terminal displays, the design system minimizes drop shadows, eliminating consumer soft-glow and ambient blurred drops. Hierarchy is established strictly through **monochrome boundary tiers and tonal surface contrasts**.

### Depth Tiers
- **Tier 0 (Chassis Base):** Background wash (`#F8FAFC`). Flat surface.
- **Tier 1 (Clinical Cards & Split Panes):** `#FFFFFF` fill bounded by a crisp 1px solid border (`#CBD5E1`). No box-shadow.
- **Tier 2 (Active States & Modals):** High-priority focus panels or interpreter overlays use an intentional technical edge-shadow: `0 1px 3px 0 rgba(15, 23, 42, 0.08), 0 4px 12px 0 rgba(15, 23, 42, 0.05)` with a 1px border (`#94A3B8`).
- **Tier 3 (Emergency Banner Stacking):** Fixed status banners stack via absolute z-index with high-contrast perimeter fills and zero feathering.

## Shapes

The design system adopts **Soft Geometric (Level 1)** geometry. Radii are restricted to 4px (`rounded-sm`) and 6px (`rounded-md`), evoking precision-molded medical hardware consoles and laboratory instruments.

- **Panels, Windows, and Camera Overlays:** 4px radius.
- **Interactive Buttons & Badges:** 4px radius.
- **Form Controls & Quick-Panel Chips:** 4px radius.
- **Fully Pill-Shaped Radii (9999px):** Strictly forbidden to prevent consumer chat aesthetics.

## Components

### 1. Camera Framing Boundary Guide
- **Container:** High-contrast 4px rounded viewport border (`#CBD5E1`).
- **Targeting Bounding Reticle:** 2px stroke monoline target guides in Primary Teal (`#0D5C75`) denoting upper-torso and facial zone. Turns to Sage Green (`#15803D`) when signing hands and face are properly inside optical tracking bounds; shifts to Amber (`#B45309`) with an inline warning ("HANDS OUT OF FRAME") when occluded.
- **FPS & Latency Diagnostic:** Monospaced 11px uppercase readout top-right: `ISL ENGINE: ACTIVE | 60 FPS | 42MS`.

### 2. Clinical Status Badges (Strict 3-State Logic)
All recognition states display an explicit icon and high-contrast uppercase text:
- **State 1: Ready (`#F1F5F9` bg / `#475569` text / `#94A3B8` 1px border):** Visual: 8px static gray target dot.
- **State 2: Recognizing (`#FEF3C7` bg / `#B45309` text / `#F59E0B` 1px border):** Visual: 8px pulse dot + `CAPTURING ISL...` or `LISTENING (HINDI)...`
- **State 3: Recognized (`#DCFCE7` bg / `#15803D` text / `#86EFAC` 1px border):** Visual: Checkmark + `VERIFIED / 98% CONFIDENCE`.

### 3. Structured Clinical Transcript Stream
- Replaces chat bubbles with an indexed tabular clinical log.
- Each entry contains:
  - Left gutter: High-density timestamp (`14:22:08`) and speaker identifier badge (`PATIENT [ISL]` or `DOCTOR [TAMIL -> EN]`).
  - Core body: Large high-contrast text (`body-xl` for immediate parsing).
  - Sub-row: Literal translation metadata, confidence percentage, and a `[REQUEST CLARIFICATION]` text button.

### 4. High-Legibility Quick-Panel Chips
- Rectangular 4px border chips (`#FFFFFF` background, 1px `#CBD5E1` border, `#0F172A` text).
- Categorized by standard clinical history triage: "PAIN SCALE", "DURATION", "MEDICATION ALLERGY", "PREVIOUS SURGERY".
- Pressing injects prompt direct to visual ISL prompter screen with zero latency.

### 5. High-Visibility Human Interpreter Escalation Banner
- Persistent top-level container: Crimson emergency border (`#B91C1C`), background (`#FEF2F2`), text (`#991B1B`).
- Button: Solid `#B91C1C` background, `#FFFFFF` bold text, 4px radius. Monoline 2px video-headset icon.
- Label: `REQUEST LIVE MEDICAL ISL INTERPRETER (URGENT)`.

### 6. Buttons & Inputs
- **Buttons:** 40px standard height, uppercase label, 4px border-radius, no soft shadow. Focused states feature a 2px offset solid outline in `#0D5C75`.
- **Inputs & Selectors:** Solid white, 1px `#CBD5E1` border, active language switcher displays standard ISO language tokens alongside native script (`HI - हिन्दी`, `TA - தமிழ்`, `EN - English`).