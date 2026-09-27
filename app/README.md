# Setu · accessible tertiary services

Setu is a shared communication space for service users and service staff, for SIH26199. It opens in **General services**, usable at any service desk. A service selector provides contextual written phrases and five-turn guided conversations for:

- General services: understanding a process and its charges
- Banking: asking about an unexpected account charge
- Hospitality: asking about a reservation and check-in
- Retail: understanding return and exchange options
- Transport & travel: asking about routes, timings and fares
- Public services: understanding an application and required documents

These are communication examples, not live service integrations. Setu does not process bookings, transactions, account changes or applications. Any service can use custom text and the general communication phrases. Changing the setting preserves the draft and previous messages; selecting another phrase replaces the draft. Messages retain their original service context in the transcript and download.

## Run and demonstrate

From this `app` directory (Node 22.12+):

```sh
npm ci
npm run dev
```

Open http://127.0.0.1:5174 (or the address printed by Vite if that port is occupied).

1. Choose **Service type**, or keep **General services**.
2. Click **Start guided demo**, review the draft and click **Send to staff**.
3. Click **Select next phrase**, review the staff reply and click **Send to user**.
4. Continue selecting and sending until all five turns are complete.
5. Try another setting, write your own message, or search **Phrase library**. **Download conversation** exports the actual transcript; **New conversation** asks before clearing it.

## Working features and honest limits

- Two-way written phrases and typed messages, editable drafts, source-labelled shared display, transcript download and session reset. Messages clear on page reload.
- Written phrases in English and Hindi. Six existing speech language choices are retained. Other selections use an explicit English fallback for written phrases. Custom text is never translated automatically.
- Real browser speech input where supported. Microphone permission and possibly an online service are required. Actual results remain drafts for review. Failures never generate sample transcripts.
- Read-aloud uses installed browser voices and reports missing voices or playback errors. Audible output and live microphone accuracy still need device testing.
- **Try sign recognition** retains the original local model, tracker, segmenter and dictionary matching. General-vocabulary results go to an editable draft. Live signer accuracy is unverified; full service-specific sentences and ISL grammar are not supported.
- **Explore recorded signs** uses actual recorded frames. Word matching is not grammatical ISL translation. Unsupported words stay text, with no unrelated sign substituted.
- **Access & capabilities** includes original installation and landmark-recording tools. Recording exports data; it does not train the model.
- Keyboard focus, a skip link, status announcements, larger text, responsive layouts and a native reset dialog are included. Automated accessibility checks do not replace evaluation with Deaf/ISL users and assistive technology.
- Planned: verified service-specific ISL guidance/grammar, wider phrase coverage and native-speaker/ISL-user review. The selectable service settings already support text and written phrases; they are not merely roadmap cards.

## Offline production demo

```sh
npm run build
npm run preview -- --port 4174
```

Open http://127.0.0.1:4174 online, wait for the service worker to register, then reload online once to cache built assets. Disconnect and reload: the cached phrase/text interface remains usable. First use needs internet. Recognition and playback require their assets cached separately; speech may need internet. Development mode does not register a service worker.

## Verification

```sh
npm run check
npm test
npm run lint
npx playwright install chromium
npm run test:e2e
```

The browser suite requires a current production build and port 4174 free. It tests both communication directions, all six service scenarios, preserved messages/drafts across setting changes, Hindi and typed replies, language fallback, transcript export/reset, unsupported speech, actual sign playback, missing model/camera errors, cached offline reload, mobile/large-text/keyboard interaction and automated WCAG A/AA checks.

Fault tests inject unavailable devices/services; they do not prove live microphone or camera translation accuracy. Existing recognition dependencies produce a large build-chunk warning and lint reports four warnings, no errors.

## Updated files

- `src/App.tsx`, `src/ServiceWorkspace.css`, `src/context/SessionContext.tsx`: general service workspace, service selection, guided demos and shared conversation.
- `src/lib/phrases.ts`, `domains.ts`, `phrasebook.ts`, `phrasebookTable.ts`, `components/PhraseBoard.tsx`: service-specific English/Hindi phrases and shared communication requests.
- `src/lib/browserSpeech.ts`, `speech.ts`, `serviceVocabulary.ts`, `glossTranslate.ts`, `reverse.ts`, `sentence.ts`, `components/SignBridge.tsx`, `hooks/useSignLibrary.ts`: preserved real speech/ISL infrastructure with honest status and vocabulary limits.
- `index.html`, `public/manifest.webmanifest`, `public/favicon.svg`, `public/sw.js`: neutral identity and refreshed offline cache.
- `e2e/counter.spec.ts`, `playwright.config.ts`, package files and `.gitignore`: browser and accessibility verification.

The previous sector-specific screens have been replaced. Raw model tensors, class-index labels and recording archives are preserved to avoid breaking trained-model integrity. Presentation files are unchanged.
