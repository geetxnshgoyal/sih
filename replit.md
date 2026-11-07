# Setu — Indian Sign Language Bridge

Setu helps Deaf and hard-of-hearing people communicate with staff using Indian Sign Language, text, and speech in six Indian languages. The imported web app runs entirely in the browser; camera frames and recognition stay on the device.

## Run and verify

- Start Preview with the managed workflow `artifacts/setu: web`.
- `pnpm --filter @workspace/setu test` — original recognition, segmentation, and playback regression tests.
- `pnpm --filter @workspace/setu run check:glosses` — validates phrase references against bundled sign recordings.
- `pnpm --filter @workspace/setu run typecheck` — frontend TypeScript check.
- The managed workflow supplies `PORT` and `BASE_PATH`; do not start a root-level dev command.
- No API credentials or database are needed for Setu.

## Structure

- `artifacts/setu/` — migrated React + Vite app, original styles, tests, tools, and public assets.
- `.migration-backup/` — original imported repository, including research/training scripts and static prototypes, retained for reference.
- `artifacts/api-server/`, `artifacts/mockup-sandbox/`, and `lib/` — workspace scaffold, not used by Setu's browser-only flows.

## Preservation requirements

- Preserve the original hash navigation, role selection, domain settings, and on-device recognition behavior.
- Keep the bundled model, sign shards, vision WASM, PWA manifest, and production service worker. Asset URLs derive from Vite's base path.
- MediaPipe stays on the imported version to match the bundled vision runtime.
- Replit handles HTTPS for camera access; the app does not need a self-signed local HTTPS plugin.
- The original external project links and SEO canonical URLs are preserved. Change the canonical and sharing URLs only when the actual published destination is known.

## Device limitations

Camera and microphone use requires browser permissions and appropriate hardware. Speech recognition and available speech voices depend on the browser. Service-worker caching is production-only, matching the original app.
