# Privacy

Last updated: 6 September 2026. Applies to Setu as published at
`geetxnshgoyal.github.io/sih`.

Setu is used by Deaf patients in clinical settings, so the honest answer to
"where does my video go" matters more here than in most apps. The answer is
that it does not go anywhere, and the detail below explains exactly what that
means and where the edges are.

## Your camera

**Video never leaves your device.** The camera stream is read by MediaPipe
inside your browser, converted to 65 skeleton points per frame, and classified
by a 2.1 MB model that also runs in your browser. No frame, no landmark and no
recognition result is uploaded, and there is no server to upload it to. Nothing
is recorded unless you deliberately use the Record screen, and those recordings
are saved by you, to your own device, as a file you choose to download.

Turning the camera off, or closing the tab, ends this completely.

## What is stored on your device

Two things, in your browser's local storage, readable only by this site:

| key | what | why |
|---|---|---|
| `setu.domain` | health or travel | remembers which phrase set you use |
| `setu.install.snoozed` | a timestamp | so the install offer is not shown again for 30 days |

Clearing site data removes both. Neither identifies you. There are no cookies,
no analytics, no advertising and no tracking of any kind.

## Where the edges actually are

Three places where something does leave your device. None involve your video,
and we would rather state them than let you assume otherwise.

**Hosting.** The site is served by GitHub Pages, so GitHub sees the ordinary
request information any web server sees, including your IP address, and applies
[its own privacy statement](https://docs.github.com/site-policy/privacy-policies/github-privacy-statement).

**Fonts.** Two stylesheets load from Google Fonts, which discloses your IP
address to Google. This is a convenience, not a requirement; self-hosting the
fonts would remove it.

**Speech.** When Setu speaks a phrase aloud it hands the *text* to your
operating system's speech service through the browser's Web Speech API. On most
devices this is synthesised locally, but some platforms use a cloud voice, in
which case that text reaches your OS vendor. The text is a phrase from the
board, never anything about you, but on those platforms it is not strictly local.

Once loaded, Setu works with the network off. The service worker caches the app,
the model and the phrase tables, so an installed copy keeps working with no
connection at all, which is the point.

## Health information

Setu processes what you sign or tap in order to speak it aloud. It does not
store, transmit or retain any of it. Nothing you say through Setu is written to
a transcript that outlives the tab, and no consultation record is kept.

Setu is a communication aid. It is not a medical device, it does not diagnose or
treat anything, and its sign recognition is unreliable enough that no clinical
decision should rest on it. See `ARCHITECTURE.md` §5.

## Children

Setu has no accounts and collects nothing, so it is safe for anyone to use. It
is intended for use with a clinician or carer present.

## Contact

Setu is built by Team Awaaz for Smart India Hackathon 2026. Questions, or a
report of anything in this document that is not true in practice, belong in an
issue at <https://github.com/geetxnshgoyal/sih/issues>.
