# Outreach: ISH News and ISH Shiksha

Two messages. Send the first; the second only if they reply and want detail.

Why this matters more than it looks: a Deaf-run organisation publicly backing
this is worth more to a judge than any accuracy number in the repo, and it is
the difference between building *with* the Deaf community and building *about*
them. It also settles the standing of the data before anyone asks.

Contact: the ISH News YouTube channel's About tab, or Instagram
(@ishnews / @ishshiksha). Ask for whoever handles partnerships or education.

---

## Message 1: the ask

> Subject: Indian Sign Language recognition, built by students, asking your permission
>
> Hello,
>
> We are a student team from India building a free, open-source tool that helps
> Deaf patients talk with doctors when no interpreter is available. It recognises
> Indian Sign Language on a phone or laptop camera and speaks it aloud in eleven
> Indian languages. It is free, runs entirely on the device, and no video ever
> leaves it.
>
> We are writing for two reasons.
>
> **First, to ask permission.** To teach the system what ISL looks like, we used
> your published videos: your news bulletins, and the word-of-the-day clips on ISH
> Shiksha. We do not store or republish any video. We extract 65 skeleton points
> per frame, from which no video, face or identity can be reconstructed, and the
> model learns only from those coordinates. We would rather ask you now than have
> you discover it later, and we will remove it immediately if you would prefer.
>
> **Second, to say thank you, and that your subtitles are unusually valuable.**
> Because your bulletins carry manually written English subtitles with accurate
> timings, each subtitle becomes a paired example of ISL and its English meaning.
> That kind of paired data barely exists for Indian Sign Language. It exists for
> German and Chinese sign language, and researchers have used those to build
> translation systems. Your archive is the Indian equivalent, and as far as we can
> tell nobody has used it this way before.
>
> We would like to work with you rather than around you. If you are open to it, we
> would value a short conversation about what would be useful to you, and about
> anything we have got wrong. We are students, not a company, and this will stay
> free.
>
> Thank you for the work you publish. It is the reason any of this was possible.
>
> [names], Team Awaaz
> [repo link] · [live link]

---

## Message 2: the detail, only if they ask

> Thank you for replying. Here is exactly what we did.
>
> **What we took.** From ISH News, 171 bulletins. From ISH Shiksha, 455
> word-of-the-day clips. For each we downloaded the video once, ran Google's
> MediaPipe on it to get 65 body and hand coordinates per frame, and deleted the
> video. What we keep is a list of numbers per frame, plus your English subtitle
> text where there was one.
>
> **What that gives us.** 7,231 pairs of "this stretch of signing means this
> English sentence". For comparison, the German dataset the research field is
> built on has 8,257, and the Chinese one 20,654. Yours is already comparable, and
> your archive holds far more than we have processed.
>
> **What we have not done.** We have not published the extracted data, and we will
> not without your agreement. We have not republished any video. Nothing is behind
> a paywall and nothing ever will be: the licence is MIT.
>
> **Where we are honest about limits.** The recognition part works for 83 signs and
> is right about 68% of the time for a signer it has never seen, so we present it
> as an aid and not an interpreter. The sentence translation does not work yet.
> Our attempts scored close to zero, which we think needs hardware we do not have.
> We say this in the app itself, because a tool that overstates itself in a
> hospital is dangerous.
>
> **What would help most.** Three things, in order:
>
> 1. Permission to continue, and to publish the extracted coordinates so other
>    researchers can use them. That would be the first openly available dataset of
>    this kind for ISL.
> 2. A conversation with a Deaf ISL user about our phrase list. It was machine
>    translated and has not been checked by anyone who signs.
> 3. If it is ever possible: recordings of *pain*, *yes*, *no* and *please*. Those
>    four do not exist in any public ISL dataset we could find, and they are the
>    words a patient needs most.

---

## Notes before sending

- Fill in names, repo link and live link.
- Send from a personal address, not a shared team one. It reads as a person.
- Do not attach the dataset. Offer it if they ask.
- If they say no: delete `data/isl_sentences/`, `data/ishnews_landmarks/` and
  `data/shiksha_landmarks/`, and say so. All three are already gitignored, so
  nothing derived from their work has been published.
