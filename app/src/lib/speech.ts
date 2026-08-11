/**
 * Speech output.
 *
 * Translation policy: the demo vocabulary is precomputed and bundled, so the
 * common path speaks in ~50ms with no network at all. Bhashini is only for
 * glosses outside the table, where a 500ms round trip is invisible because the
 * user is still finishing their sentence. A government API must never sit on
 * the critical path of a live demo.
 */

export const LANGUAGES = [
  { code: "hi-IN", label: "हिन्दी · Hindi" },
  { code: "ta-IN", label: "தமிழ் · Tamil" },
  { code: "te-IN", label: "తెలుగు · Telugu" },
  { code: "bn-IN", label: "বাংলা · Bengali" },
  { code: "mr-IN", label: "मराठी · Marathi" },
  { code: "en-IN", label: "English (India)" },
] as const;

export type LangCode = (typeof LANGUAGES)[number]["code"];

/** Precomputed demo vocabulary. Keys are cleaned INCLUDE class names. */
export const PHRASES: Record<string, Partial<Record<LangCode, string>>> = {
  Hello: { "hi-IN": "नमस्ते", "ta-IN": "வணக்கம்", "te-IN": "నమస్కారం", "bn-IN": "নমস্কার", "mr-IN": "नमस्कार", "en-IN": "Hello" },
  "Thank you": { "hi-IN": "धन्यवाद", "ta-IN": "நன்றி", "te-IN": "ధన్యవాదాలు", "bn-IN": "ধন্যবাদ", "mr-IN": "धन्यवाद", "en-IN": "Thank you" },
  Help: { "hi-IN": "मुझे मदद चाहिए", "ta-IN": "எனக்கு உதவி வேண்டும்", "te-IN": "నాకు సహాయం కావాలి", "bn-IN": "আমার সাহায্য দরকার", "mr-IN": "मला मदत हवी आहे", "en-IN": "I need help" },
  Water: { "hi-IN": "पानी", "ta-IN": "தண்ணீர்", "te-IN": "నీళ్ళు", "bn-IN": "জল", "mr-IN": "पाणी", "en-IN": "Water" },
  "Good Morning": { "hi-IN": "सुप्रभात", "ta-IN": "காலை வணக்கம்", "te-IN": "శుభోదయం", "bn-IN": "সুপ্রভাত", "mr-IN": "सुप्रभात", "en-IN": "Good morning" },
  "How are you": { "hi-IN": "आप कैसे हैं", "ta-IN": "நீங்கள் எப்படி இருக்கிறீர்கள்", "te-IN": "మీరు ఎలా ఉన్నారు", "bn-IN": "আপনি কেমন আছেন", "mr-IN": "तुम्ही कसे आहात", "en-IN": "How are you" },
};

/** Table first, gloss itself as fallback. Never blocks on a network call. */
export function phraseFor(gloss: string, lang: LangCode): string {
  return PHRASES[gloss]?.[lang] ?? gloss;
}

let voices: SpeechSynthesisVoice[] = [];
export function refreshVoices() {
  voices = window.speechSynthesis?.getVoices() ?? [];
  return voices;
}

export function voiceFor(lang: LangCode): SpeechSynthesisVoice | null {
  if (!voices.length) refreshVoices();
  const base = lang.split("-")[0];
  return (
    voices.find((v) => v.lang.replace("_", "-") === lang) ??
    voices.find((v) => v.lang.replace("_", "-").startsWith(`${base}-`)) ??
    null
  );
}

export function speak(text: string, lang: LangCode) {
  const synth = window.speechSynthesis;
  if (!synth) return;
  synth.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = lang;
  u.rate = 0.92;
  const v = voiceFor(lang);
  if (v) u.voice = v;
  synth.speak(u);
}
