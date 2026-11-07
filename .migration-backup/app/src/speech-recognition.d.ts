/**
 * Web Speech API types.
 *
 * SpeechRecognition is still not in TypeScript's DOM lib because the spec never
 * reached Recommendation: it ships prefixed in Chrome/Edge/Safari and is
 * absent in Firefox. Declaring only what we use, rather than pulling a
 * dependency for four members.
 */
interface SpeechRecognitionAlternative {
  readonly transcript: string;
  readonly confidence: number;
}
interface SpeechRecognitionResult {
  readonly isFinal: boolean;
  readonly length: number;
  [index: number]: SpeechRecognitionAlternative;
}
interface SpeechRecognitionResultList {
  readonly length: number;
  [index: number]: SpeechRecognitionResult;
}
interface SpeechRecognitionEvent extends Event {
  readonly resultIndex: number;
  readonly results: SpeechRecognitionResultList;
}
interface SpeechRecognitionErrorEvent extends Event {
  readonly error: string;
  readonly message: string;
}
interface SpeechRecognition extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((e: SpeechRecognitionEvent) => void) | null;
  onerror: ((e: SpeechRecognitionErrorEvent) => void) | null;
  onend: ((e: Event) => void) | null;
  onstart: ((e: Event) => void) | null;
}
declare const SpeechRecognition: { new (): SpeechRecognition } | undefined;
declare const webkitSpeechRecognition: { new (): SpeechRecognition } | undefined;
