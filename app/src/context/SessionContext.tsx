import {
  createContext,
  useCallback,
  useContext,
  useState,
  type ReactNode,
} from "react";
import type { Language, Role } from "../lib/phrases";
export type Message = {
  id: string;
  role: Role;
  text: string;
  language: Language;
  source:
    "Phrase board" | "Typed text" | "Speech draft" | "Sign recognition draft";
  timestamp: string;
  phraseId?: string;
  service: string;
};
type Session = {
  messages: Message[];
  addMessage: (message: Omit<Message, "id" | "timestamp">) => void;
  resetSession: () => void;
};
const SessionContext = createContext<Session | null>(null);
export function SessionProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const addMessage = useCallback(
    (message: Omit<Message, "id" | "timestamp">) => {
      setMessages((previous) => [
        ...previous,
        {
          ...message,
          id: crypto.randomUUID(),
          timestamp: new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
        },
      ]);
    },
    [],
  );
  const resetSession = useCallback(() => setMessages([]), []);
  return (
    <SessionContext.Provider value={{ messages, addMessage, resetSession }}>
      {children}
    </SessionContext.Provider>
  );
}
export function useSession() {
  const context = useContext(SessionContext);
  if (!context) throw new Error("Open Setu inside its session provider.");
  return context;
}
