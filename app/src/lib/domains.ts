/** Shared communication tools across tertiary-service settings. */
import { services, type ServiceId } from "./phrases";
export type DomainId = ServiceId;
export interface QuickPhrase {
  glosses: string[];
  caption: string;
  urgent?: boolean;
}
export interface Domain {
  id: DomainId;
  label: string;
  tagline: string;
  station: string;
  quick: QuickPhrase[];
  synonyms: Record<string, string>;
}
export const DOMAIN_LIST: Domain[] = services.map((service) => ({
  id: service.id,
  label: service.label,
  tagline: "Accessible two-way service communication",
  station: service.label,
  quick: [
    { glosses: ["Hello"], caption: "Hello" },
    { glosses: ["Thank you"], caption: "Thank you" },
  ],
  synonyms: {},
}));
export const DOMAINS = Object.fromEntries(
  DOMAIN_LIST.map((domain) => [domain.id, domain]),
) as Record<DomainId, Domain>;
export const getDomain = (): DomainId => "general";
export const getServerDomain = getDomain;
export const subscribeDomain = (_listener: () => void) => () => {};
export const saveDomain = (_id: DomainId) => {};
