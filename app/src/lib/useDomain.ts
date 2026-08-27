import { useSyncExternalStore } from "react";
import {
  DOMAINS, getDomain, getServerDomain, saveDomain, subscribeDomain,
  type Domain, type DomainId, type RoleId,
} from "./domains";

/**
 * The active setting, as a React value.
 *
 * domains.ts has carried a subscribable store since the Travel submission and
 * nothing ever read it: every caller passed the literal "health", so a shop
 * till rendered as a hospital and the claim that one build serves every sector
 * was true of the model and false of the interface. This is the missing wire.
 *
 * useSyncExternalStore rather than an effect, because the store reads
 * localStorage, which is unavailable during prerender and throws in some
 * privacy modes. getServerDomain keeps the first paint on a safe default and
 * the client adopts the stored value without a cascading re-render.
 */
export function useDomain(): Domain {
  const id = useSyncExternalStore(subscribeDomain, getDomain, getServerDomain);
  return DOMAINS[id];
}

export function useSetDomain(): (id: DomainId) => void {
  return saveDomain;
}

/** What this setting calls the person on a given side of the counter. */
export function roleLabel(domain: Domain, role: RoleId): string {
  return domain.roles[role].label;
}

export function roleLower(domain: Domain, role: RoleId): string {
  return domain.roles[role].lower;
}
