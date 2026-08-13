import type { Domain, QuickPhrase } from "../lib/domains";

/**
 * Tap-to-say phrases for the current setting.
 *
 * This is the Deaf person's side of the bridge when signing is not practical —
 * hands full, injured, holding luggage, or simply faster to tap. It runs the
 * FORWARD direction: the phrase is spoken aloud to the hearing person, exactly
 * as a recognised sign would be.
 *
 * That means playability does not apply here. These glosses never need bundled
 * pose frames; they only need to be things the person wants to say. (The reverse
 * direction is where playability matters — see the library guard in reverse.ts.)
 */
export default function QuickPhrases({
  domain,
  onSay,
  disabled = false,
}: {
  domain: Domain;
  onSay: (phrase: QuickPhrase) => void;
  disabled?: boolean;
}) {
  return (
    <section className="card quick">
      <div className="card-h">
        <span>Tap to say</span>
        <span className="mono">{domain.station}</span>
      </div>
      <div className="quick-grid">
        {domain.quick.map((q) => (
          <button
            key={q.glosses.join("|")}
            className={q.urgent ? "qp urgent" : "qp"}
            onClick={() => onSay(q)}
            disabled={disabled}
            title={q.glosses.join(" · ")}
          >
            <span className="qp-caption">{q.caption}</span>
            <span className="qp-gloss">{q.glosses.join(" · ")}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
