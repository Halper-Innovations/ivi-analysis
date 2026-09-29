const SEC_8K_DEFINITION =
  "an SEC filing companies must publish within days of a material corporate event";

/** Definitions keyed by the platform's raw state-machine tokens. */
export const GLOSSARY: Readonly<Record<string, string>> = {
  BUSTED_IPO:
    "an IPO from the last two years trading at least 50% below its debut close — a forced-seller setup",
  SPINOFF:
    "a business distributed to shareholders as a new standalone stock; early trading is often indiscriminate",
  CH11_EMERGENCE:
    "a company exiting bankruptcy with a restructured balance sheet and fresh equity",
  "8-K": SEC_8K_DEFINITION,
  "10-K": "the annual audited SEC filing",
  DCF: "discounted cash flow — value from projected cash flows",
  EPV: "earnings power value — value from current normalized earnings, no growth assumed",
  GRAHAM: "Benjamin Graham's conservative formula based on earnings and book value",
  MARGIN_OF_SAFETY: "the discount of price to estimated fair value",
  EVENT_PENDING:
    "a filed corporate event is awaiting owner review before the name is actionable",
  QUARANTINE: "held out of the actionable pool by a data-quality or structural gate",
  UNREVIEWED_8K: SEC_8K_DEFINITION,
};

const TOKEN_ALIASES: Readonly<Record<string, string>> = {
  "8K": "8-K",
  "10K": "10-K",
  CHAPTER_11_EMERGENCE: "CH11_EMERGENCE",
};

/** Look up raw API tokens case-insensitively, including EVENT_PENDING flags. */
export function glossaryDefinition(raw: string | null): string | undefined {
  if (!raw) return undefined;
  const token = raw.trim().replace(/^EVENT_PENDING:/i, "").toUpperCase();
  const snakeToken = token.replace(/[\s-]+/g, "_");
  const canonical = TOKEN_ALIASES[token] ?? TOKEN_ALIASES[snakeToken] ?? token;
  return GLOSSARY[canonical] ?? GLOSSARY[snakeToken];
}
