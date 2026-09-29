/** ⌘K palette search helpers. */

import type { SearchResult } from "./api";
import { humanizeToken } from "./floor";

/** Census names arrive in filing caps ("BOYD GAMING CORP") — present them
 *  in title case, but leave names that already carry case untouched. */
export function presentRegistrantName(name: string | null): string | null {
  if (!name) return null;
  if (/[a-z]/.test(name)) return name;
  return name.toLowerCase().replace(/(^|[\s(-])[a-z]/g, (ch) => ch.toUpperCase());
}

/** One-line hint for a search result: the company name, then its watchlist
 *  status when tracked or its sector otherwise. */
export function searchResultHint(result: SearchResult): string {
  const parts = [presentRegistrantName(result.name)];
  parts.push(humanizeToken(result.presented_status ?? result.sector));
  return parts.filter(Boolean).join(" · ");
}

export interface PalettePage {
  title: string;
  hint: string;
  to: string;
}

