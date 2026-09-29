import type { CSSProperties, HTMLAttributes } from "react";

interface PanelOptions {
  /** Load-choreography stagger index (40ms steps). */
  condenseIndex?: number;
}

type GlassPanelProps = HTMLAttributes<HTMLDivElement> & PanelOptions;
type SectionProps = HTMLAttributes<HTMLElement> & PanelOptions;

/** Card-tier glass surface with the condense-in load animation. */
export function GlassPanel({ children, condenseIndex, className = "", ...rest }: GlassPanelProps) {
  return (
    <div
      className={`glass condense ${className}`}
      style={
        condenseIndex !== undefined
          ? ({ ["--condense-i" as string]: condenseIndex } as CSSProperties)
          : undefined
      }
      {...rest}
    >
      {children}
    </div>
  );
}

/** Flat page section with a top hairline and the shared load animation. */
export function Section({ children, condenseIndex, className = "", ...rest }: SectionProps) {
  return (
    <section
      className={`section condense ${className}`}
      style={
        condenseIndex !== undefined
          ? ({ ["--condense-i" as string]: condenseIndex } as CSSProperties)
          : undefined
      }
      {...rest}
    >
      {children}
    </section>
  );
}
