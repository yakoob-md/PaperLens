import type { CSSProperties } from "react";

export const TEXT_SHIMMER_CLASS_NAME = "reasoning-text-shimmer";
export const TEXT_SHIMMER_KEYFRAMES = `
  @keyframes reasoning-shimmer {
    from { background-position: 140% 0; }
    to { background-position: -40% 0; }
  }
`;

export function textShimmerStyle(duration: number): CSSProperties {
  return { animationDuration: `${duration}s` };
}
