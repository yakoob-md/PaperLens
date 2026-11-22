import { useReducedMotion } from "motion/react";
import { useEffect, useState, type CSSProperties } from "react";

const alphabet = "/-+*?";

export function TextScramble({ text, className, style }: {
  text: string;
  className?: string;
  style?: CSSProperties;
}) {
  const reduce = useReducedMotion();
  const [visible, setVisible] = useState(text);

  useEffect(() => {
    if (reduce) return;
    let frame = 0;
    const timer = window.setInterval(() => {
      frame += 1;
      setVisible(Array.from(text, (character, index) =>
        character === " " || index * 8 <= frame * 32
          ? character
          : alphabet[(index + frame) % alphabet.length],
      ).join(""));
      if (frame * 4 >= text.length) window.clearInterval(timer);
    }, 32);
    return () => window.clearInterval(timer);
  }, [text, reduce]);

  return <span className={className} style={style}>{reduce ? text : visible}</span>;
}
