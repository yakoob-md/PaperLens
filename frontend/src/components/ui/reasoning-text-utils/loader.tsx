export function Loader({ size, speed }: {
  variant: "ascii-line";
  size: number;
  speed: number;
  label: string;
}) {
  return <span aria-hidden="true" className="reasoning-ascii"
    style={{ fontSize: size, animationDuration: `${speed}s` }}>|</span>;
}
