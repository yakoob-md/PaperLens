import type { ComponentPropsWithoutRef } from "react";
import { cn } from "@/lib/utils";

interface AuroraBackgroundProps extends ComponentPropsWithoutRef<"div"> {
  showRadialGradient?: boolean;
}

export function AuroraBackground({
  children,
  className,
  showRadialGradient = true,
  ...props
}: AuroraBackgroundProps) {
  return (
    <div className={cn("aurora-background", className)} {...props}>
      <div className="aurora-background__viewport" aria-hidden="true">
        <div className={cn(
          "aurora-background__lights",
          showRadialGradient && "aurora-background__lights--masked",
        )} />
      </div>
      {children}
    </div>
  );
}
