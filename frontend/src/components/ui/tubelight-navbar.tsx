import { motion, useReducedMotion } from "motion/react";
import type { MouseEvent } from "react";

export const workflowRoutes = [
  { id: "review", label: "Paper Review", href: "/paper-review" },
  { id: "idea", label: "Research Idea", href: "/research-idea" },
] as const;

export type Workflow = (typeof workflowRoutes)[number]["id"];

export function TubelightNavbar({ active, onNavigate }: {
  active: Workflow;
  onNavigate: (workflow: Workflow, href: string) => void;
}) {
  const reduceMotion = useReducedMotion();

  function navigate(event: MouseEvent<HTMLAnchorElement>, workflow: Workflow, href: string) {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return;
    event.preventDefault();
    onNavigate(workflow, href);
  }

  return (
    <nav className="tubelight-navbar" aria-label="PaperLens workflows">
      {workflowRoutes.map(({ id, label, href }) => (
        <a key={id} href={href} className="tubelight-item" aria-current={id === active ? "page" : undefined}
          onClick={event => navigate(event, id, href)}>
          {id === active && (
            <motion.span layoutId={reduceMotion ? undefined : "paperlens-tubelight"}
              initial={false} transition={{ type: "spring", stiffness: 380, damping: 32 }}
              className="tubelight-active" aria-hidden="true">
              <span className="tubelight-lamp" />
            </motion.span>
          )}
          <span className="tubelight-label">{label}</span>
        </a>
      ))}
    </nav>
  );
}
