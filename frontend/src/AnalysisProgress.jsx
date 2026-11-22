import { useEffect, useState } from "react";
﻿import { Check, CircleAlert } from "lucide-react";
import ReasoningText from "@/components/ui/reasoning-text";

function StepList({ steps, failed }) {
  return (
    <ol className="analysis-step-list">
      {steps.map((step, index) => (
        <li key={index}>
          {failed && index === steps.length - 1
            ? <CircleAlert size={14} aria-label="Stopped at" />
            : <Check size={14} aria-label="Completed" />}
          <span>{step}</span>
        </li>
      ))}
    </ol>
  );
}

function ActiveStep({ label }) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const started = performance.now();
    const timer = window.setInterval(() => setSeconds(Math.floor((performance.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const elapsed = seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return (
    <div className="analysis-active">
      <ReasoningText phrases={[label]} variant="swap" />
      <span aria-hidden="true" className="analysis-elapsed">{elapsed}</span>
    </div>
  );
}

export default function AnalysisProgress({ steps, running, failed }) {
  if (!steps.length) return null;
  if (!running) {
    return (
      <details className="analysis-progress">
        <summary>Processing steps</summary>
        <StepList steps={steps} failed={failed} />
      </details>
    );
  }
  return (
    <section className="analysis-progress" aria-label="Processing steps">
      <StepList steps={steps.slice(0, -1)} />
      <ActiveStep key={steps.length} label={steps.at(-1)} />
    </section>
  );
}
