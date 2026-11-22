import React from "react";
import { createRoot } from "react-dom/client";
import { AuroraBackground } from "@/components/ui/aurora-background";
import App from "./App.jsx";
import "./style.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <AuroraBackground>
      <App />
    </AuroraBackground>
  </React.StrictMode>,
);
