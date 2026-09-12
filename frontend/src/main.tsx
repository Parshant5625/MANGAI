import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { MangaiAssistant } from "./components/MangaiAssistant";
import { MangaiChrome } from "./components/navigation/MangaiChrome";
import "./styles.css";
import "./phase12-polish.css";
import "./final-command-center.css";
import "./premium-dashboard.css";
import "./final-visual-overrides.css";
import "./reserve-map-view.css";
import "./global-contrast.css";
import "./ultimate-contrast.css";
import "./final-theme.css";
import "./mangai-assistant.css";
import "./styles/mangai-command-center.css";
import "./styles/mangai-shell.css";
import "./styles/mangai-zero-shell.css";
import "./styles/command-center-rebuild.css";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("MANGAI frontend root element (#root) was not found.");
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <MangaiChrome />
    <App />
    <MangaiAssistant />
  </React.StrictMode>,
);
