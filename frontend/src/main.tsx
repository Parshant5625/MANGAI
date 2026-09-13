import React from "react";
import ReactDOM from "react-dom/client";
import { MANGAICommandCenter } from "./components/MANGAICommandCenter";
import { installTargetedUIEnhancements } from "./targeted-ui-enhancements";
import "./styles.css";
import "./phase12-polish.css";
import "./reference-dashboard.css";
import "./reference-dashboard-layout.css";
import "./command-center-reference.css";
import "./styles/mangai-production-terminal.css";
import "./targeted-ui-fixes.css";
import "./styles/model-monitoring-targeted.css";
import "maplibre-gl/dist/maplibre-gl.css";

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("MANGAI frontend root element (#root) was not found.");

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <MANGAICommandCenter />
  </React.StrictMode>,
);

installTargetedUIEnhancements();
