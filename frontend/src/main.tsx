import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";
import "./phase12-polish.css";
import "./final-command-center.css";
import "./premium-dashboard.css";
import "./final-visual-overrides.css";
import "./reserve-map-view.css";
import "./global-contrast.css";
import "./ultimate-contrast.css";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("MANGAI frontend root element (#root) was not found.");
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
