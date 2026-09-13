import React from "react";
import ReactDOM from "react-dom/client";
import { ReferenceDashboard } from "./components/ReferenceDashboard";
import "./styles.css";
import "./phase12-polish.css";
import "./reference-dashboard.css";
import "./reference-dashboard-layout.css";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("MANGAI frontend root element (#root) was not found.");
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <ReferenceDashboard />
  </React.StrictMode>,
);
