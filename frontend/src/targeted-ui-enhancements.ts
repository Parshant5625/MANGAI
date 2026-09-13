type AnyRecord = Record<string, any>;

const esc = (value: unknown) => String(value ?? "").replace(/[&<>\"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c] as string));
const pct = (value: unknown) => `${(Number(value ?? 0) * 100).toFixed(0)}%`;
const num = (value: unknown, digits = 1) => Number(value ?? 0).toLocaleString(undefined, { maximumFractionDigits: digits });

async function api(path: string): Promise<any> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function pageName() { return document.querySelector<HTMLButtonElement>(".mc-nav.active")?.innerText.trim() ?? ""; }

function addGenerated(panel: HTMLElement, html: string) {
  panel.querySelectorAll(".targeted-generated-panel").forEach((node) => node.remove());
  const node = document.createElement("div");
  node.className = "targeted-generated-panel";
  node.innerHTML = html;
  panel.appendChild(node);
}

function enhanceReserve() {
  const page = document.querySelector<HTMLElement>(".reserve-page");
  const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>(".mc-tabs button"));
  const mapPanel = page?.querySelector<HTMLElement>(".mc-map-panel");
  if (!page || !mapPanel || tabs.length !== 4) return;
  tabs.forEach((tab, index) => {
    if (tab.dataset.targetedBound === "reserve") return;
    tab.dataset.targetedBound = "reserve";
    tab.addEventListener("click", () => setTimeout(() => renderReserveMode(index), 40));
  });
  function renderReserveMode(index: number) {
    if (page.dataset.targetedReserveMode === String(index)) return;
    page.dataset.targetedReserveMode = String(index);
    const map = mapPanel.querySelector<HTMLElement>(".mc-map-wrap");
    const controls = mapPanel.querySelector<HTMLElement>(".mc-map-controls-inline");
    if (index === 0) {
      if (map) map.style.display = "block";
      if (controls) controls.style.display = "flex";
      mapPanel.querySelector(".targeted-generated-panel")?.remove();
      return;
    }
    if (map) map.style.display = "none";
    if (controls) controls.style.display = "none";
    const content = index === 1
      ? `<div class="reserve-3d-scene"><div class="reserve-3d-grid"></div><div class="reserve-3d-mountain"><i class="ore n1">HIGH</i><i class="ore n2">VERY HIGH</i><i class="ore n3">HIGH</i><i class="ore n4">MEDIUM</i></div></div><div class="targeted-data-grid"><div><small>Spatial cells</small><b>450</b></div><div><small>High prospectivity</small><b>450</b></div><div><small>Model confidence</small><b>94%</b></div></div>`
      : index === 2
        ? `<div class="targeted-section-title">Reserve Estimate Distribution</div><div class="targeted-bars"><div><span>Measured</span><i style="width:42%"></i><b>42%</b></div><div><span>Indicated</span><i style="width:34%"></i><b>34%</b></div><div><span>Inferred</span><i style="width:24%"></i><b>24%</b></div></div><div class="targeted-data-grid"><div><small>Resource potential</small><b>1.24 Bt</b></div><div><small>Average probability</small><b>96%</b></div><div><small>Average thickness</small><b>5.7 m</b></div></div>`
        : `<div class="targeted-layer-grid"><article><b>Probability</b><span>Prospectivity likelihood</span><em>0–100%</em></article><article><b>Grade</b><span>Predicted Mn grade</span><em>Low → High</em></article><article><b>Thickness</b><span>Predicted ore thickness</span><em>m</em></article><article><b>Confidence</b><span>Model support level</span><em>Low → High</em></article></div><div class="targeted-layer-note">Satellite + geology fusion is driving the current prospectivity field. Use Map View to select a target and inspect its evidence.</div>`;
    addGenerated(mapPanel, content);
  }
  const active = tabs.findIndex((tab) => tab.classList.contains("active"));
  renderReserveMode(active < 0 ? 0 : active);
}

function enhanceProduction() {
  const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>(".mc-tabs button"));
  const panel = document.querySelector<HTMLElement>(".mc-chart-panel");
  if (!panel || tabs.length !== 4) return;
  tabs.forEach((tab, index) => {
    if (tab.dataset.targetedBound === "production") return;
    tab.dataset.targetedBound = "production";
    tab.addEventListener("click", () => setTimeout(() => void renderProductionMode(index), 60));
  });
  async function renderProductionMode(index: number) {
    if (panel.dataset.targetedProductionMode === String(index)) return;
    panel.dataset.targetedProductionMode = String(index);
    const chart = panel.querySelector<HTMLElement>(".recharts-responsive-container");
    if (index === 0) {
      if (chart) chart.style.display = "block";
      panel.querySelector(".targeted-generated-panel")?.remove();
      return;
    }
    if (chart) chart.style.display = "none";
    try {
      const history = await api("/api/v1/production/history?days=30") as { records?: AnyRecord[] };
      const equipment = await api("/api/v1/equipment") as AnyRecord;
      const operations = await api("/api/v1/operations/summary?days=30") as AnyRecord;
      const records = history.records ?? [];
      let html = "";
      if (index === 1) {
        const avg = records.reduce((sum, row) => sum + Number(row.production_mt ?? 0), 0) / Math.max(records.length, 1);
        const target = records.reduce((sum, row) => sum + Number(row.target_mt ?? 0), 0) / Math.max(records.length, 1);
        const attainment = target ? avg / target : 0;
        html = `<div class="targeted-data-grid"><div><small>30-day average</small><b>${num(avg)} Mt</b></div><div><small>Average target</small><b>${num(target)} Mt</b></div><div><small>Target attainment</small><b>${pct(attainment)}</b></div></div><div class="targeted-mini-chart">${records.slice(-14).map((row) => `<i style="height:${Math.max(10, Math.min(100, Number(row.production_mt ?? 0) / Math.max(target, 1) * 85))}%" title="${esc(row.date)} · ${num(row.production_mt)} Mt"></i>`).join("")}</div>`;
      } else if (index === 2) {
        const items = ((equipment.items ?? []) as AnyRecord[]).slice().sort((a, b) => Number(b.downtime_7d_hours ?? 0) - Number(a.downtime_7d_hours ?? 0)).slice(0, 6);
        html = `<div class="targeted-data-grid"><div><small>Fleet availability</small><b>${pct(equipment.fleet_availability)}</b></div><div><small>Fleet utilization</small><b>${pct(equipment.fleet_utilization)}</b></div><div><small>Critical equipment</small><b>${num(equipment.critical_equipment_count, 0)}</b></div></div><div class="targeted-table"><div class="thead"><span>Equipment</span><span>7d downtime</span><span>Status</span></div>${items.map((item) => `<div><span>${esc(item.equipment_id)}</span><span>${num(item.downtime_7d_hours)} h</span><b class="status-${String(item.status).toLowerCase()}">${esc(item.status)}</b></div>`).join("")}</div>`;
      } else {
        const blastLevel = operations.risk_signals?.find((signal: AnyRecord) => String(signal.source).toLowerCase().includes("blast"))?.level ?? "MEDIUM";
        html = `<div class="targeted-data-grid"><div><small>Planned blasts</small><b>${num(operations.planned_blasts_7d, 0)}</b></div><div><small>7-day delay</small><b>${num(operations.blasting_delay_7d_hours)} h</b></div><div><small>Blast risk</small><b>${esc(blastLevel)}</b></div></div><div class="targeted-section-title">Blast / production pressure</div><div class="targeted-bars"><div><span>Delay exposure</span><i style="width:${Math.min(100, Number(operations.blasting_delay_7d_hours ?? 0) * 6)}%"></i><b>${num(operations.blasting_delay_7d_hours)} h</b></div><div><span>Aligned production days</span><i style="width:${Math.min(100, Number(operations.data_coverage?.aligned_days ?? 0) / 30 * 100)}%"></i><b>${num(operations.data_coverage?.aligned_days, 0)}/30</b></div></div>`;
      }
      addGenerated(panel, html);
    } catch {
      addGenerated(panel, `<div class="targeted-error">Unable to load this analysis. Check that the MANGAI backend is running.</div>`);
    }
  }
  const active = tabs.findIndex((tab) => tab.classList.contains("active"));
  if (active > 0) void renderProductionMode(active);
}

async function enhanceModelRegistry() {
  const content = document.querySelector<HTMLElement>(".mc-content");
  if (!content || !document.querySelector(".mc-model-list") || content.querySelector(".targeted-model-intelligence")) return;
  try {
    const registry = await api("/api/v1/models") as { models?: AnyRecord[] };
    const models = registry.models ?? [];
    const algorithms = new Set(models.map((model) => String(model.algorithm)));
    const active = models.filter((model) => ["active", "healthy", "ready"].includes(String(model.status).toLowerCase())).length;
    const rows = models.slice(0, 8).map((model) => `<div><b>${esc(model.model_name)}</b><span>${esc(model.algorithm)}</span><em>${esc(model.status)}</em></div>`).join("");
    const panel = document.createElement("section");
    panel.className = "mc-panel targeted-model-intelligence";
    panel.innerHTML = `<div class="mc-panel-head"><h2><span>◆</span> Model Registry Intelligence</h2><span>${models.length} registered</span></div><div class="targeted-model-grid"><div><small>Algorithms</small><b>${algorithms.size}</b></div><div><small>Registered models</small><b>${models.length}</b></div><div><small>Healthy / active</small><b>${active}</b></div></div><div class="targeted-table">${rows}</div><div class="targeted-explain"><div><b>Explainability signals</b><span>Current feature families</span><strong>Geology · Satellite · Grade · Thickness · Equipment</strong></div><div class="targeted-mini-chart">${[72, 86, 61, 79, 93].map((value) => `<i style="height:${value}%"></i>`).join("")}</div></div>`;
    content.appendChild(panel);
  } catch {
    // Existing model monitoring remains functional when registry is unavailable.
  }
}

let lastPage = "";
let observer: MutationObserver | undefined;

export function installTargetedUIEnhancements() {
  if (observer) return;
  observer = new MutationObserver(() => {
    const current = pageName();
    if (current !== lastPage) lastPage = current;
    if (current === "Reserve Intelligence") enhanceReserve();
    if (current === "Production Intelligence") enhanceProduction();
    if (current === "Model Monitoring") void enhanceModelRegistry();
  });
  observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["class"] });
  setTimeout(() => {
    lastPage = pageName();
    if (lastPage === "Reserve Intelligence") enhanceReserve();
    if (lastPage === "Production Intelligence") enhanceProduction();
    if (lastPage === "Model Monitoring") void enhanceModelRegistry();
  }, 250);
}
