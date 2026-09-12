import { Activity, Cpu, Database, Flame, Satellite, ShieldCheck } from "lucide-react";

const signalPoints = "0,62 18,58 36,61 54,46 72,51 90,35 108,41 126,28 144,32 162,18 180,24 198,13";

export function OverviewSignalRail() {
  return (
    <aside className="overview-signal-rail panel">
      <div className="signal-rail-header">
        <div>
          <span className="section-kicker"><Activity size={13} /> LIVE SIGNAL BOARD</span>
          <h3>Mine intelligence pulse</h3>
        </div>
        <span className="signal-live"><i /> LIVE</span>
      </div>

      <div className="signal-orbit-card">
        <div className="signal-orbit orbit-a" />
        <div className="signal-orbit orbit-b" />
        <div className="signal-orbit orbit-c" />
        <div className="signal-core"><span>M</span><small>AI</small></div>
        <span className="signal-node node-a" />
        <span className="signal-node node-b" />
        <span className="signal-node node-c" />
        <div className="signal-orbit-label">MANGAI CORE</div>
      </div>

      <div className="signal-chart-card">
        <div className="signal-chart-head">
          <div><span>Operational signal</span><strong>+18.4%</strong></div>
          <small>7D TREND</small>
        </div>
        <svg viewBox="0 0 198 74" preserveAspectRatio="none" aria-label="Operational signal trend">
          <defs>
            <linearGradient id="signalFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#48d9aa" stopOpacity=".32" />
              <stop offset="100%" stopColor="#48d9aa" stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d={`M ${signalPoints} L 198 74 L 0 74 Z`} fill="url(#signalFill)" />
          <polyline points={signalPoints} fill="none" stroke="#58dfb3" strokeWidth="2.2" vectorEffect="non-scaling-stroke" />
          <circle cx="198" cy="13" r="3.5" fill="#f2ca68" />
        </svg>
        <div className="signal-axis"><span>−6d</span><span>today</span></div>
      </div>

      <div className="thermal-proxy-card">
        <div className="signal-chart-head">
          <div><span><Flame size={13} /> Thermal surface</span><strong>32.8°C</strong></div>
          <small>DEMO PROXY</small>
        </div>
        <div className="thermal-ramp"><i /><i /><i /><i /><i /><i /><i /></div>
        <div className="thermal-values"><span>24°</span><span>28°</span><span>32°</span><span>36°</span><span>40°+</span></div>
        <p>Visual proxy only until spatial Landsat surface-temperature data is connected.</p>
      </div>

      <div className="signal-health-grid">
        <SignalStatus icon={Satellite} label="Satellite" value="SYNC" tone="ok" />
        <SignalStatus icon={Database} label="Data quality" value="100%" tone="ok" />
        <SignalStatus icon={Cpu} label="Models" value="READY" tone="ok" />
        <SignalStatus icon={ShieldCheck} label="Safety" value="GUARDED" tone="gold" />
      </div>
    </aside>
  );
}

function SignalStatus({ icon: Icon, label, value, tone }: { icon: typeof Satellite; label: string; value: string; tone: "ok" | "gold" }) {
  return (
    <div className="signal-status">
      <Icon size={14} />
      <span>{label}</span>
      <strong className={tone}>{value}</strong>
    </div>
  );
}
