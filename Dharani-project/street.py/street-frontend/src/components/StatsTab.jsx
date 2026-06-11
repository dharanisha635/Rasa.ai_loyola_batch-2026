import { useEffect, useState } from "react";
import { fetchStats } from "../api";

function Tooltip({ text, x, y, visible }) {
  if (!visible) return null;
  return (
    <div style={{
      position:"fixed", left:x+12, top:y-10,
      background:"#1e293b", border:"1px solid #334155",
      borderRadius:6, padding:"6px 10px", fontSize:12,
      color:"#e2e8f0", pointerEvents:"none", zIndex:1000,
      boxShadow:"0 4px 12px rgba(0,0,0,0.3)", whiteSpace:"nowrap"
    }}>{text}</div>
  );
}

function TestCard({ name, stat, verdict, detail, cls }) {
  const [hov, setHov] = useState(false);
  return (
    <div className={`test-card ${cls}`}
      style={{ cursor:"pointer", transition:"all 0.2s",
               transform: hov ? "translateY(-2px)" : "none" }}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}>
      <div className="test-name">{name}</div>
      <div className="test-stat">{stat}</div>
      <div className="test-verdict">{verdict}</div>
      {hov && (
        <div style={{ fontSize:11, color:"var(--color-text-secondary)",
                      marginTop:6, lineHeight:1.5 }}>{detail}</div>
      )}
    </div>
  );
}

function StatCard({ label, value, detail, color }) {
  const [hov, setHov] = useState(false);
  return (
    <div className="stat-card"
      style={{ cursor:"pointer", transition:"all 0.2s",
               transform: hov ? "translateY(-2px)" : "none",
               boxShadow: hov ? "0 4px 20px rgba(0,0,0,0.2)" : "none" }}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}>
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ color: color || "#818cf8" }}>{value}</div>
      {hov && detail && (
        <div style={{ fontSize:11, color:"var(--color-text-secondary)",
                      marginTop:6, lineHeight:1.5 }}>{detail}</div>
      )}
    </div>
  );
}

function HoverBar({ label, value, count, extraInfo }) {
  const [hov, setHov] = useState(false);
  const [pos, setPos] = useState({ x:0, y:0 });
  return (
    <div style={{ display:"flex", alignItems:"center", gap:8, cursor:"pointer",
                  padding:"4px 6px", borderRadius:6, transition:"background 0.15s",
                  background: hov ? "var(--color-background-secondary)" : "transparent" }}
      onMouseEnter={e => { setHov(true); setPos({ x:e.clientX, y:e.clientY }); }}
      onMouseMove={e  => setPos({ x:e.clientX, y:e.clientY })}
      onMouseLeave={()  => setHov(false)}>
      <span style={{ fontSize:11, color:"var(--color-text-secondary)",
                     width:60, flexShrink:0 }}>{label}</span>
      <div style={{ flex:1, height:18, background:"var(--color-background-secondary)",
                    borderRadius:4, overflow:"hidden", position:"relative" }}>
        <div style={{
          height:"100%", width:`${value}%`, background:"#4ade80",
          borderRadius:4, transition:"width 0.4s ease",
          minWidth: value > 0 ? 2 : 0
        }}/>
        <div style={{ position:"absolute", top:"50%", left:8,
                      transform:"translateY(-50%)", fontSize:11,
                      color:"#fff", fontWeight:500 }}>{value}%</div>
      </div>
      <span style={{ fontSize:11, color:"var(--color-text-secondary)",
                     width:34, textAlign:"right" }}>n={count}</span>
      <Tooltip text={extraInfo} x={pos.x} y={pos.y} visible={hov}/>
    </div>
  );
}

function DonutChart({ cleanCount, dirtyCount, total, accuracy }) {
  const [hov, setHov] = useState(null);
  const cleanDash = (cleanCount / total) * 100;
  const dirtyDash = (dirtyCount / total) * 100;
  return (
    <div style={{ display:"flex", alignItems:"center", gap:"2rem", padding:"1rem 0" }}>
      <div style={{ position:"relative", width:130, height:130, flexShrink:0 }}>
        <svg viewBox="0 0 36 36"
             style={{ transform:"rotate(-90deg)", width:130, height:130 }}>
          <circle cx="18" cy="18" r="15.9" fill="none"
            stroke="var(--color-border-tertiary)" strokeWidth="3"/>
          <circle cx="18" cy="18" r="15.9" fill="none"
            stroke={hov==="clean" ? "#86efac" : "#4ade80"}
            strokeWidth={hov==="clean" ? "4.5" : "3"}
            strokeDasharray={`${cleanDash} ${100-cleanDash}`}
            strokeDashoffset="0"
            style={{ cursor:"pointer", transition:"all 0.2s" }}
            onMouseEnter={() => setHov("clean")}
            onMouseLeave={() => setHov(null)}/>
          <circle cx="18" cy="18" r="15.9" fill="none"
            stroke={hov==="dirty" ? "#fca5a5" : "#f87171"}
            strokeWidth={hov==="dirty" ? "4.5" : "3"}
            strokeDasharray={`${dirtyDash} ${100-dirtyDash}`}
            strokeDashoffset={`-${cleanDash}`}
            style={{ cursor:"pointer", transition:"all 0.2s" }}
            onMouseEnter={() => setHov("dirty")}
            onMouseLeave={() => setHov(null)}/>
        </svg>
        <div style={{ position:"absolute", top:"50%", left:"50%",
                      transform:"translate(-50%,-50%)", textAlign:"center" }}>
          {hov==="clean" && <>
            <div style={{ fontSize:16, fontWeight:500, color:"#4ade80" }}>{cleanCount}</div>
            <div style={{ fontSize:10, color:"var(--color-text-secondary)" }}>Clean</div>
          </>}
          {hov==="dirty" && <>
            <div style={{ fontSize:16, fontWeight:500, color:"#f87171" }}>{dirtyCount}</div>
            <div style={{ fontSize:10, color:"var(--color-text-secondary)" }}>Dirty</div>
          </>}
          {!hov && <>
            <div style={{ fontSize:16, fontWeight:500,
                          color:"var(--color-text-primary)" }}>{total}</div>
            <div style={{ fontSize:10,
                          color:"var(--color-text-secondary)" }}>total</div>
          </>}
        </div>
      </div>
      <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
        {[
          { key:"clean", color:"#4ade80", hbg:"rgba(74,222,128,0.1)",
            label:`Clean — ${cleanCount} (${((cleanCount/total)*100).toFixed(1)}%)` },
          { key:"dirty", color:"#f87171", hbg:"rgba(248,113,113,0.1)",
            label:`Dirty — ${dirtyCount} (${((dirtyCount/total)*100).toFixed(1)}%)` },
        ].map(item => (
          <div key={item.key}
            style={{ display:"flex", alignItems:"center", gap:8, padding:"6px 10px",
                     borderRadius:6, cursor:"pointer", transition:"background 0.2s",
                     background: hov===item.key ? item.hbg : "transparent" }}
            onMouseEnter={() => setHov(item.key)}
            onMouseLeave={() => setHov(null)}>
            <div style={{ width:12, height:12, borderRadius:2, background:item.color }}/>
            <span style={{ fontSize:13,
                           color:"var(--color-text-secondary)" }}>{item.label}</span>
          </div>
        ))}
        <div style={{ fontSize:12, color:"var(--color-text-secondary)",
                      paddingLeft:10,
                      borderLeft:"2px solid var(--color-border-tertiary)" }}>
          Accuracy:{" "}
          <span style={{ color:"#818cf8", fontWeight:500 }}>{accuracy}%</span>
        </div>
      </div>
    </div>
  );
}

function UncertaintyBar({ item }) {
  const [hov, setHov] = useState(false);
  const [pos, setPos] = useState({ x:0, y:0 });
  const fname = item.filename ? item.filename.split(/[\\/]/).pop() : "image";
  const color = item.correct ? "#4ade80" : "#f87171";
  return (
    <div style={{ position:"relative" }}
      onMouseEnter={e => { setHov(true); setPos({ x:e.clientX, y:e.clientY }); }}
      onMouseMove={e  => setPos({ x:e.clientX, y:e.clientY })}
      onMouseLeave={()  => setHov(false)}>
      <div style={{
        height: hov ? 13 : 9, borderRadius:2, minWidth:2,
        width:`${(item.uncertainty_std / 0.15) * 100}%`,
        background: hov ? (item.correct ? "#86efac" : "#fca5a5") : color,
        transition:"all 0.15s", cursor:"pointer"
      }}/>
      <Tooltip
        text={`${fname} | std=${item.uncertainty_std.toFixed(3)} | entropy=${item.predictive_entropy?.toFixed(3) ?? "—"} | ${item.correct ? "✓ Correct" : "✗ Wrong"}`}
        x={pos.x} y={pos.y} visible={hov}/>
    </div>
  );
}

function PlotImage({ src, label }) {
  const [hov,     setHov]     = useState(false);
  const [expanded,setExpanded]= useState(false);
  return (
    <>
      <div
        style={{ borderRadius:8, overflow:"hidden", cursor:"pointer",
                 border:`1px solid ${hov ? "#818cf8" : "var(--color-border-tertiary)"}`,
                 transition:"border-color 0.2s",
                 transform: hov ? "translateY(-2px)" : "none",
                 boxShadow: hov ? "0 4px 20px rgba(0,0,0,0.3)" : "none" }}
        onMouseEnter={() => setHov(true)}
        onMouseLeave={() => setHov(false)}
        onClick={() => setExpanded(true)}>
        <img src={src} alt={label} style={{ width:"100%", display:"block" }}/>
        <div style={{ padding:"6px 10px", fontSize:11,
                      color:"var(--color-text-secondary)",
                      background:"var(--color-background-secondary)",
                      display:"flex", justifyContent:"space-between" }}>
          <span>{label}</span>
          <span style={{ color:"#818cf8" }}>click to expand</span>
        </div>
      </div>

      {expanded && (
        <div
          style={{ position:"fixed", inset:0, background:"rgba(0,0,0,0.85)",
                   zIndex:2000, display:"flex", alignItems:"center",
                   justifyContent:"center", padding:"2rem" }}
          onClick={() => setExpanded(false)}>
          <div style={{ maxWidth:"90vw", maxHeight:"90vh", position:"relative" }}>
            <img src={src} alt={label}
                 style={{ maxWidth:"100%", maxHeight:"85vh",
                          borderRadius:8, display:"block" }}/>
            <div style={{ textAlign:"center", marginTop:10, fontSize:13,
                          color:"#94a3b8" }}>{label}</div>
            <div style={{ textAlign:"center", fontSize:12,
                          color:"#64748b", marginTop:4 }}>
              Click anywhere to close
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default function StatsTab() {
  const [stats,   setStats]   = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    fetchStats()
      .then(setStats)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="tab-container">
      <p style={{ color:"var(--color-text-secondary)" }}>Loading statistics…</p>
    </div>
  );
  if (error) return (
    <div className="tab-container">
      <div className="error-box">⚠ {error} — make sure Flask is running</div>
    </div>
  );
  if (!stats?.summary) return (
    <div className="tab-container">
      <p style={{ color:"var(--color-text-secondary)" }}>
        No stats data. Run evaluate.py first.
      </p>
    </div>
  );

  const summary    = stats.summary;
  const tests      = stats.tests;
  const unc        = stats.uncertainty;
  const correct    = summary.filter(d => d.correct).length;
  const total      = summary.length;
  const cleanCount = summary.filter(d => d.predicted === "Clean").length;
  const dirtyCount = summary.filter(d => d.predicted === "Dirty").length;
  const accuracy   = (correct / total * 100).toFixed(1);

  const calBins = [
    { label:"0.5–0.6", lo:0.5,  hi:0.6  },
    { label:"0.6–0.7", lo:0.6,  hi:0.7  },
    { label:"0.7–0.8", lo:0.7,  hi:0.8  },
    { label:"0.8–0.9", lo:0.8,  hi:0.91 },
  ].map(b => {
    const inBin   = summary.filter(d => d.confidence >= b.lo && d.confidence < b.hi);
    const acc     = inBin.length
      ? Math.round(inBin.filter(d => d.correct).length / inBin.length * 100)
      : 0;
    const avgConf = inBin.length
      ? (inBin.reduce((s,d) => s+d.confidence, 0) / inBin.length * 100).toFixed(1)
      : 0;
    return { ...b, count:inBin.length, acc, avgConf };
  });

  const uncSorted = unc
    ? [...unc].sort((a,b) => a.uncertainty_std - b.uncertainty_std)
    : [];

  const plots = [
    { src:"/plots/confidence_analysis.png",
      label:"Confidence distribution & calibration analysis" },
    { src:"/plots/confusion_matrix_analysis.png",
      label:"Confusion matrix, ROC curve & threshold sensitivity" },
    { src:"/plots/uncertainty_analysis.png",
      label:"Bayesian uncertainty — MC Dropout analysis" },
  ];

  return (
    <div className="tab-container">
      <h2 className="tab-title">Model statistics</h2>
      <p style={{ fontSize:13, color:"var(--color-text-secondary)", marginTop:-8 }}>
        Hover over any element for details
      </p>

      {/* Metric cards */}
      <div className="cards-row">
        <StatCard label="Accuracy"    value={`${accuracy}%`}
          detail="29 of 34 test images correctly classified"
          color="#4ade80"/>
        <StatCard label="ROC-AUC"     value="0.8997"
          detail="Near-excellent discrimination. Above 0.85 is strong."
          color="#818cf8"/>
        <StatCard label="Dirty F1"    value="85.71%"
          detail="Precision 83.3% and Recall 88.2% for Dirty class"
          color="#f87171"/>
        <StatCard label="Test images" value={total}
          detail={`${cleanCount} Clean + ${dirtyCount} Dirty — balanced`}
          color="#38bdf8"/>
      </div>

      {/* Statistical tests */}
      {tests && (
        <div className="section">
          <h3 className="section-title">Statistical tests</h3>
          <div className="cards-row">
            <TestCard name="Binomial test"
              stat={`p = ${tests.binomial_test.p_value}`}
              verdict="Significantly better than chance ✓"
              detail="Only 0.0019% probability this happened by luck"
              cls="pass"/>
            <TestCard name="McNemar's test"
              stat={`p = ${tests.mcnemar_test.p_value}`}
              verdict="Significantly better than baseline ✓"
              detail="CNN corrects 14 images the baseline misses vs only 2"
              cls="pass"/>
            <TestCard name="Bootstrap 95% CI"
              stat={`[${(tests.bootstrap_ci.accuracy.ci_low*100).toFixed(1)}% – ${(tests.bootstrap_ci.accuracy.ci_high*100).toFixed(1)}%]`}
              verdict="Accuracy confidence interval"
              detail="10,000 bootstrap resamples. Wide = small test set (n=34)"
              cls="info"/>
          </div>
        </div>
      )}

      {/* Charts */}
      <div className="charts-row">
        <div className="chart-card">
          <h3 className="chart-title">Prediction distribution</h3>
          <DonutChart cleanCount={cleanCount} dirtyCount={dirtyCount}
                      total={total} accuracy={accuracy}/>
        </div>
        <div className="chart-card">
          <h3 className="chart-title">Calibration: confidence vs accuracy</h3>
          <p style={{ fontSize:11, color:"var(--color-text-secondary)",
                      margin:"0 0 8px" }}>
            Hover each bar to see details
          </p>
          <div style={{ display:"flex", flexDirection:"column", gap:4 }}>
            {calBins.map(b => (
              <HoverBar key={b.label} label={b.label} value={b.acc}
                count={b.count}
                extraInfo={`Accuracy: ${b.acc}% | Avg conf: ${b.avgConf}% | n=${b.count}`}/>
            ))}
          </div>
          <div style={{ display:"flex", gap:12, marginTop:10 }}>
            <div style={{ display:"flex", alignItems:"center", gap:6 }}>
              <div style={{ width:10, height:10, borderRadius:2,
                            background:"#4ade80" }}/>
              <span style={{ fontSize:11,
                             color:"var(--color-text-secondary)" }}>Accuracy %</span>
            </div>
          </div>
        </div>
      </div>

      {/* Uncertainty strip */}
      {uncSorted.length > 0 && (
        <div className="chart-card full-width">
          <h3 className="chart-title">
            Per-image uncertainty — MC Dropout std (hover for details)
          </h3>
          <p style={{ fontSize:11, color:"var(--color-text-secondary)",
                      margin:"0 0 10px" }}>
            Sorted lowest → highest · Green = correct · Red = wrong ·
            Bar width = uncertainty (max 0.15)
          </p>
          <div style={{ display:"flex", flexDirection:"column", gap:4 }}>
            {uncSorted.map((d,i) => <UncertaintyBar key={i} item={d}/>)}
          </div>
          <div style={{ display:"flex", gap:16, marginTop:10, flexWrap:"wrap" }}>
            {[
              { color:"#4ade80", label:"Correct prediction" },
              { color:"#f87171", label:"Wrong prediction"   },
            ].map(l => (
              <div key={l.label} style={{ display:"flex", alignItems:"center", gap:6 }}>
                <div style={{ width:10, height:10, borderRadius:2,
                              background:l.color }}/>
                <span style={{ fontSize:11,
                               color:"var(--color-text-secondary)" }}>{l.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Analysis plots */}
      <div className="chart-card full-width">
        <h3 className="chart-title">Detailed analysis plots</h3>
        <p style={{ fontSize:11, color:"var(--color-text-secondary)",
                    margin:"0 0 12px" }}>
          Generated from Phase 1 statistical analysis · Click any plot to expand
        </p>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12 }}>
          {plots.map(p => <PlotImage key={p.src} src={p.src} label={p.label}/>)}
        </div>
      </div>

    </div>
  );
}