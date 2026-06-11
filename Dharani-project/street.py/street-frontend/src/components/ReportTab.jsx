import { useState } from "react";
import { generateReport } from "../api";

export default function ReportTab({ lastPrediction }) {
  const [report,  setReport]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  async function handleGenerate() {
    if (!lastPrediction) return;
    setLoading(true);
    setError(null);
    try {
      const text = await generateReport(lastPrediction);
      setReport(text);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  if (!lastPrediction) {
    return (
      <div className="tab-container">
        <h2 className="tab-title">Generate report</h2>
        <div className="empty-state">
          <span className="empty-icon">📋</span>
          <p>No prediction yet. Go to the Predict tab and analyse an image first.</p>
        </div>
      </div>
    );
  }

  const isDirty = lastPrediction.label === "Dirty";

  return (
    <div className="tab-container">
      <h2 className="tab-title">AI-generated cleanliness report</h2>

      {/* Prediction summary */}
      <div className={`report-summary ${isDirty ? "dirty" : "clean"}`}>
        {lastPrediction.imageUrl && (
          <img src={lastPrediction.imageUrl} alt="Analysed street" className="report-thumb" />
        )}
        <div className="report-meta">
          <div className="report-label">{lastPrediction.label}</div>
          <div className="report-metrics">
            <span>Confidence: {(lastPrediction.confidence * 100).toFixed(1)}%</span>
            <span>Uncertainty: {lastPrediction.uncertainty}</span>
            <span>Entropy: {lastPrediction.entropy}</span>
            <span>{lastPrediction.is_ambiguous ? "⚠ Ambiguous" : "✓ Reliable"}</span>
          </div>
        </div>
      </div>

      <button
        className="predict-btn"
        onClick={handleGenerate}
        disabled={loading}
      >
        {loading ? "Generating report…" : "Generate AI Report"}
      </button>

      {error && <div className="error-box">⚠ {error}</div>}

      {report && (
        <div className="report-card">
          <div className="report-header">
            <span>📄</span>
            <span>Municipal Cleanliness Assessment</span>
            <span className="report-date">{new Date().toLocaleDateString()}</span>
          </div>
          <p className="report-body">{report}</p>
          <button
            className="copy-btn"
            onClick={() => navigator.clipboard.writeText(report)}
          >
            Copy report
          </button>
        </div>
      )}
    </div>
  );
}