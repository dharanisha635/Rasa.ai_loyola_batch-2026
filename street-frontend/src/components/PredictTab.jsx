import { useState, useRef } from "react";
import { predictImage } from "../api";

export default function PredictTab({ onPrediction }) {
  const [image,      setImage]      = useState(null);
  const [preview,    setPreview]    = useState(null);
  const [result,     setResult]     = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState(null);
  const fileRef = useRef();

  function handleFile(file) {
    if (!file) return;
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }

  function handleDrop(e) {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  }

  async function handlePredict() {
    if (!image) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictImage(image);
      setResult(res);
      onPrediction({ ...res, imageUrl: preview });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const isDirty = result?.label === "Dirty";

  return (
    <div className="tab-container">
      <h2 className="tab-title">Analyse a street image</h2>

      {/* Upload area */}
      <div
        className="upload-zone"
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => fileRef.current.click()}
      >
        {preview
          ? <img src={preview} alt="Street preview" className="preview-img" />
          : <div className="upload-placeholder">
              <span className="upload-icon">📁</span>
              <p>Drag & drop or click to upload a street image</p>
              <p className="upload-hint">JPG, PNG, JPEG supported</p>
            </div>
        }
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={e => handleFile(e.target.files[0])}
        />
      </div>

      <button
        className="predict-btn"
        onClick={handlePredict}
        disabled={!image || loading}
      >
        {loading ? "Analysing…" : "Analyse Image"}
      </button>

      {error && <div className="error-box">⚠ {error}</div>}

      {/* Result card */}
      {result && (
        <div className={`result-card ${isDirty ? "dirty" : "clean"}`}>
          <div className="result-header">
            <span className="result-icon">{isDirty ? "🚨" : "✅"}</span>
            <span className="result-label">{result.label}</span>
          </div>

          <div className="metrics-grid">
            <div className="metric">
              <div className="metric-label">Confidence</div>
              <div className="metric-value">
                {(result.confidence * 100).toFixed(1)}%
              </div>
              <div className="metric-bar">
                <div
                  className="metric-fill"
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>
            </div>
            <div className="metric">
              <div className="metric-label">Uncertainty (MC Dropout)</div>
              <div className="metric-value">{result.uncertainty}</div>
              <div className="metric-bar">
                <div
                  className="metric-fill uncertainty"
                  style={{ width: `${(result.uncertainty / 0.15) * 100}%` }}
                />
              </div>
            </div>
            <div className="metric">
              <div className="metric-label">Predictive Entropy</div>
              <div className="metric-value">{result.entropy}</div>
            </div>
            <div className="metric">
              <div className="metric-label">Reliability</div>
              <div className="metric-value">
                {result.is_ambiguous ? "⚠ Ambiguous" : "✓ Reliable"}
              </div>
            </div>
          </div>

          <p className="result-advice">
            {isDirty
              ? "This street requires cleaning attention. Flag for municipal review."
              : "This street appears clean. Continue routine monitoring."}
          </p>
        </div>
      )}
    </div>
  );
}