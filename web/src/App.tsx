import { useEffect, useState, type CSSProperties } from "react";
import { formatDuration, formatPercent, progressPercent } from "./lib";

type Status = {
  ready: boolean;
  message: string;
  device: string;
  datasetReady: boolean;
  splitReady: boolean;
  testSamples: number;
  checkpoint: null | { name: string; stage: string; epoch: number | null; score: number | null };
};

type Result = {
  sampleCount: number;
  durationSeconds: number;
  metrics: { accuracy: number; precision: number; recall: number; macroF1: number };
  classes: string[];
  confusionMatrix: number[][];
  perClass: { name: string; support: number; accuracy: number }[];
  samples: { datasetIndex: number; actual: string; predicted: string; confidence: number; isCorrect: boolean }[];
};

type Job = {
  state: "idle" | "running" | "complete" | "failed";
  progress: { done: number; total: number };
  error?: string;
  result?: Result;
};

const API_BASE = import.meta.env.VITE_API_URL ?? "";

async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "The dashboard API did not respond.");
  }
  return response.json();
}

function Metric({ label, value, lead = false }: { label: string; value: number; lead?: boolean }) {
  return (
    <article className={lead ? "metric metric-lead" : "metric"}>
      <span>{label}</span><strong>{formatPercent(value)}</strong>
    </article>
  );
}

function ConfusionMatrix({ result }: { result: Result }) {
  const max = Math.max(...result.confusionMatrix.flat(), 1);
  return (
    <div className="matrix-wrap" tabIndex={0} aria-label="Confusion matrix; rows are actual and columns are predicted classes">
      <table className="matrix">
        <thead><tr><th aria-label="Actual versus predicted" className="matrix-corner">A\P</th>{result.classes.map((name) => <th key={name} title={name}>{name.slice(0, 3)}</th>)}</tr></thead>
        <tbody>{result.confusionMatrix.map((row, rowIndex) => (
          <tr key={result.classes[rowIndex]}>
            <th title={result.classes[rowIndex]}>{result.classes[rowIndex].slice(0, 3)}</th>
            {row.map((value, columnIndex) => (
              <td key={columnIndex} className={rowIndex === columnIndex ? "correct-cell" : ""}
                  style={{ "--heat": value / max } as CSSProperties}
                  title={`${result.classes[rowIndex]} → ${result.classes[columnIndex]}: ${value}`}>{value}</td>
            ))}
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [job, setJob] = useState<Job>({ state: "idle", progress: { done: 0, total: 0 } });
  const [sampleLimit, setSampleLimit] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api<Status>("/api/status"), api<Job>("/api/evaluations/current")])
      .then(([nextStatus, nextJob]) => { setStatus(nextStatus); setJob(nextJob); })
      .catch((reason) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (job.state !== "running") return;
    const timer = window.setInterval(() => {
      api<Job>("/api/evaluations/current").then(setJob).catch((reason) => setError(reason.message));
    }, 750);
    return () => window.clearInterval(timer);
  }, [job.state]);

  async function runEvaluation() {
    setError("");
    try {
      setJob(await api<Job>("/api/evaluations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sampleLimit }),
      }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not start evaluation.");
    }
  }

  const result = job.result;
  const progress = progressPercent(job.progress.done, job.progress.total);

  return (
    <>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="LE-SatCLR home"><span className="brand-mark">LE</span><span>SatCLR</span></a>
        <div className="system-status"><span className={status?.ready ? "status-dot ready" : "status-dot"} />{status ? `${status.device.toUpperCase()} · ${status.ready ? "System ready" : "Setup needed"}` : "Connecting"}</div>
      </header>

      <main id="top">
        <section className="masthead">
          <div className="masthead-copy">
            <p className="eyebrow">HELD-OUT EVALUATION / EURO<span>SAT</span></p>
            <h1>Proof,<br />not promise.</h1>
            <p className="lede">Put the final land-cover classifier through its untouched test split. Every score here comes from a real image and a real forward pass.</p>
            <div className="dataset-facts" aria-label="Evaluation facts">
              <span><b>10</b> land-cover classes</span><span><b>64²</b> RGB imagery</span><span><b>42</b> fixed seed</span>
            </div>
          </div>

          <aside className="run-panel" aria-labelledby="run-title">
            <div className="panel-topline"><span>TEST CONSOLE</span><span>v0.1</span></div>
            <h2 id="run-title">Run evaluation</h2>
            {!status && !error && <div className="status-skeleton" aria-label="Loading model status" />}
            {status && <div className={status.ready ? "readiness ready-box" : "readiness"} role="status">
              <span className="readiness-icon">{status.ready ? "✓" : "!"}</span>
              <div><strong>{status.ready ? "Model connected" : "Not ready yet"}</strong><small>{status.message}</small></div>
            </div>}
            <dl className="model-meta">
              <div><dt>Checkpoint</dt><dd>{status?.checkpoint?.name || "—"}</dd></div>
              <div><dt>Training stage</dt><dd>{status?.checkpoint?.stage || "—"}</dd></div>
              <div><dt>Test samples</dt><dd>{status?.testSamples?.toLocaleString() || "—"}</dd></div>
            </dl>
            <div className="control-label"><label htmlFor="sample-limit">Evaluation size</label><span>Real test images only</span></div>
            <select id="sample-limit" value={sampleLimit} onChange={(event) => setSampleLimit(Number(event.target.value))} disabled={job.state === "running"}>
              <option value={0}>Full test split</option>
              <option value={100}>Quick check · 100 images</option>
              <option value={500}>Extended check · 500 images</option>
            </select>
            <button className="run-button" onClick={runEvaluation} disabled={!status?.ready || job.state === "running"}>
              <span>{job.state === "running" ? `Evaluating ${progress}%` : result ? "Run again" : "Evaluate final model"}</span><span aria-hidden="true">↗</span>
            </button>
            {job.state === "running" && <div className="progress-wrap"><div className="progress-status" aria-live="polite"><span>Forward pass in progress</span><b>{job.progress.done.toLocaleString()} / {job.progress.total.toLocaleString()}</b></div><div className="progress-track" role="progressbar" aria-label="Evaluation progress" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}><span style={{ width: `${progress}%` }} /></div></div>}
            {(error || job.error) && <p className="error-message" role="alert">{error || job.error}</p>}
            {status && !status.checkpoint && <p className="setup-note">Set <code>LE_SATCLR_CHECKPOINT</code> to your final <code>.pt</code> file, then restart the API.</p>}
          </aside>
        </section>

        {result ? <section className="results" aria-live="polite">
          <div className="section-heading"><div><p className="eyebrow">LATEST RUN</p><h2>Test performance</h2><p className="section-summary">A measured view of how the classifier behaves beyond its training data.</p></div><p>{result.sampleCount.toLocaleString()} images · {formatDuration(result.durationSeconds)}</p></div>
          <div className="metrics-row"><Metric label="Accuracy" value={result.metrics.accuracy} lead /><Metric label="Macro F1" value={result.metrics.macroF1} /><Metric label="Precision" value={result.metrics.precision} /><Metric label="Recall" value={result.metrics.recall} /></div>

          <div className="analysis-grid">
            <article className="analysis-panel"><div className="panel-heading"><h3>Confusion matrix</h3><span>ACTUAL ↓ · PREDICTED →</span></div><ConfusionMatrix result={result} /></article>
            <article className="analysis-panel"><div className="panel-heading"><h3>Class accuracy</h3><span>{result.perClass.length} CLASSES</span></div>
              <ol className="class-list">{result.perClass.map((item) => <li key={item.name}><div><span>{item.name}</span><b>{formatPercent(item.accuracy)}</b></div><div className="class-bar"><span style={{ width: formatPercent(item.accuracy) }} /></div><small>{item.support} test images</small></li>)}</ol>
            </article>
          </div>

          <div className="section-heading samples-heading"><div><p className="eyebrow">GROUND TRUTH CHECK</p><h2>Sample predictions</h2></div><p>A deterministic cross-section of the run</p></div>
          <div className="sample-grid">{result.samples.map((sample) => <article className="sample-card" key={sample.datasetIndex}>
            <div className="sample-image"><img src={`${API_BASE}/api/test-images/${sample.datasetIndex}`} alt={`EuroSAT test image labeled ${sample.actual}`} loading="lazy" /><span className={sample.isCorrect ? "verdict correct" : "verdict"}>{sample.isCorrect ? "MATCH" : "MISS"}</span></div>
            <div className="sample-copy"><small>MODEL SAYS</small><strong>{sample.predicted}</strong><span>{formatPercent(sample.confidence)} confidence</span><span className="truth">Truth · {sample.actual}</span></div>
          </article>)}</div>
        </section> : <section className="waiting" aria-label="Awaiting evaluation"><div className="waiting-number">01</div><div><p className="eyebrow">RESULTS DECK</p><h2>Your evidence lands here.</h2><p>Connect the final checkpoint and run the untouched test split to reveal metrics, class-level errors, and individual predictions.</p><ol className="waiting-flow"><li><b>01</b><span>Connect a classifier checkpoint</span></li><li><b>02</b><span>Run the held-out test split</span></li><li><b>03</b><span>Inspect errors and predictions</span></li></ol></div></section>}
      </main>
      <footer><span>LE-SatCLR</span><span>LABEL-EFFICIENT SATELLITE CLASSIFICATION</span><span>SEED 42</span></footer>
    </>
  );
}
