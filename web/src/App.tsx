import { useEffect, useRef, useState, type CSSProperties } from "react";
import { formatApiError, formatDuration, formatPercent, isEvaluationPending, parseSampleLimit, pollDelayMs, progressPercent, sampleLimitError } from "./lib";

type Status = {
  ready: boolean;
  message: string;
  device: string;
  datasetReady: boolean;
  splitReady: boolean;
  testSamples: number;
  checkpoint: null | { name: string; stage: string; epoch: number | null; score: number | null };
};

type Checkpoint = {
  name: string;
  stage: string;
  labelPercent: number | null;
  epoch: number | null;
  score: number | null;
};

type Result = {
  sampleCount: number;
  checkpoint: string;
  durationSeconds: number;
  metrics: { accuracy: number; precision: number; recall: number; macroF1: number };
  classes: string[];
  confusionMatrix: number[][];
  perClass: { name: string; support: number; accuracy: number }[];
  samples: { datasetIndex: number; actual: string; predicted: string; confidence: number; isCorrect: boolean }[];
};

type Job = {
  state: "idle" | "starting" | "running" | "complete" | "failed";
  progress: { done: number; total: number };
  startedAt?: string;
  error?: string;
  result?: Result;
};

const API_BASE = import.meta.env.VITE_API_URL ?? "";

async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  init?.signal?.addEventListener("abort", abort, { once: true });
  if (init?.signal?.aborted) controller.abort();
  let timedOut = false;
  const timer = window.setTimeout(() => { timedOut = true; controller.abort(); }, 15000);
  try {
    const response = await fetch(`${API_BASE}${url}`, { ...init, signal: controller.signal });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(formatApiError(body?.detail, "The dashboard API did not respond. Please try again."));
    }
    return await response.json();
  } catch (reason) {
    if (init?.signal?.aborted) throw reason;
    if (timedOut) throw new Error("The request timed out after 15 seconds. Please try again.");
    if (reason instanceof TypeError) throw new Error("Could not connect to the dashboard API. Check your connection and try again.");
    throw reason;
  } finally {
    window.clearTimeout(timer);
    init?.signal?.removeEventListener("abort", abort);
  }
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
        <caption>Confusion matrix: actual classes in rows, predicted classes in columns.</caption>
        <thead><tr><th scope="col" aria-label="Actual versus predicted" className="matrix-corner">A\P</th>{result.classes.map((name) => <th key={name} scope="col" title={name} aria-label={`Predicted ${name}`}>{name.slice(0, 3)}</th>)}</tr></thead>
        <tbody>{result.confusionMatrix.map((row, rowIndex) => (
          <tr key={result.classes[rowIndex]}>
            <th scope="row" title={result.classes[rowIndex]} aria-label={`Actual ${result.classes[rowIndex]}`}>{result.classes[rowIndex].slice(0, 3)}</th>
            {row.map((value, columnIndex) => (
              <td key={columnIndex} className={rowIndex === columnIndex ? "correct-cell" : ""}
                  style={{ "--heat": value / max } as CSSProperties}
                  aria-label={`Actual ${result.classes[rowIndex]}, predicted ${result.classes[columnIndex]}: ${value}`}
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
  const [sampleLimit, setSampleLimit] = useState("25");
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
  const [error, setError] = useState("");
  const [now, setNow] = useState(() => Date.now());
  const [connectionRetry, setConnectionRetry] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const submission = useRef<AbortController | null>(null);

  useEffect(() => () => { submission.current?.abort(); }, []);

  useEffect(() => {
    const controller = new AbortController();
    const init = { signal: controller.signal };
    setError("");
    Promise.all([api<Status>("/api/status", init), api<Job>("/api/evaluations/current", init), api<Checkpoint[]>("/api/checkpoints", init)])
      .then(([nextStatus, nextJob, nextCheckpoints]) => {
        if (controller.signal.aborted) return;
        setStatus(nextStatus); setJob(nextJob); setCheckpoints(nextCheckpoints);
      })
      .catch((reason) => {
        if (controller.signal.aborted) return;
        setError(reason instanceof Error ? reason.message : "The dashboard API did not respond.");
        controller.abort();
      });
    return () => controller.abort();
  }, [connectionRetry]);

  useEffect(() => {
    if (!isEvaluationPending(job.state)) return;
    setNow(Date.now());
    const clock = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(clock);
  }, [job.state]);

  useEffect(() => {
    if (!isEvaluationPending(job.state) || isSubmitting) return;
    let timer = 0;
    let failures = 0;
    const controller = new AbortController();
    const poll = async () => {
      if (controller.signal.aborted) return;
      let failed = false;
      try {
        const nextJob = await api<Job>("/api/evaluations/current", { signal: controller.signal });
        if (controller.signal.aborted) return;
        failures = 0;
        setError("");
        setJob(nextJob);
        if (!isEvaluationPending(nextJob.state)) return;
      } catch (reason) {
        if (controller.signal.aborted) return;
        failed = true;
        setError(reason instanceof Error ? reason.message : "The dashboard API did not respond.");
      }
      const delay = pollDelayMs(failures, failed, document.hidden);
      if (failed) failures += 1;
      if (!controller.signal.aborted) timer = window.setTimeout(poll, delay);
    };
    void poll();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [job.state, isSubmitting]);

  const elapsedSeconds = job.startedAt
    ? Math.max(0, Math.round((now - Date.parse(job.startedAt)) / 1000))
    : 0;

  const parsedSampleLimit = parseSampleLimit(sampleLimit);
  const validationError = sampleLimitError(sampleLimit, status?.testSamples);
  const model = selectedCheckpoint ? checkpoints.find((item) => item.name === selectedCheckpoint) : status?.checkpoint;

  async function runEvaluation() {
    if (!status?.ready || validationError || parsedSampleLimit === null || isEvaluationPending(job.state) || submission.current) return;
    const controller = new AbortController();
    submission.current = controller;
    setIsSubmitting(true);
    setError("");
    setJob((current) => ({
      ...current,
      state: "starting",
      startedAt: new Date().toISOString(),
      error: undefined,
      progress: { done: 0, total: parsedSampleLimit === 0 ? status.testSamples : parsedSampleLimit },
    }));
    try {
      const nextJob = await api<Job>("/api/evaluations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sampleLimit: parsedSampleLimit, checkpoint: selectedCheckpoint || null }),
        signal: controller.signal,
      });
      if (controller.signal.aborted || submission.current !== controller) return;
      setJob(nextJob);
    } catch (reason) {
      if (controller.signal.aborted || submission.current !== controller) return;
      const message = reason instanceof Error ? reason.message : "Could not start evaluation.";
      setError(message);
      setJob((current) => ({ ...current, state: "failed", error: message }));
    } finally {
      if (!controller.signal.aborted && submission.current === controller) {
        submission.current = null;
        setIsSubmitting(false);
      }
    }
  }

  const result = job.result;
  const progress = progressPercent(job.progress.done, job.progress.total);
  const isEstimate = Boolean(result && status && result.sampleCount < status.testSamples);
  const isPending = isEvaluationPending(job.state);

  return (
    <>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="LE-SatCLR home"><span className="brand-mark">LE</span><span>SatCLR</span></a>
        <div className="system-status"><span className={status?.ready ? "status-dot ready" : "status-dot"} />{status ? `${status.device.toUpperCase()} · ${status.ready ? "System ready" : "Setup needed"}` : "Connecting"}</div>
      </header>

      <main id="top">
        <section className="masthead">
          <div className="masthead-copy">
            <p className="eyebrow">TEST SPLIT · 2,700 EURO<span>SAT</span> IMAGES</p>
            <h1>Measure the<br />final model.</h1>
            <p className="lede">Run the trained land-cover classifier on the test split and see accuracy, per-class recall, and sample predictions.</p>
            <p className="protocol-note">The split is held out from labels only: transductive pretraining already saw these images unlabeled, so metrics reflect labeled evaluation on data the classifier never saw annotated.</p>
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
            <div className="control-label"><label htmlFor="model-pick">Model</label><span>1% vs 10% labels</span></div>
            <select id="model-pick" value={selectedCheckpoint} onChange={(event) => setSelectedCheckpoint(event.target.value)} disabled={isPending}>
              <option value="">Newest model (auto)</option>
              {checkpoints.map((item) => <option key={item.name} value={item.name}>
                {item.stage} · {item.labelPercent === null ? "labels n/a" : `${item.labelPercent}% labels`} ({item.name})
              </option>)}
            </select>
            <div className="control-label"><label htmlFor="sample-limit">Images to evaluate</label><span>Random held-out images</span></div>
            <input id="sample-limit" type="number" min={0} max={status?.testSamples} step={1} value={sampleLimit}
              onChange={(event) => setSampleLimit(event.target.value)} disabled={isPending}
              aria-invalid={Boolean(validationError)} aria-describedby={validationError ? "sample-limit-note sample-limit-error" : "sample-limit-note"} />
            <p id="sample-limit-note" className="control-note">Use 0 for the full test split. Smaller runs are random estimates.</p>
            {validationError && <p id="sample-limit-error" className="error-message state-message" aria-live="polite">{validationError}</p>}
            <div className="sample-presets" aria-label="Evaluation size shortcuts">
              {[25, 100, 500].map((size) => <button key={size} type="button" className={parsedSampleLimit === size ? "selected" : ""} aria-pressed={parsedSampleLimit === size} onClick={() => setSampleLimit(String(size))} disabled={isPending || !status || size > status.testSamples}>{size} images</button>)}
              <button type="button" className={parsedSampleLimit === 0 ? "selected" : ""} aria-pressed={parsedSampleLimit === 0} onClick={() => setSampleLimit("0")} disabled={isPending || !status}>Full split</button>
            </div>
            <button className="run-button" onClick={runEvaluation} disabled={!status?.ready || isPending || Boolean(validationError)}>
              <span>{job.state === "starting" ? "Starting evaluation…" : job.state === "running" ? `Evaluating ${progress}%` : result ? "Run again" : "Evaluate final model"}</span><span aria-hidden="true">↗</span>
            </button>
            {isPending && <div className="progress-wrap"><div className="progress-status" aria-live="polite"><span>{job.state === "starting" ? `Contacting evaluation server · ${elapsedSeconds}s elapsed` : `Forward pass in progress · ${elapsedSeconds}s elapsed`}</span><b>{job.progress.done.toLocaleString()} / {job.progress.total.toLocaleString()} images</b></div><div className={job.state === "starting" ? "progress-track starting" : "progress-track"} role="progressbar" aria-label="Evaluation progress" aria-valuetext={job.state === "starting" ? "Starting evaluation" : `${progress}% complete`} aria-valuenow={job.state === "running" ? progress : undefined} aria-valuemin={0} aria-valuemax={100}><span style={job.state === "running" ? { width: `${progress}%` } : undefined} /></div></div>}
            {(error || job.error) && <p className="error-message" role="alert">{error || job.error}</p>}
            {status && !status.checkpoint && <p className="setup-note">Set <code>LE_SATCLR_CHECKPOINT</code> to your final <code>.pt</code> file, then restart the API.</p>}
          </aside>
        </section>

        {result ? <section className="results" aria-live="polite">
          <div className="section-heading"><div><p className="eyebrow">LATEST RUN · {result.checkpoint}</p><h2>Test performance</h2><p className="section-summary">Accuracy is the share of test images the model labeled correctly — higher is better. Compare the 1% and 10% models to see what extra labels buy.</p></div><p>{result.sampleCount.toLocaleString()} images · {formatDuration(result.durationSeconds)}{isEstimate ? " · random estimate" : " · full split"}</p></div>
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
        </section> : <section className="waiting" aria-label="Awaiting evaluation"><div className="waiting-number">01</div><div><p className="eyebrow">RESULTS DECK</p><h2>Results appear here.</h2><p>Connect the final checkpoint and run the untouched test split to see metrics, class-level errors, and individual predictions.</p><ol className="waiting-flow"><li><b>01</b><span>Connect a classifier checkpoint</span></li><li><b>02</b><span>Run the held-out test split</span></li><li><b>03</b><span>Inspect errors and predictions</span></li></ol></div></section>}
      </main>
      <footer><span>LE-SatCLR</span><span>LABEL-EFFICIENT SATELLITE CLASSIFICATION</span><span>SEED 42</span></footer>
    </>
  );
}
