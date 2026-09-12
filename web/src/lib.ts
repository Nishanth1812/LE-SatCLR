export function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatDuration(seconds: number) {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
}

export function progressPercent(done: number, total: number) {
  return total ? Math.min(100, Math.max(0, Math.round((done / total) * 100))) : 0;
}

export function isEvaluationPending(state: string) {
  return state === "starting" || state === "running";
}
