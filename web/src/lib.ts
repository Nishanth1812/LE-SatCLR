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

export function formatApiError(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const parts = detail.map((item) => {
      if (item && typeof item === "object" && "msg" in item) {
        const entry = item as { msg?: unknown; loc?: unknown };
        const msg = typeof entry.msg === "string" ? entry.msg.trim() : "";
        if (!msg) return "";
        const loc = Array.isArray(entry.loc) ? entry.loc.filter((part) => part !== "body" && (typeof part === "string" || typeof part === "number")).join(".") : "";
        return loc ? `${loc}: ${msg}` : msg;
      }
      return typeof item === "string" ? item.trim() : "";
    }).filter(Boolean);
    if (parts.length) return parts.join("; ");
  }
  return fallback;
}

export function parseSampleLimit(raw: string): number | null {
  const text = raw.trim();
  if (!text || !/^\d+$/.test(text)) return null;
  return Number(text);
}

export function sampleLimitError(raw: string, max: number | undefined): string {
  const value = parseSampleLimit(raw);
  if (value === null) return "Enter a whole number of images, or 0 for the full split.";
  if (max !== undefined && value > max) return `Only ${max.toLocaleString()} test images are available.`;
  return "";
}

export function pollDelayMs(attempt: number, failed: boolean, hidden: boolean): number {
  if (failed) return Math.min(500 * 2 ** attempt, 8000);
  return hidden ? 2000 : 200;
}
