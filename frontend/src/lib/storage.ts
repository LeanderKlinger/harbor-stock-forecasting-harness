// Named localStorage keys live here so a future grep finds the surface area.

const HIDDEN_EXPERIMENTS_KEY = "hidden_experiments_v1";

export function loadHiddenExperiments(): Set<string> {
  try {
    const raw = localStorage.getItem(HIDDEN_EXPERIMENTS_KEY);
    if (!raw) return new Set();
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return new Set();
    return new Set(parsed.filter((x): x is string => typeof x === "string"));
  } catch {
    return new Set();
  }
}

export function saveHiddenExperiments(hidden: Set<string>): void {
  try {
    localStorage.setItem(HIDDEN_EXPERIMENTS_KEY, JSON.stringify([...hidden]));
  } catch {
    // Storage quota or disabled storage — fail silently; the in-memory state
    // still works for the session.
  }
}
