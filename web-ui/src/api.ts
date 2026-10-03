export type YearLevel = "rendered" | "downloaded" | "available" | "missing";
export type StageStatus = "pending" | "running" | "done" | "failed" | "skipped";

export interface LocationSummary {
  id: string;
  file: string;
  location_name: string;
  slug: string;
  provider: string;
  year_start: number;
  year_end: number;
  center_lat?: number | null;
  center_lon?: number | null;
  area_km2?: number | null;
  srs?: string | null;
  profile?: string | null;
  mode?: string | null;
  bbox?: string | null;
  paths: Record<string, string>;
  base_json: string;
  providers?: string[];
  error?: string;
}

export interface YearCell {
  year: number;
  level: YearLevel;
  requested: boolean;
  available: boolean;
  downloaded: boolean;
  rendered: boolean;
}

export interface StageNode {
  id: string;
  label: string;
  status: StageStatus;
  detail?: string | null;
}

export interface PipelineDag {
  stages: StageNode[];
  edges: { from: string; to: string }[];
}

export interface JobSnapshot {
  id: string;
  name: string;
  location_id: string;
  status: string;
  message: string;
  exit_code: number | null;
  artifact_path: string | null;
  error: string | null;
  progress_label: string;
  logs: string[];
}

export interface LocationStatus {
  location: LocationSummary;
  timeline: YearCell[];
  dag: PipelineDag;
  artifacts: Record<string, boolean>;
  active_job: JobSnapshot | null;
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export function fetchLocations() {
  return api<{ locations: LocationSummary[]; count: number }>("/api/locations");
}

export function fetchStatus(locationId: string) {
  return api<LocationStatus>(`/api/locations/${encodeURIComponent(locationId)}/status`);
}

export function fetchCli(locationId: string, kind: "index" | "run" = "run") {
  return api<{ command: string; just: string; location_id: string; kind: string }>(
    `/api/locations/${encodeURIComponent(locationId)}/cli?kind=${kind}`,
  );
}

export function startRun(locationId: string, kind: "index" | "run") {
  return api<JobSnapshot>("/api/runs", {
    method: "POST",
    body: JSON.stringify({ location_id: locationId, kind }),
  });
}

export function fetchJob(jobId: string) {
  return api<JobSnapshot>(`/api/runs/${encodeURIComponent(jobId)}`);
}
