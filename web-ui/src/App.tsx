import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchCli,
  fetchJob,
  fetchLocations,
  fetchStatus,
  startRun,
  type JobSnapshot,
  type LocationStatus,
  type LocationSummary,
  type StageNode,
  type YearCell,
} from "./api";

function Timeline({ cells }: { cells: YearCell[] }) {
  return (
    <div>
      <div className="legend">
        <span>
          <i style={{ background: "#1b7f4a" }} />
          rendered
        </span>
        <span>
          <i style={{ background: "#2f6fed" }} />
          downloaded
        </span>
        <span>
          <i style={{ background: "#c9850a" }} />
          available
        </span>
        <span>
          <i style={{ background: "rgba(255,255,255,0.15)" }} />
          missing
        </span>
      </div>
      <div className="timeline">
        {cells.map((cell, i) => (
          <div
            key={cell.year}
            className={`year-chip ${cell.level}`}
            style={{ animationDelay: `${i * 28}ms` }}
            title={`${cell.year}: ${cell.level}`}
          >
            {cell.year}
          </div>
        ))}
      </div>
    </div>
  );
}

function Dag({ stages }: { stages: StageNode[] }) {
  const byId = useMemo(() => Object.fromEntries(stages.map((s) => [s.id, s])), [stages]);
  const core = ["index", "download", "render", "validate"]
    .map((id) => byId[id])
    .filter(Boolean);
  const optional = ["dem", "osm", "raw_export"].map((id) => byId[id]).filter(Boolean);

  return (
    <div>
      <div className="dag-row">
        {core.map((stage, i) => (
          <div key={stage.id} style={{ display: "contents" }}>
            {i > 0 ? <div className="dag-arrow">→</div> : null}
            <div className={`dag-node ${stage.status}`} title={stage.detail || stage.status}>
              {stage.label}
              <small>{stage.status}</small>
            </div>
          </div>
        ))}
      </div>
      <div className="optional-label">optional</div>
      <div className="dag-row">
        {optional.map((stage) => (
          <div
            key={stage.id}
            className={`dag-node ${stage.status}`}
            title={stage.detail || stage.status}
          >
            {stage.label}
            <small>{stage.status}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [locations, setLocations] = useState<LocationSummary[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [status, setStatus] = useState<LocationStatus | null>(null);
  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [cli, setCli] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetchLocations()
      .then((data) => {
        setLocations(data.locations);
        const prefer =
          data.locations.find((l) => l.id === "poznan")?.id || data.locations[0]?.id || "";
        setSelected(prefer);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  const refreshStatus = useCallback(async (id: string) => {
    if (!id) return;
    setLoading(true);
    setError("");
    try {
      const next = await fetchStatus(id);
      setStatus(next);
      if (next.active_job) setJob(next.active_job);
      const deeplink = await fetchCli(id, "run");
      setCli(deeplink.command);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selected) void refreshStatus(selected);
  }, [selected, refreshStatus]);

  useEffect(() => {
    if (!job || (job.status !== "pending" && job.status !== "running")) return;
    const timer = window.setInterval(() => {
      void fetchJob(job.id)
        .then((next) => {
          setJob(next);
          if (next.status === "success" || next.status === "failed") {
            void refreshStatus(selected);
          }
        })
        .catch(() => undefined);
    }, 1200);
    return () => window.clearInterval(timer);
  }, [job, refreshStatus, selected]);

  async function onRun(kind: "index" | "run") {
    if (!selected) return;
    setBusy(true);
    setError("");
    try {
      const started = await startRun(selected, kind);
      setJob(started);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function copyCli() {
    if (!cli) return;
    try {
      await navigator.clipboard.writeText(cli);
    } catch {
      setError("Could not copy to clipboard");
    }
  }

  const loc = status?.location;
  const jobRunning = job?.status === "pending" || job?.status === "running";

  return (
    <div className="app">
      <header className="hero">
        <h1 className="brand">satmap</h1>
        <p className="tagline">
          Pick a location, read year coverage and pipeline state from disk, then kick a run —
          or copy the CLI deeplink and stay in the terminal.
        </p>
        <div className="hero-controls">
          <div className="field">
            <label htmlFor="location">Location</label>
            <select
              id="location"
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              disabled={!locations.length}
            >
              {locations.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.location_name} ({item.id})
                </option>
              ))}
            </select>
          </div>
          <button type="button" onClick={() => void refreshStatus(selected)} disabled={!selected || loading}>
            Refresh status
          </button>
        </div>
        {error ? <p className="error">{error}</p> : null}
      </header>

      {loc ? (
        <>
          <section className="panel">
            <h2>
              <span
                className={`status-dot ${
                  status?.artifacts.validation_report
                    ? ""
                    : status?.artifacts.download_manifest
                      ? "warn"
                      : "bad"
                }`}
              />
              {loc.location_name}
            </h2>
            <div className="meta-grid">
              <div className="meta-item">
                <span>Provider</span>
                <strong>{loc.provider}</strong>
              </div>
              <div className="meta-item">
                <span>Years</span>
                <strong>
                  {loc.year_start}–{loc.year_end}
                </strong>
              </div>
              <div className="meta-item">
                <span>SRS</span>
                <strong>{loc.srs || "—"}</strong>
              </div>
              <div className="meta-item">
                <span>Area</span>
                <strong>{loc.area_km2 != null ? `${loc.area_km2} km²` : "—"}</strong>
              </div>
              <div className="meta-item">
                <span>Profile / mode</span>
                <strong>
                  {loc.profile || "—"} / {loc.mode || "—"}
                </strong>
              </div>
              <div className="meta-item">
                <span>Artifacts</span>
                <strong>{loc.paths.artifacts_dir?.split("/").pop() || "—"}</strong>
              </div>
            </div>
            <div className="actions">
              <button
                type="button"
                className="primary"
                disabled={busy || jobRunning}
                onClick={() => void onRun("index")}
              >
                Run index
              </button>
              <button
                type="button"
                className="primary"
                disabled={busy || jobRunning}
                onClick={() => void onRun("run")}
              >
                Run full pipeline
              </button>
              <button type="button" onClick={() => void copyCli()} disabled={!cli}>
                Copy CLI
              </button>
            </div>
            {cli ? <div className="cli-box">{cli}</div> : null}
            {job ? (
              <div className="job-box">
                <strong>
                  Job {job.id} · {job.name} · {job.status}
                </strong>
                <div>{job.message}</div>
                {job.logs.length ? <pre>{job.logs.slice(-20).join("\n")}</pre> : null}
              </div>
            ) : null}
          </section>

          <section className="panel">
            <h2>Year timeline</h2>
            <Timeline cells={status?.timeline || []} />
          </section>

          <section className="panel">
            <h2>Pipeline DAG</h2>
            <Dag stages={status?.dag.stages || []} />
          </section>
        </>
      ) : (
        <section className="panel">
          <h2>{loading ? "Loading…" : "Select a location"}</h2>
        </section>
      )}
    </div>
  );
}
