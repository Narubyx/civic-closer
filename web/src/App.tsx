import { useEffect, useState } from 'react';
import ArViewer from './components/ArViewer';
import {
  ApiError,
  GeometrySummary,
  HealthResponse,
  RunManifest,
  SegmentResponse,
  fetchArtifact,
  fetchHealth,
  postSegment,
} from './lib/api';
import './app.css';

// Stable Phase 1 deployment endpoint (Modal keeps the URL across deploys).
// Can be overridden with VITE_MODAL_URL.
const DEFAULT_MODAL_URL = 'https://nru-suu-saa--voxel-sam-ar-ingest-fastapi-app.us-east.modal.run';
const MODAL_URL = ((import.meta.env.VITE_MODAL_URL as string | undefined) ?? DEFAULT_MODAL_URL).replace(
  /\/$/,
  '',
);
const ARTIFACTS_URL = (import.meta.env.VITE_ARTIFACTS_URL as string | undefined) ?? './demo';
const MODEL_GLB = (import.meta.env.VITE_MODEL_GLB as string | undefined) ?? './models/model.glb';
const MODEL_USDZ = (import.meta.env.VITE_MODEL_USDZ as string | undefined) ?? './models/model.usdz';

function errText(err: unknown): string {
  if (err instanceof ApiError) return `HTTP ${err.status}: ${err.message}`;
  return err instanceof Error ? err.message : String(err);
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthErr, setHealthErr] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [x, setX] = useState('150');
  const [y, setY] = useState('240');
  const [numKf, setNumKf] = useState('8');
  const [segmenting, setSegmenting] = useState(false);
  const [segment, setSegment] = useState<SegmentResponse | null>(null);
  const [segmentErr, setSegmentErr] = useState('');
  const [artifactsUrl, setArtifactsUrl] = useState(ARTIFACTS_URL);
  const [manifest, setManifest] = useState<RunManifest | null>(null);
  const [geometry, setGeometry] = useState<GeometrySummary | null>(null);
  const [artifactsErr, setArtifactsErr] = useState('');
  const [loadingArtifacts, setLoadingArtifacts] = useState(false);

  useEffect(() => {
    if (!MODAL_URL) {
      setHealthErr('VITE_MODAL_URL not defined (see web/.env.example).');
      return;
    }
    fetchHealth(MODAL_URL).then(setHealth).catch((e: unknown) => setHealthErr(errText(e)));
  }, []);

  async function loadArtifacts(url: string) {
    setLoadingArtifacts(true);
    setArtifactsErr('');
    try {
      const [m, g] = await Promise.all([
        fetchArtifact<RunManifest>(url, 'manifest.json'),
        fetchArtifact<GeometrySummary>(url, 'geometry.json'),
      ]);
      setManifest(m);
      setGeometry(g);
    } catch (e: unknown) {
      setManifest(null);
      setGeometry(null);
      setArtifactsErr(errText(e));
    } finally {
      setLoadingArtifacts(false);
    }
  }

  useEffect(() => {
    void loadArtifacts(artifactsUrl);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function onSegment() {
    setSegmentErr('');
    setSegment(null);
    if (!file) {
      setSegmentErr('Pick an .mp4 video first.');
      return;
    }
    const px = Number(x);
    const py = Number(y);
    const n = Number(numKf);
    if (!Number.isFinite(px) || !Number.isFinite(py) || px < 0 || py < 0) {
      setSegmentErr('x and y must be pixels >= 0.');
      return;
    }
    if (!Number.isInteger(n) || n < 4 || n > 64) {
      setSegmentErr('num_keyframes must be an integer in [4, 64].');
      return;
    }
    setSegmenting(true);
    try {
      setSegment(await postSegment(MODAL_URL, file, px, py, n));
    } catch (e: unknown) {
      setSegmentErr(errText(e));
    } finally {
      setSegmenting(false);
    }
  }

  return (
    <div className="page">
      <header>
        <h1>VOXEL-SAM AR</h1>
        <p className="muted">360° video → segmentation → watertight mesh → WebAR</p>
      </header>

      <section className="card">
        <h2>1 · Modal backend (live)</h2>
        <p className="mono small">{MODAL_URL || '(no URL)'}</p>
        {healthErr && <p className="error">{healthErr}</p>}
        {health ? (
          <p>
            <span className="pill ok">{health.status}</span>{' '}
            <span className="muted">
              {health.service} · GPU SAM:{' '}
              {Object.entries(health.checkpoints)
                .map(([k, v]) => `${k}=${v ? 'yes' : 'no'}`)
                .join(' ')}
            </span>
          </p>
        ) : (
          !healthErr && <p className="muted">Connecting…</p>
        )}
      </section>

      <section className="card">
        <h2>2 · Segment 360° video (live)</h2>
        <div className="form-row">
          <label>
            .mp4 video
            <input
              type="file"
              accept="video/mp4,video/quicktime"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
          <label>
            x (px)
            <input value={x} inputMode="decimal" onChange={(e) => setX(e.target.value)} />
          </label>
          <label>
            y (px)
            <input value={y} inputMode="decimal" onChange={(e) => setY(e.target.value)} />
          </label>
          <label>
            keyframes
            <input value={numKf} inputMode="numeric" onChange={(e) => setNumKf(e.target.value)} />
          </label>
          <button onClick={() => void onSegment()} disabled={segmenting}>
            {segmenting ? 'Segmenting…' : 'Segment'}
          </button>
        </div>
        {segmentErr && <p className="error">{segmentErr}</p>}
        {segment && (
          <div>
            <p>
              <span className="pill ok">{segment.job_id}</span>{' '}
              <span className="muted">
                {segment.num_keyframes} keyframes · {segment.meta.sam_backend} ·{' '}
                {segment.meta.total_frames} frames @ {segment.meta.fps.toFixed(1)} fps
              </span>
            </p>
            <div className="kf-grid">
              {segment.keyframes.map((k) => (
                <figure key={k.index}>
                  <img src={`data:image/png;base64,${k.png_b64}`} alt={`keyframe ${k.index}`} />
                  <figcaption className="mono small">
                    f{k.index} · [{k.bbox_xyxy.join(',')}]
                  </figcaption>
                </figure>
              ))}
            </div>
          </div>
        )}
      </section>

      <section className="card">
        <h2>3 · Geometry + agent (run manifest)</h2>
        <div className="form-row">
          <label className="grow">
            Artifacts URL (manifest.json + geometry.json)
            <input value={artifactsUrl} onChange={(e) => setArtifactsUrl(e.target.value)} />
          </label>
          <button onClick={() => void loadArtifacts(artifactsUrl)} disabled={loadingArtifacts}>
            {loadingArtifacts ? 'Loading…' : 'Load'}
          </button>
        </div>
        {artifactsErr && (
          <div>
            <p className="error">{artifactsErr}</p>
            <p className="muted">Generate a run and serve it over HTTP, e.g.:</p>
            <pre className="mono small">
              {`python -m baseten.infer --keyframes-dir ./kf --no-remote\npython -m agent.loop --geometry ./kf/geometry.json --exec modal\ncd agent/runs/<run_id> && python -m http.server 8080`}
            </pre>
          </div>
        )}
        {geometry && (
          <p className="muted">
            Input · {geometry.source} · {geometry.vertices.length} vertices ·{' '}
            {geometry.edges.length} edges · {geometry.faces ? geometry.faces.length : 0} faces ·{' '}
            {geometry.source_keyframes.length} views · conf {geometry.confidence.toFixed(2)} · h≈
            {geometry.estimated_height_m} m
          </p>
        )}
        {manifest && (
          <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Cycle</th>
                <th>Kind</th>
                <th>Exit</th>
                <th>OK</th>
                <th>Mesh out</th>
              </tr>
            </thead>
            <tbody>
              {manifest.cycles.map((c) => (
                <tr key={c.cycle}>
                  <td>{c.cycle}</td>
                  <td>{c.kind}</td>
                  <td>{c.exit_code}</td>
                  <td>{c.report?.ok ? 'yes' : 'no'}</td>
                  <td className="small">
                    {(c.report?.errors ?? []).join('; ') ||
                      `v=${c.report?.stats?.vertices ?? '?'} f=${c.report?.stats?.faces ?? '?'}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        )}
        {manifest && (
          <p>
            <span className={`pill ${manifest.status === 'success' ? 'ok' : 'warn'}`}>{manifest.status}</span>{' '}
            <span className="muted">
              backend={manifest.backend} · usdz={manifest.usdz?.ok ? manifest.usdz.converter : 'none'}
            </span>
          </p>
        )}
      </section>

      <section className="card">
        <h2>4 · WebAR viewer + QR</h2>
        <ArViewer glb={MODEL_GLB} usdz={MODEL_USDZ} title="Object cloned with VOXEL-SAM AR" />
      </section>

      <footer className="muted small">
        VOXEL-SAM AR · hackathon demo · GLB for Android/desktop, USDZ for iOS (Quick Look)
      </footer>
    </div>
  );
}
