import { useEffect, useState } from 'react';
import type { CSSProperties, DetailedHTMLProps, HTMLAttributes, ReactNode } from 'react';
import { QRCodeSVG } from 'qrcode.react';
import '@google/model-viewer';

type ModelViewerProps = DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement> & {
  src?: string;
  'ios-src'?: string;
  alt?: string;
  ar?: boolean;
  'ar-modes'?: string;
  'camera-controls'?: boolean;
  'auto-rotate'?: boolean;
  'shadow-intensity'?: string;
  exposure?: string;
  'environment-image'?: string;
  children?: ReactNode;
  style?: CSSProperties;
};

declare module 'react' {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace JSX {
    interface IntrinsicElements {
      'model-viewer': ModelViewerProps;
    }
  }
}

interface Props {
  glb: string;
  usdz: string;
  title: string;
}

export default function ArViewer({ glb, usdz, title }: Props) {
  const [pageUrl, setPageUrl] = useState('');
  const [isLoopback, setIsLoopback] = useState(false);
  const [arStatus, setArStatus] = useState('Checking AR support…');

  useEffect(() => {
    const syncUrl = () => {
      setPageUrl(window.location.href);
      const h = window.location.hostname;
      setIsLoopback(h === 'localhost' || h === '127.0.0.1' || h === '[::1]' || h === '');
    };
    syncUrl();
    const update = () => syncUrl();
    window.addEventListener('hashchange', update);
    // model-viewer exposes canActivateAR after loading the model.
    const timer = setInterval(() => {
      const mv = document.querySelector('model-viewer') as unknown as {
        canActivateAR?: boolean;
      } | null;
      if (mv && typeof mv.canActivateAR === 'boolean') {
        setArStatus(
          mv.canActivateAR
            ? 'AR available on this device'
            : 'AR not available here — scan the QR with your phone',
        );
        clearInterval(timer);
      }
    }, 800);
    const stop = setTimeout(() => clearInterval(timer), 15000);
    return () => {
      window.removeEventListener('hashchange', update);
      clearInterval(timer);
      clearTimeout(stop);
    };
  }, []);

  return (
    <div className="ar-grid">
      <div className="viewer-wrap">
        <model-viewer
          src={glb}
          ios-src={usdz}
          alt={title}
          ar
          ar-modes="webxr scene-viewer quick-look"
          camera-controls
          auto-rotate
          shadow-intensity="1"
          exposure="0.9"
          environment-image="neutral"
          style={{ width: '100%', height: '420px', background: '#0b0e14', borderRadius: '12px' }}
        >
          <button slot="ar-button" className="ar-button">
            View in your space (AR)
          </button>
        </model-viewer>
        <p className="muted">{arStatus}</p>
      </div>
      <div className="qr-card">
        {pageUrl ? <QRCodeSVG value={pageUrl} size={200} level="M" /> : <p>Loading QR…</p>}
        {isLoopback && (
          <p className="small">
            You are on localhost: publish this web app or serve it on your LAN (`vite preview --host`) so the QR
            opens on your phone.
          </p>
        )}
        <p className="muted">
          Scan with your phone to open this same page and place the model next to the real object.
        </p>
        <p className="mono small">{pageUrl}</p>
      </div>
    </div>
  );
}
