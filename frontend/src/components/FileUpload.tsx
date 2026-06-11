import { useState, useCallback } from 'react';
import { uploadFile } from '../services/api';
import type { FileInfo } from '../types';

interface Props {
  onFileLoaded: (file: FileInfo) => void;
  compact?: boolean;
}

export default function FileUpload({ onFileLoaded, compact }: Props) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.dta')) {
      setError('Only .DTA files accepted');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const info = await uploadFile(file);
      onFileLoaded(info);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  }, [onFileLoaded]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  if (compact) {
    return (
      <label className="upload-compact">
        <input
          type="file" accept=".dta,.DTA" style={{ display: 'none' }}
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />
        {loading ? <span className="loading-dot" /> : <span>Load file</span>}
      </label>
    );
  }

  return (
    <div
      className={`upload-zone ${dragging ? 'dragging' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        type="file" accept=".dta,.DTA" id="file-input" style={{ display: 'none' }}
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />
      <label htmlFor="file-input" className="upload-content">
        {loading ? (
          <div className="upload-loading">
            <div className="loading-bar" />
            <span>Parsing binary data...</span>
          </div>
        ) : (
          <>
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" opacity={0.5}>
              <polyline points="2,12 6,8 10,16 14,4 18,14 22,10" />
            </svg>
            <span className="upload-title">Drop .DTA file here</span>
            <span className="upload-hint">or click to browse</span>
          </>
        )}
      </label>
      {error && <div className="upload-error">{error}</div>}
    </div>
  );
}
