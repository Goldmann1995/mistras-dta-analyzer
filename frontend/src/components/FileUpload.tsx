import { useState, useCallback } from 'react';
import { Upload, FileText, X } from 'lucide-react';
import { uploadFile } from '../services/api';
import type { FileInfo } from '../types';

interface Props {
  onFileLoaded: (file: FileInfo) => void;
}

export default function FileUpload({ onFileLoaded }: Props) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.dta')) {
      setError('请选择 .DTA 文件');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const info = await uploadFile(file);
      onFileLoaded(info);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '上传失败');
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

  return (
    <div
      className={`file-upload ${dragging ? 'dragging' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        type="file"
        accept=".dta,.DTA"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        id="file-input"
        style={{ display: 'none' }}
      />
      <label htmlFor="file-input" className="upload-label">
        {loading ? (
          <div className="upload-spinner">解析中...</div>
        ) : (
          <>
            <Upload size={32} />
            <span>拖放 DTA 文件到此处，或点击选择</span>
          </>
        )}
      </label>
      {error && (
        <div className="upload-error">
          <X size={14} /> {error}
        </div>
      )}
    </div>
  );
}
