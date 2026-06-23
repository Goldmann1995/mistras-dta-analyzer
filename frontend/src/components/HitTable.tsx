import { useState, useEffect, useCallback } from 'react';
import { getHits } from '../services/api';
import type { FileInfo, HitRecord } from '../types';

interface Props {
  file: FileInfo;
  onSelectHit?: (index: number) => void;
}

const COLS: { key: keyof HitRecord; label: string; sortKey: string; fmt?: (v: number) => string }[] = [
  { key: 'index', label: '#', sortKey: '' },
  { key: 'time', label: 'Time (s)', sortKey: 'SSSSSSSS.mmmuuun', fmt: (v) => v.toFixed(6) },
  { key: 'channel', label: 'CH', sortKey: 'CH' },
  { key: 'amplitude', label: 'Amp', sortKey: 'AMP' },
  { key: 'energy', label: 'Energy', sortKey: 'ENER' },
  { key: 'duration', label: 'Dur (us)', sortKey: 'DURATION' },
  { key: 'rise', label: 'Rise', sortKey: 'RISE' },
  { key: 'counts', label: 'Counts', sortKey: 'COUN' },
  { key: 'rms', label: 'RMS', sortKey: 'RMS', fmt: (v) => v.toFixed(4) },
  { key: 'peak_frequency', label: 'P-FRQ', sortKey: 'P-FRQ' },
  { key: 'abs_energy', label: 'Abs Energy', sortKey: 'ABS-ENERGY', fmt: (v) => v.toFixed(4) },
  { key: 'entropy', label: 'Entropy', sortKey: '', fmt: (v) => v != null ? v.toFixed(4) : '' },
];

const PAGE = 50;

export default function HitTable({ file, onSelectHit }: Props) {
  const [hits, setHits] = useState<HitRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [sortBy, setSortBy] = useState<string | undefined>();
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  const [chFilter, setChFilter] = useState<number | undefined>();
  const [selected, setSelected] = useState<number | null>(null);

  const fetch = useCallback(async () => {
    const r = await getHits(file.file_id, {
      offset: page * PAGE, limit: PAGE,
      sort_by: sortBy, sort_order: sortOrder, channel: chFilter,
    });
    setHits(r.hits);
    setTotal(r.total);
  }, [file.file_id, page, sortBy, sortOrder, chFilter]);

  useEffect(() => { fetch(); }, [fetch]);

  const handleSort = (sk: string) => {
    if (!sk) return;
    if (sortBy === sk) setSortOrder(o => o === 'asc' ? 'desc' : 'asc');
    else { setSortBy(sk); setSortOrder('asc'); }
    setPage(0);
  };

  const pages = Math.ceil(total / PAGE);

  return (
    <div className="view-hits">
      <div className="hits-toolbar">
        <div className="toolbar-left">
          <label className="filter-label">Channel</label>
          <select className="filter-select" value={chFilter ?? ''} onChange={(e) => { setChFilter(e.target.value ? Number(e.target.value) : undefined); setPage(0); }}>
            <option value="">All</option>
            {file.channels.map(ch => <option key={ch} value={ch}>CH{ch}</option>)}
          </select>
        </div>
        <span className="toolbar-count">{total.toLocaleString()} records</span>
      </div>

      <div className="table-wrap">
        <table className="data-table hit-table">
          <thead>
            <tr>
              {COLS.map(({ key, label, sortKey }) => (
                <th key={key} onClick={() => handleSort(sortKey)} className={sortKey ? 'sortable' : ''}>
                  {label}
                  {sortBy === sortKey && <span className="sort-arrow">{sortOrder === 'asc' ? ' ↑' : ' ↓'}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {hits.map((h) => (
              <tr key={h.index} className={selected === h.index ? 'selected' : ''} onClick={() => { setSelected(h.index); onSelectHit?.(h.index); }}>
                {COLS.map(({ key, fmt }) => {
                  const v = h[key];
                  return <td key={key}>{v == null ? '-' : fmt ? fmt(v as number) : v}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="table-pager">
        <button disabled={page === 0} onClick={() => setPage(p => p - 1)}>Prev</button>
        <span className="pager-info">{page + 1} / {pages || 1}</span>
        <button disabled={page >= pages - 1} onClick={() => setPage(p => p + 1)}>Next</button>
      </div>
    </div>
  );
}
