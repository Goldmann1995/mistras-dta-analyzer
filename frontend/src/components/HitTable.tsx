import { useState, useEffect, useCallback } from 'react';
import { ChevronLeft, ChevronRight, ArrowUpDown } from 'lucide-react';
import { getHits } from '../services/api';
import type { FileInfo, HitRecord } from '../types';

interface Props {
  file: FileInfo;
  onSelectHit?: (index: number) => void;
}

const COLUMNS: { key: keyof HitRecord; label: string; sortKey: string }[] = [
  { key: 'index', label: '#', sortKey: '' },
  { key: 'time', label: '时间(s)', sortKey: 'SSSSSSSS.mmmuuun' },
  { key: 'channel', label: '通道', sortKey: 'CH' },
  { key: 'amplitude', label: '振幅', sortKey: 'AMP' },
  { key: 'energy', label: '能量', sortKey: 'ENER' },
  { key: 'duration', label: '持续(μs)', sortKey: 'DURATION' },
  { key: 'rise', label: '上升时间', sortKey: 'RISE' },
  { key: 'counts', label: '计数', sortKey: 'COUN' },
  { key: 'peak_counts', label: '峰值计数', sortKey: 'PCNTS' },
  { key: 'rms', label: 'RMS', sortKey: 'RMS' },
  { key: 'avg_frequency', label: '平均频率', sortKey: 'A-FRQ' },
  { key: 'peak_frequency', label: '峰值频率', sortKey: 'P-FRQ' },
  { key: 'abs_energy', label: '绝对能量', sortKey: 'ABS-ENERGY' },
];

const PAGE_SIZE = 50;

export default function HitTable({ file, onSelectHit }: Props) {
  const [hits, setHits] = useState<HitRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [sortBy, setSortBy] = useState<string | undefined>();
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  const [channelFilter, setChannelFilter] = useState<number | undefined>();
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  const fetchData = useCallback(async () => {
    const res = await getHits(file.file_id, {
      offset: page * PAGE_SIZE,
      limit: PAGE_SIZE,
      sort_by: sortBy,
      sort_order: sortOrder,
      channel: channelFilter,
    });
    setHits(res.hits);
    setTotal(res.total);
  }, [file.file_id, page, sortBy, sortOrder, channelFilter]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleSort = (sortKey: string) => {
    if (!sortKey) return;
    if (sortBy === sortKey) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(sortKey);
      setSortOrder('asc');
    }
    setPage(0);
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className="hit-table-container">
      <div className="table-toolbar">
        <div className="filter-group">
          <label>通道:</label>
          <select
            value={channelFilter ?? ''}
            onChange={(e) => {
              setChannelFilter(e.target.value ? Number(e.target.value) : undefined);
              setPage(0);
            }}
          >
            <option value="">全部</option>
            {file.channels.map((ch) => (
              <option key={ch} value={ch}>CH{ch}</option>
            ))}
          </select>
        </div>
        <div className="table-info">
          共 {total.toLocaleString()} 条记录
        </div>
      </div>

      <div className="table-scroll">
        <table className="hit-table">
          <thead>
            <tr>
              {COLUMNS.map(({ key, label, sortKey }) => (
                <th
                  key={key}
                  onClick={() => handleSort(sortKey)}
                  className={sortKey ? 'sortable' : ''}
                >
                  {label}
                  {sortBy === sortKey && (
                    <ArrowUpDown size={12} className={`sort-icon ${sortOrder}`} />
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {hits.map((hit) => (
              <tr
                key={hit.index}
                className={selectedIdx === hit.index ? 'selected' : ''}
                onClick={() => {
                  setSelectedIdx(hit.index);
                  onSelectHit?.(hit.index);
                }}
              >
                {COLUMNS.map(({ key }) => (
                  <td key={key}>
                    {key === 'time'
                      ? (hit[key] as number)?.toFixed(6)
                      : key === 'rms'
                        ? (hit[key] as number)?.toFixed(4)
                        : key === 'abs_energy'
                          ? (hit[key] as number)?.toFixed(4)
                          : hit[key] ?? '-'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="table-pagination">
        <button onClick={() => setPage(Math.max(0, page - 1))} disabled={page === 0}>
          <ChevronLeft size={16} />
        </button>
        <span>{page + 1} / {totalPages || 1}</span>
        <button onClick={() => setPage(Math.min(totalPages - 1, page + 1))} disabled={page >= totalPages - 1}>
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
}
