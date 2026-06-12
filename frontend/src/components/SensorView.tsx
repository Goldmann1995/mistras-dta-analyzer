import { useState, useCallback, useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line, Html } from '@react-three/drei';
import * as THREE from 'three';
import { getSourceLocation } from '../services/api';
import type { FileInfo, SourceLocationResult, SourceEvent } from '../types';

interface Props { file: FileInfo; }

function GridPlane({ size }: { size: number }) {
  return (
    <gridHelper args={[size, 20, '#1a2234', '#111827']} rotation={[0, 0, 0]} />
  );
}

function AxisLabels({ size }: { size: number }) {
  const half = size / 2;
  return (
    <>
      <Line points={[[0, 0, 0], [half, 0, 0]]} color="#f87171" lineWidth={2} />
      <Line points={[[0, 0, 0], [0, half, 0]]} color="#34d399" lineWidth={2} />
      <Line points={[[0, 0, 0], [0, 0, half]]} color="#60a5fa" lineWidth={2} />
      <Text position={[half + 0.02, 0, 0]} fontSize={0.03} color="#f87171" anchorX="left">X</Text>
      <Text position={[0, half + 0.02, 0]} fontSize={0.03} color="#34d399" anchorX="left">Y</Text>
      <Text position={[0, 0, half + 0.02]} fontSize={0.03} color="#60a5fa" anchorX="left">Z</Text>
    </>
  );
}

function SensorMarker({ position, channel, isSelected, onClick }: {
  position: [number, number, number]; channel: number; isSelected: boolean;
  onClick: () => void;
}) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(() => {
    if (ref.current && isSelected) {
      ref.current.scale.setScalar(1 + Math.sin(Date.now() * 0.005) * 0.2);
    }
  });

  return (
    <group position={position}>
      <mesh ref={ref} onClick={onClick}>
        <octahedronGeometry args={[0.015, 0]} />
        <meshStandardMaterial color={isSelected ? '#22d3ee' : '#a78bfa'} emissive={isSelected ? '#22d3ee' : '#a78bfa'} emissiveIntensity={0.5} />
      </mesh>
      <Html distanceFactor={0.5} style={{ pointerEvents: 'none' }}>
        <div style={{
          background: 'rgba(12,18,34,0.9)', border: '1px solid rgba(255,255,255,0.15)',
          borderRadius: 4, padding: '2px 6px', fontSize: 10, color: '#e2e8f0',
          fontFamily: 'IBM Plex Mono, monospace', whiteSpace: 'nowrap',
        }}>
          CH{channel}
        </div>
      </Html>
    </group>
  );
}

function EventDot({ position, amplitude, maxAmp, selected, onClick }: {
  position: [number, number, number]; amplitude: number; maxAmp: number;
  selected: boolean; onClick: () => void;
}) {
  const norm = maxAmp > 0 ? amplitude / maxAmp : 0.5;
  const color = new THREE.Color().setHSL(0.08 - norm * 0.08, 0.9, 0.5 + norm * 0.3);
  const size = 0.004 + norm * 0.008;

  return (
    <mesh position={position} onClick={onClick}>
      <sphereGeometry args={[selected ? size * 1.8 : size, 12, 12]} />
      <meshStandardMaterial
        color={selected ? '#ffffff' : color}
        emissive={selected ? '#22d3ee' : color}
        emissiveIntensity={selected ? 1 : 0.3}
        transparent opacity={selected ? 1 : 0.7}
      />
    </mesh>
  );
}

function Scene({ sensors, events, maxAmp, selectedEvent, onSelectEvent, onSelectSensor, selectedSensor, gridSize }: {
  sensors: { ch: number; pos: [number, number, number] }[];
  events: (SourceEvent & { loc3: [number, number, number] })[];
  maxAmp: number; selectedEvent: number | null; selectedSensor: number | null;
  onSelectEvent: (i: number) => void; onSelectSensor: (ch: number) => void;
  gridSize: number;
}) {
  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[1, 1, 1]} intensity={1} />
      <GridPlane size={gridSize} />
      <AxisLabels size={gridSize} />
      {sensors.map(s => (
        <SensorMarker key={s.ch} position={s.pos} channel={s.ch}
          isSelected={selectedSensor === s.ch} onClick={() => onSelectSensor(s.ch)} />
      ))}
      {events.map((e, i) => (
        <EventDot key={i} position={e.loc3} amplitude={e.amplitude} maxAmp={maxAmp}
          selected={selectedEvent === i} onClick={() => onSelectEvent(i)} />
      ))}
      <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
    </>
  );
}

export default function SensorView({ file }: Props) {
  const [positions, setPositions] = useState<Record<number, [number, number, number]>>(() => {
    const init: Record<number, [number, number, number]> = {};
    const n = file.channels.length;
    file.channels.forEach((ch, i) => {
      const angle = (2 * Math.PI * i) / n;
      init[ch] = [
        Number((0.1 * Math.cos(angle)).toFixed(4)),
        Number((0.1 * Math.sin(angle)).toFixed(4)),
        0,
      ];
    });
    return init;
  });

  const [velocity, setVelocity] = useState(5400);
  const [timeWindow, setTimeWindow] = useState(1);
  const [result, setResult] = useState<SourceLocationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<number | null>(null);
  const [selectedSensor, setSelectedSensor] = useState<number | null>(null);
  const [colorBy, setColorBy] = useState<'amplitude' | 'energy' | 'time'>('amplitude');

  const compute = useCallback(async () => {
    setLoading(true);
    try {
      const posMap: Record<string, number[]> = {};
      for (const [ch, pos] of Object.entries(positions)) {
        posMap[String(ch)] = pos;
      }
      const r = await getSourceLocation(file.file_id, posMap, velocity, timeWindow / 1000);
      setResult(r);
      setSelectedEvent(null);
    } finally {
      setLoading(false);
    }
  }, [file.file_id, positions, velocity, timeWindow]);

  const updatePos = (ch: number, axis: number, value: number) => {
    setPositions(prev => {
      const next = { ...prev };
      const p: [number, number, number] = [...(next[ch] || [0, 0, 0])] as [number, number, number];
      p[axis] = value;
      next[ch] = p;
      return next;
    });
  };

  const locatedEvents = useMemo(() => {
    if (!result) return [];
    return result.events
      .filter(e => e.location !== null)
      .map(e => ({
        ...e,
        loc3: (e.location!.length >= 3
          ? [e.location![0], e.location![1], e.location![2]]
          : e.location!.length === 2
          ? [e.location![0], e.location![1], 0]
          : [e.location![0], 0, 0]) as [number, number, number],
      }));
  }, [result]);

  const maxAmp = useMemo(() => {
    if (locatedEvents.length === 0) return 1;
    return Math.max(...locatedEvents.map(e => e.amplitude));
  }, [locatedEvents]);

  const sensors = useMemo(() =>
    Object.entries(positions).map(([ch, pos]) => ({
      ch: Number(ch),
      pos: pos as [number, number, number],
    })),
  [positions]);

  const gridSize = useMemo(() => {
    const allCoords = sensors.map(s => s.pos).flat();
    locatedEvents.forEach(e => allCoords.push(...e.loc3));
    const maxCoord = Math.max(0.1, ...allCoords.map(Math.abs));
    return Math.ceil(maxCoord * 2 * 10) / 10 + 0.1;
  }, [sensors, locatedEvents]);

  const selEvt = selectedEvent !== null ? locatedEvents[selectedEvent] : null;

  return (
    <div className="view-wavelet">
      <div className="panel-grid-2">
        <div className="panel" style={{ maxHeight: 360, overflowY: 'auto' }}>
          <div className="panel-head">Sensor Positions (m)</div>
          <table className="hit-table" style={{ fontSize: 12 }}>
            <thead>
              <tr><th>CH</th><th>X</th><th>Y</th><th>Z</th></tr>
            </thead>
            <tbody>
              {file.channels.map(ch => (
                <tr key={ch} className={selectedSensor === ch ? 'selected' : ''} onClick={() => setSelectedSensor(ch)}>
                  <td style={{ fontWeight: 600, color: '#a78bfa' }}>CH{ch}</td>
                  {[0, 1, 2].map(axis => (
                    <td key={axis}>
                      <input type="number" step={0.01} value={positions[ch]?.[axis] ?? 0}
                        onChange={e => updatePos(ch, axis, Number(e.target.value))}
                        style={{ width: 72, background: 'var(--bg-1)', border: '1px solid var(--border)',
                          color: 'var(--text-1)', padding: '3px 6px', borderRadius: 4,
                          fontFamily: 'var(--font-data)', fontSize: 12 }}
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>

          <div className="wf-controls" style={{ marginTop: 8 }}>
            <div className="ctrl-group">
              <label>Velocity (m/s)</label>
              <input type="number" min={100} step={100} value={velocity}
                onChange={e => setVelocity(Number(e.target.value))} style={{ width: 80 }} />
            </div>
            <div className="ctrl-group">
              <label>Window (ms)</label>
              <input type="number" min={0.1} step={0.1} value={timeWindow}
                onChange={e => setTimeWindow(Number(e.target.value))} style={{ width: 60 }} />
            </div>
            <button className="ctrl-btn" onClick={compute}>Locate</button>
          </div>
        </div>

        <div className="panel" style={{ padding: 0, overflow: 'hidden', minHeight: 360 }}>
          <Canvas camera={{ position: [gridSize * 0.6, gridSize * 0.6, gridSize * 0.6], fov: 50 }}
            style={{ background: '#060a13', borderRadius: 6 }}>
            <Scene sensors={sensors} events={locatedEvents} maxAmp={maxAmp}
              selectedEvent={selectedEvent} selectedSensor={selectedSensor}
              onSelectEvent={setSelectedEvent} onSelectSensor={setSelectedSensor}
              gridSize={gridSize} />
          </Canvas>
        </div>
      </div>

      {loading && <div className="loading-indicator">Computing source locations...</div>}

      {result && (
        <>
          <div className="metrics-row compact">
            <div className="metric">
              <span className="metric-val">{result.total_events}</span>
              <span className="metric-key">Total Events</span>
            </div>
            <div className="metric">
              <span className="metric-val">{result.located_events}</span>
              <span className="metric-key">Located</span>
            </div>
            <div className="metric">
              <span className="metric-val">{result.velocity}<small>m/s</small></span>
              <span className="metric-key">Velocity</span>
            </div>
            <div className="metric">
              <span className="metric-val">{Object.keys(result.sensor_positions).length}</span>
              <span className="metric-key">Sensors</span>
            </div>
          </div>

          {selEvt && (
            <div className="panel">
              <div className="panel-head">
                Selected Event
                <span className="panel-tag">
                  {selEvt.num_channels} channels · Amp: {selEvt.amplitude.toFixed(1)} dB
                </span>
              </div>
              <div className="wf-meta">
                <span>Time: {selEvt.time.toFixed(6)} s</span>
                <span>Channels: {selEvt.channels.join(', ')}</span>
                {selEvt.location && (
                  <span>Location: ({selEvt.location.map(v => v.toFixed(4)).join(', ')}) m</span>
                )}
              </div>
            </div>
          )}

          <div className="panel">
            <div className="panel-head">
              Located Events ({locatedEvents.length})
              <span className="panel-tag">Click row to highlight in 3D</span>
            </div>
            <div style={{ maxHeight: 300, overflowY: 'auto' }}>
              <table className="hit-table">
                <thead>
                  <tr>
                    <th>#</th><th>Time (s)</th><th>CH</th><th>Amp (dB)</th>
                    <th>X (m)</th><th>Y (m)</th><th>Z (m)</th>
                  </tr>
                </thead>
                <tbody>
                  {locatedEvents.slice(0, 200).map((e, i) => (
                    <tr key={i} onClick={() => setSelectedEvent(i)}
                      className={selectedEvent === i ? 'selected' : ''}
                      style={{ cursor: 'pointer' }}>
                      <td>{i + 1}</td>
                      <td>{e.time.toFixed(6)}</td>
                      <td>{e.channels.join(',')}</td>
                      <td>{e.amplitude.toFixed(1)}</td>
                      <td>{e.loc3[0].toFixed(4)}</td>
                      <td>{e.loc3[1].toFixed(4)}</td>
                      <td>{e.loc3[2].toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
