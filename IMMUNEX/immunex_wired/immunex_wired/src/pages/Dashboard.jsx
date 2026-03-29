import React, { useState, useEffect } from 'react';
import { getOrchestratorHealth, getLayerHealthStats, getMeshNodes, getPendingApprovals } from '../api/immunexApi';

export default function Dashboard({ navigateTo }) {
  const [expandedRows, setExpandedRows] = useState({});
  const toggleRow = (i) => setExpandedRows(prev => ({ ...prev, [i]: !prev[i] }));

  const [layerHealth, setLayerHealth] = useState([
    { name: 'L1 Detection',     latency: '—', online: null },
    { name: 'L2 Correlation',   latency: '—', online: null },
    { name: 'L3 Response',      latency: '—', online: null },
    { name: 'L4 Immunity',      latency: '—', online: null },
    { name: 'L5 Threat Memory', latency: '—', online: null },
  ]);
  const [meshNodes, setMeshNodes]   = useState([]);
  const [pendingCount, setPendingCount] = useState('—');
  const [liveAlerts, setLiveAlerts] = useState([]);
  const [lastRefresh, setLastRefresh] = useState(null);
  const [orchestratorOnline, setOrchestratorOnline] = useState(null);

  const criticalCount = liveAlerts.filter(a => a.sev === 'critical').length;
  const activeCount   = liveAlerts.filter(a => a.status === 'active' || a.status === 'unresolved').length;

  const fetchHealth = async () => {
    try {
      await getOrchestratorHealth();
      setOrchestratorOnline(true);
      const stats = await getLayerHealthStats();
      setLayerHealth(stats);
      setLastRefresh(new Date().toLocaleTimeString('en', { hour12: false }));
    } catch {
      setOrchestratorOnline(false);
    }
  };

  const fetchNodes = async () => {
    try {
      const d = await getMeshNodes();
      const nodes = d.nodes || {};
      setMeshNodes(Object.entries(nodes).map(([url, info]) => ({
        url, alive: info.alive || info.online,
        gpu: info.gpu_eligible, latency: info.latency_ms,
        device: info.device,
      })));
    } catch {}
  };

  const fetchPending = async () => {
    try {
      const raw = await getPendingApprovals();
      const items = Array.isArray(raw) ? raw : (raw.pending || raw.items || []);
      setPendingCount(items.length);
    } catch {}
  };

  useEffect(() => {
    fetchHealth();
    fetchNodes();
    fetchPending();
    const iv = setInterval(() => { fetchHealth(); fetchNodes(); fetchPending(); }, 30000);
    return () => clearInterval(iv);
  }, []);

  const onlineCount = layerHealth.filter(l => l.online).length;

  return (
    <div className="section-page active" style={{ display: 'block' }}>

      {/* ── Hero ── */}
      <div className="dash-hero">
        <div>
          <div className="section-eyebrow">System Overview</div>
          <h1 className="section-headline">Your threat landscape,<br />right now.</h1>
          <p className="section-sub" style={{ marginBottom: 24 }}>
            Live monitoring across all detection surfaces. Expert-curated intelligence streaming in real-time.
          </p>
          <div className="dash-actions">
            <button className="btn btn-primary"   onClick={() => navigateTo('detect')}>Inject Logs</button>
            <button className="btn btn-secondary" onClick={() => navigateTo('attack-graph')}>View Threats</button>
          </div>
        </div>
        <div className="terminal-card">
          <div className="terminal-header">
            <div className="term-dot r"></div>
            <div className="term-dot y"></div>
            <div className="term-dot g"></div>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', marginLeft: 8, letterSpacing: 1 }}>immunex@detect:~</span>
          </div>
          <div className="terminal-body">
            <span className="term-line sys">[SYS] Initializing entropy pool...</span>
            <span className="term-line sys">[MON] Stream connection: wss://detect.immunex.ai/v4</span>
            <span className="term-line sys">[MON] Packet inspection active on eth0</span>
            <span className="term-line alert">[ALRT] High frequency pattern detected: 192.168.1.104</span>
            <span className="term-line sys">[SYS] Kernel hardening verified.</span>
            <span className="term-line info">[INTL] MITRE T1059.001 match found in local buffer</span>
            <span className="term-line sys">&gt; <span className="term-cursor"></span></span>
          </div>
        </div>
      </div>

      <div className="section-body">

        {/* ── Stats Row ── */}
        <div className="stats-row">
          <div className="stat-block">
            <div className="stat-label">Total Incidents</div>
            <span className="stat-num">{liveAlerts.length}</span>
          </div>
          <div className="stat-block">
            <div className="stat-label">Active Threats</div>
            <span className="stat-num red">{activeCount}</span>
          </div>
          <div className="stat-block">
            <div className="stat-label">Pending Actions</div>
            <span className="stat-num">{pendingCount}</span>
          </div>
          <div className="stat-block">
            <div className="stat-label">Critical Alerts</div>
            <span className="stat-num" style={{ color: '#c0392b' }}>{criticalCount}</span>
          </div>
          <div className="stat-block">
            <div className="stat-label">Layers Online</div>
            <span className="stat-num accent">
              {layerHealth.every(l => l.online === null) ? '—' : `${onlineCount}/5`}
            </span>
          </div>
        </div>

        {/* ── Two Column: Alerts Table + Severity Breakdown ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 24, marginBottom: 32 }}>

          {/* Alerts Table */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <h2 className="section-title" style={{ margin: 0 }}>Recent Intelligence Alerts</h2>
              <button className="btn btn-secondary btn-sm" onClick={() => navigateTo('alerts')}>View All →</button>
            </div>
            <div className="card" style={{ padding: 0 }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Time</th>
                    <th>Type</th>
                    <th>Severity</th>
                    <th>Source IP</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {liveAlerts.slice(0, 8).map((a, i) => (
                    <React.Fragment key={i}>
                      <tr onClick={() => toggleRow(i)} style={{ cursor: 'pointer' }}>
                        <td style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 600 }}>{a.id}</td>
                        <td style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>{a.ts}</td>
                        <td style={{ fontWeight: 500 }}>{a.type}</td>
                        <td><span className={`sev-tag sev-${a.sev}`}>{a.sev}</span></td>
                        <td style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>{a.ip}</td>
                        <td><span className={`status-tag status-${a.status}`}>{a.status}</span></td>
                      </tr>
                      {expandedRows[i] && (
                        <tr>
                          <td colSpan={6} style={{ background: 'var(--bg2)', padding: '12px 16px' }}>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, fontFamily: 'var(--mono)', fontSize: 11 }}>
                              <div><div style={{ color: 'var(--text3)', marginBottom: 3 }}>MITRE Tactic</div><div style={{ fontWeight: 600 }}>{a.mitre}</div></div>
                              <div><div style={{ color: 'var(--text3)', marginBottom: 3 }}>Attack Vector</div><div style={{ fontWeight: 600 }}>{a.vector}</div></div>
                              <div><div style={{ color: 'var(--text3)', marginBottom: 3 }}>Recommendation</div><div style={{ fontWeight: 600 }}>{a.rec}</div></div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Severity Breakdown */}
          <div>
            <h2 className="section-title" style={{ marginBottom: 14 }}>Severity Breakdown</h2>
            <div className="card">
              {['critical', 'high', 'medium', 'low'].map(sev => {
                const count = liveAlerts.filter(a => a.sev === sev).length;
                const pct   = liveAlerts.length ? Math.round((count / liveAlerts.length) * 100) : 0;
                return (
                  <div key={sev} style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                      <span className={`sev-tag sev-${sev}`}>{sev}</span>
                      <span style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 600 }}>{count}</span>
                    </div>
                    <div style={{ height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${pct}%`, borderRadius: 3, background: sev === 'critical' ? '#c0392b' : sev === 'high' ? '#d4680a' : sev === 'medium' ? '#b8860b' : '#3d5a3e' }}></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* ── System Health (live) ── */}
        <div style={{ marginBottom: 32 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <h2 className="section-title" style={{ margin: 0 }}>System Health</h2>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              {lastRefresh && (
                <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)' }}>
                  Last checked: {lastRefresh}
                </span>
              )}
              <button className="btn btn-secondary btn-sm" onClick={fetchHealth}>Refresh</button>
            </div>
          </div>
          <div className="health-grid">
            {layerHealth.map(({ name, latency, online }) => (
              <div key={name} className="health-item">
                <span className="health-name">{name}</span>
                <div className="health-status">
                  <span className="health-latency">{latency}</span>
                  <div className={`health-dot ${online === null ? '' : online ? '' : 'offline'}`}></div>
                </div>
              </div>
            ))}
            <div className="health-item">
              <span className="health-name">Orchestrator</span>
              <div className="health-status">
                <span className="health-latency">—</span>
                <div className={`health-dot ${orchestratorOnline === null ? '' : orchestratorOnline ? '' : 'offline'}`}></div>
              </div>
            </div>
            {meshNodes.slice(0, 3).map(n => {
              const label = n.url.replace(/https?:\/\//, '').replace(':8000', '');
              return (
                <div key={n.url} className="health-item">
                  <span className="health-name">{label}</span>
                  <div className="health-status">
                    <span className="health-latency">{n.latency ? n.latency.toFixed(0) + 'ms' : '—'}</span>
                    <div className={`health-dot ${n.alive ? '' : 'offline'}`}></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* ── Quick Actions ── */}
        <div>
          <h2 className="section-title" style={{ marginBottom: 14 }}>Quick Actions</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
            {[
              { label: 'Inject Logs',    icon: '⬆', section: 'detect',       desc: 'Feed telemetry into detection engine' },
              { label: 'View Playbook',  icon: '📋', section: 'playbook',     desc: 'Review pending response actions' },
              { label: 'Attack Graph',   icon: '🕸', section: 'attack-graph', desc: 'Visualize threat kill chain' },
              { label: 'AIS Engine',     icon: '🧠', section: 'ais',          desc: 'Trigger adaptive retraining' },
            ].map(({ label, icon, section, desc }) => (
              <button key={section} onClick={() => navigateTo(section)}
                style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10, padding: '16px 18px', textAlign: 'left', cursor: 'pointer', transition: 'all .2s' }}
                onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--accentbg)'}
                onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
              >
                <div style={{ fontSize: 22, marginBottom: 8 }}>{icon}</div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 600, letterSpacing: 1, textTransform: 'uppercase', color: 'var(--text)', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 11, color: 'var(--text3)', lineHeight: 1.4 }}>{desc}</div>
              </button>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
