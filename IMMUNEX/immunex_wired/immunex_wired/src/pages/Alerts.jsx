import React, { useState } from 'react';
import { ALERT_DATA } from '../data/mock';

export default function Alerts() {
  const [filter, setFilter] = useState('all');
  const [expandedRows, setExpandedRows] = useState({});

  const toggleRow = (i) => setExpandedRows(prev => ({...prev, [i]: !prev[i]}));

  const filteredData = filter === 'all' 
    ? ALERT_DATA 
    : ALERT_DATA.filter(a => a.sev === filter);

  return (
    <div className="section-page active" style={{ display: 'block' }}>
      <div className="section-page-header">
        <div className="section-eyebrow">Threat Feed</div>
        <h2 className="section-title">Live Intelligence Alerts</h2>
      </div>

      <div className="section-body">
        <div className="alerts-filter-row">
          <div className="filter-pills">
            <button className={`filter-pill ${filter === 'all' ? 'active' : ''}`} onClick={() => setFilter('all')}>All Alerts</button>
            <button className={`filter-pill ${filter === 'critical' ? 'active' : ''}`} onClick={() => setFilter('critical')}>Critical</button>
            <button className={`filter-pill ${filter === 'high' ? 'active' : ''}`} onClick={() => setFilter('high')}>High Severity</button>
            <button className={`filter-pill ${filter === 'medium' ? 'active' : ''}`} onClick={() => setFilter('medium')}>Medium / Low</button>
          </div>
          <div className="alerts-count">{filteredData.length} records found</div>
        </div>

        <div className="card">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th><th>Timestamp</th><th>Type</th><th>Severity</th><th>Source IP</th><th>Confidence</th><th>Status</th>
              </tr>
            </thead>
            <tbody>
              {filteredData.map((a, i) => (
                <React.Fragment key={a.id}>
                  <tr onClick={() => toggleRow(i)}>
                    <td style={{fontFamily:'var(--mono)', fontSize:12, fontWeight:600}}>{a.id}</td>
                    <td style={{fontFamily:'var(--mono)', fontSize:11}}>{a.ts}</td>
                    <td style={{fontWeight:500}}>{a.type}</td>
                    <td><span className={`badge ${a.sev}`}>{a.sev}</span></td>
                    <td style={{fontFamily:'var(--mono)', fontSize:12}}>{a.ip}</td>
                    <td style={{fontFamily:'var(--mono)', fontSize:12}}>{['98%','82%','94%','65%','91%','87%','99%','76%','71%','84%'][i]||'—'}</td>
                    <td><span className={`status-dot ${a.status}`}>{a.status}</span></td>
                  </tr>
                  {expandedRows[i] && (
                    <tr className="expand-row open">
                      <td colSpan="7">
                        <div className="expand-content">
                          <div className="expand-grid">
                            <div><div className="expand-field-label">MITRE Attack Tactic</div><div className="expand-field-value">{a.mitre}</div></div>
                            <div><div className="expand-field-label">Vector</div><div className="expand-field-value">{a.vector}</div></div>
                            <div><div className="expand-field-label">Detection Engine</div><div className="expand-field-value">{a.response}</div></div>
                            <div style={{marginTop:8}}>
                              <div className="expand-field-label">Rec. Response</div>
                              <div className="expand-field-value" style={{color:'var(--accentbg)'}}>{a.rec}</div>
                            </div>
                          </div>
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
    </div>
  );
}
