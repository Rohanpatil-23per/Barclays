import React from 'react';
import { REPORT_DATA } from '../data/mock';

export default function Reports() {
  const downloadReport = (r) => {
    const csv = `ID,Date,Status,Title,Description\n"${r.id}","${r.date}","${r.status}","${r.title}","${r.desc}"`;
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${r.id}.csv`;
    a.click();
  };

  return (
    <div className="section-page active" style={{ display: 'block' }}>
      <div className="section-page-header">
        <div className="section-eyebrow">Incident Archive</div>
        <h2 className="section-title">Audit Trail</h2>
      </div>

      <div className="section-body" id="reports-body">
        {REPORT_DATA.map((r, i) => (
          <div key={r.id} className={`report-card ${r.status}`}>
            <div className="report-top">
              <div>
                <div className="report-meta">
                  <span className="report-id">{r.id}</span>
                  <span className="report-date">{r.date}</span>
                  <span className={`report-status-badge ${r.status}`}>{r.status.toUpperCase()}</span>
                </div>
                <div className="report-title">{r.title}</div>
                <div className="report-tags">
                  {r.tags.map(t => <span key={t} className="report-tag">{t}</span>)}
                  <span className="report-impact">{r.impact}</span>
                </div>
                <div className="report-desc">{r.desc}</div>
              </div>
              
              <div className="report-actions">
                <button className="btn btn-secondary btn-sm" onClick={() => downloadReport(r)}>↓ Download CSV</button>
                <button className="btn btn-secondary btn-sm" onClick={() => window.print()}>🖨 Print</button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
