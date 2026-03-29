import React, { useEffect, useRef, useState } from 'react';
import { NODES, TIMELINE_EVENTS } from '../data/mock';

// ── NOTE ─────────────────────────────────────────────────────────────────────
// Static mock data only — backend (Layer 2 /correlate) disconnected.
// To reconnect: import { correlateAttack } from '../api/immunexApi'
// ─────────────────────────────────────────────────────────────────────────────

export default function AttackGraph() {
  const canvasRef = useRef(null);
  const [animating, setAnimating] = useState(false);
  const [step, setStep] = useState(0);

  const drawGraph = (upTo) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = canvas.offsetWidth || 700;
    const H = canvas.offsetHeight || 400;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#ede8dc'; ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
    const getXY = (n) => ({ x: n.x * W, y: n.y * H });
    for (let i = 0; i < Math.min(upTo - 1, NODES.length - 1); i++) {
      const a = getXY(NODES[i]), b = getXY(NODES[i + 1]);
      ctx.beginPath(); ctx.setLineDash([6, 4]); ctx.moveTo(a.x, a.y);
      const cx = (a.x + b.x) / 2;
      ctx.bezierCurveTo(cx, a.y, cx, b.y, b.x, b.y);
      ctx.strokeStyle = i === NODES.length - 2 ? '#c0392b88' : '#3d5a3e66';
      ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
      const angle = Math.atan2(b.y - a.y, b.x - a.x);
      ctx.save(); ctx.translate(b.x - 18 * Math.cos(angle), b.y - 18 * Math.sin(angle)); ctx.rotate(angle);
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-8, -4); ctx.lineTo(-8, 4); ctx.closePath();
      ctx.fillStyle = i === NODES.length - 2 ? '#c0392b' : '#3d5a3e'; ctx.fill(); ctx.restore();
    }
    NODES.slice(0, upTo).forEach((n) => {
      const { x, y } = getXY(n); const isImpact = n.label === 'IMPACT'; const r = isImpact ? 30 : 26;
      ctx.shadowColor = isImpact ? '#c0392b44' : '#3d5a3e33'; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = isImpact ? '#fef0ef' : '#fff'; ctx.fill();
      ctx.strokeStyle = isImpact ? '#c0392b' : '#1a3a5c'; ctx.lineWidth = isImpact ? 2 : 1.5; ctx.stroke(); ctx.shadowBlur = 0;
      ctx.font = `${isImpact ? 16 : 14}px serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillStyle = isImpact ? '#c0392b' : '#1a3a5c'; ctx.fillText(n.icon, x, y);
      ctx.font = `500 10px 'IBM Plex Mono', monospace`; ctx.fillStyle = '#4a4535'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(n.label.toUpperCase(), x, y + r + 8);
    });
  };

  useEffect(() => { drawGraph(NODES.length); }, []);

  const animateGraph = () => {
    if (animating) return; setAnimating(true); let cur = 0; setStep(0);
    const iv = setInterval(() => { cur++; setStep(cur); drawGraph(cur); if (cur >= NODES.length) { clearInterval(iv); setAnimating(false); } }, 380);
  };

  return (
    <div className="section-page active" style={{ display: 'block' }}>
      <div className="section-page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
          <div>
            <div className="section-eyebrow">Threat Visualization</div>
            <h2 className="section-title">Attack Graph Engine</h2>
            <p className="section-sub">Real-time reconstruction of adversarial movement across the MITRE attack chain.</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: 1.5, textTransform: 'uppercase', color: 'var(--text3)', background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 6, padding: '4px 10px' }}>● Static Mode</span>
            <button className="btn btn-primary" onClick={animateGraph} disabled={animating}>▶ Animate Attack Chain</button>
          </div>
        </div>
      </div>
      <div className="graph-page-layout">
        <div className="graph-canvas-side">
          <canvas ref={canvasRef} id="attackCanvas" style={{ width: '100%', height: '400px', display: 'block', background: 'var(--card)', border: '1px solid var(--border)', borderRadius: '12px', boxShadow: '0 4px 24px rgba(0,0,0,0.03)' }}></canvas>
        </div>
        <div className="timeline-side">
          <div className="card-label">Event Reconstruction Timeline</div>
          <div id="attack-timeline">
            {TIMELINE_EVENTS.map((e, i) => (
              <div key={i} className="timeline-item" style={{ opacity: animating && i > step ? 0.3 : 1, transition: 'opacity .3s' }}>
                <div className="timeline-dot-wrap">
                  <div className={`timeline-dot ${e.red ? 'red' : ''}`}></div>
                  {i < TIMELINE_EVENTS.length - 1 && <div className="timeline-line"></div>}
                </div>
                <div style={{ width: '55px' }}><span className="timeline-ts" style={{ display: 'block' }}>{e.ts.split(' ')[0]}</span></div>
                <div className="timeline-content">
                  <div className="timeline-title">{e.title}</div>
                  <div className="timeline-desc">{e.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
