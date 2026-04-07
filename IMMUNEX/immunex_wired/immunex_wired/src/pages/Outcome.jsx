import React, { useState, useEffect, useRef } from 'react';

export default function Outcome() {
  const [outcomeIsGood, setOutcomeIsGood] = useState(true);
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = canvas.offsetWidth || 500;
    const H = canvas.offsetHeight || 100;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,W,H);

    // If bad outcome, simulate lower peak
    const currentPeak = outcomeIsGood ? 0.947 : 0.412;
    const vals = [0.3, 0.45, 0.5, 0.62, 0.75, 0.85, currentPeak];
    const barW = W / vals.length * 0.6;
    const gap = W / vals.length;

    vals.forEach((v,i) => {
      const bh = v * (H - 20);
      const x = i * gap + gap*0.2;
      const y = H - bh;
      const alpha = 0.4 + v*0.5;
      ctx.fillStyle = outcomeIsGood ? `rgba(61,90,62,${alpha})` : `rgba(192,57,43,${alpha})`;
      ctx.beginPath();
      ctx.roundRect(x, y, barW, bh, [4,4,0,0]);
      ctx.fill();
    });

    ctx.font = '10px IBM Plex Mono';
    ctx.fillStyle = '#4a4535';
    ctx.textAlign = 'right';
    ctx.fillText(`Peak ${(currentPeak*100).toFixed(1)}%`, W-4, 12);
  }, [outcomeIsGood]);

  const toggleOutcome = () => setOutcomeIsGood(!outcomeIsGood);
  const simulateOutcome = () => setOutcomeIsGood(true);

  return (
    <div className="section-page active" style={{ display: 'block' }}>
      <div className="section-page-header">
        <div className="section-eyebrow">Incident Outcome</div>
        <h2 className="section-title">Threat stopped. System restored.</h2>
      </div>
      
      <div className="outcome-layout">
        <div className="outcome-status-card">
          <div className="outcome-status-label">Current State</div>
          <div className="outcome-state-toggle">
            <button className="outcome-toggle-btn" onClick={toggleOutcome}>● Toggle Status</button>
          </div>
          
          <div className={`outcome-big-text ${!outcomeIsGood ? 'escalating' : ''}`}>
            {outcomeIsGood ? 'THREAT STOPPED' : 'ESCALATING'}
          </div>
          <p className="outcome-desc">
            {outcomeIsGood 
              ? 'Vector neutralized at primary node. Data integrity verified across all sectors.' 
              : 'Attacker has pivoted. Lateral movement ongoing. Immediate action required.'}
          </p>
          <div className="outcome-score-num">{outcomeIsGood ? '94.7%' : '41.2%'}</div>
          <div className="outcome-score-label">Effectiveness Score</div>
          
          <button className="btn btn-primary" onClick={simulateOutcome}>⟳ Simulate Outcome</button>
        </div>
        
        <div className="outcome-right">
          <div className="chart-card">
            <div className="chart-card-label">Effectiveness Trend</div>
            <div className="chart-card-legend">
              <div className="chart-legend-dot" style={{background: outcomeIsGood ? 'var(--accentbg)' : 'var(--red)'}}></div>
              <span>Neural Response</span>
              <span className="chart-peak-label" style={{marginLeft:'auto'}}>
                Peak {outcomeIsGood ? '94.7%' : '41.2%'}
              </span>
            </div>
            <canvas ref={canvasRef} id="outcomeCanvas" height="100" style={{width:'100%',borderRadius:6}}></canvas>
          </div>
          
          <div className="incident-timeline-card">
            <div className="inc-timeline-header">
              <span className="inc-timeline-label">Incident Timeline</span>
              <span className="inc-timeline-count">6 Events Logged</span>
            </div>
            <div className="inc-timeline-item">
              <span className="inc-timeline-ts">14:22:01</span>
              <div className="inc-dot green"></div>
              <div><div className="inc-title">Final Purge Complete</div><div className="inc-desc">System-wide cleanup successful. All temporary artifacts removed.</div></div>
            </div>
            <div className="inc-timeline-item">
              <span className="inc-timeline-ts">14:18:45</span>
              <div className="inc-dot green"></div>
              <div><div className="inc-title">Redundancy Resynced</div><div className="inc-desc">Primary and secondary nodes are now in full parity.</div></div>
            </div>
            <div className="inc-timeline-item">
              <span className="inc-timeline-ts">14:05:32</span>
              <div className="inc-dot blue"></div>
              <div><div className="inc-title">Node 04 Quarantined</div><div className="inc-desc">Lateral movement prevented via automated network segmentation.</div></div>
            </div>
            <div className="inc-timeline-item">
              <span className="inc-timeline-ts">13:58:53</span>
              <div className="inc-dot" style={{background:'var(--red2)'}}></div>
              <div><div className="inc-title">Anomalous Payload Detected</div><div className="inc-desc">Encrypted traffic pattern identified as potential exfiltration attempt.</div></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
