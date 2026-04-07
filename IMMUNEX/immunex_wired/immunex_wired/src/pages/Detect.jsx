import React, { useState, useEffect, useRef } from 'react';
import { SAMPLES } from '../data/mock';
import {
  detectAnomaly, runFullPipelineFrontend, runPipeline, runDemoInjection,
  buildAlertFromLog, LAYER1, ORCHESTRATOR,
} from '../api/immunexApi';

const STREAM_TEMPLATES = [
  (ts, ip) => `[${ts}] AUTH_SUCCESS user=svc_monitor src=${ip} dst=10.0.0.1 method=cert`,
  (ts, ip) => `[${ts}] FIREWALL_ALLOW src=${ip} dst=10.0.0.5 port=443 proto=TCP bytes=1240`,
  (ts, ip) => `[${ts}] DNS_QUERY src=${ip} query=api.bank.internal type=A ttl=300`,
  (ts, ip) => `[${ts}] HTTP_GET src=${ip} path=/health status=200 latency=4ms`,
  (ts, ip) => `[${ts}] SSH_LOGIN user=sysadmin src=${ip} dst=10.0.1.5 auth=pubkey result=SUCCESS`,
  (ts, ip) => `[${ts}] FILE_ACCESS user=app_svc path=/var/log/app.log action=READ bytes=4096`,
  (ts, ip) => `[${ts}] NET_FLOW src=${ip} dst=10.0.0.2 port=5432 proto=TCP bytes=8192 duration=120ms`,
  (ts, ip) => `{"event":"auth","user":"bob.harris","src_ip":"${ip}","method":"MFA","result":"success","ts":"${ts}"}`,
];
const ANOMALY_TEMPLATES = [
  (ts, ip) => `[${ts}] FAILED_AUTH user=root src=${ip} dst=10.0.0.1 attempts=847 proto=SSH`,
  (ts, ip) => `[${ts}] PORT_SCAN src=${ip} dst_range=10.0.0.0/24 ports=1-65535 rate=1200pps`,
  (ts, ip) => `[${ts}] OUTBOUND_CONN src=${ip} dst=194.28.115.42 dst_port=4444 interval=30s beacon=true`,
  (ts, ip) => `{"event":"anomaly","src_ip":"${ip}","attack_cat":"SQL_INJECTION","confidence":0.97,"ts":"${ts}"}`,
  (ts, ip) => `[${ts}] PROCESS_CREATE parent=winword.exe child=cmd.exe host=${ip} suspicious=true mimikatz`,
  (ts, ip) => `[${ts}] FILE_RENAME src=${ip} file=report.docx new=report.docx.locked ransomware=true`,
  (ts, ip) => `[${ts}] DNS_QUERY src=${ip} query=aGVsbG93b3JsZA==.exfil.attacker.io type=TXT`,
  (ts, ip) => `[${ts}] BROWSER_HOOK host=CUSTOMER-PC src=${ip} target=https://bank.internal/login hooked=true`,
];
const ANOMALY_IPS = ['45.12.98.221', '103.35.74.10', '91.92.248.101', '77.88.21.4', '194.28.115.42', '45.129.33.21', '185.220.101.55'];
const NORMAL_IPS  = ['192.168.1.55', '10.0.2.11', '192.168.1.22', '10.0.0.10', '172.16.0.5'];
const TS = () => new Date().toISOString().replace('T', ' ').slice(0, 19);

export default function Detect({ onPipelineResult }) {
  const [logMode, setLogMode]   = useState('paste');
  const [logInput, setLogInput] = useState('');
  const [detectStatus, setDetectStatus] = useState('idle');
  const [backendMode, setBackendMode]   = useState(true); // true = hit real backend

  const [anomalyClass, setAnomalyClass] = useState('');
  const [anomalyScore, setAnomalyScore] = useState(0);
  const [confidence, setConfidence]     = useState(0);
  const [sigma, setSigma]               = useState(0);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [backendError, setBackendError] = useState('');

  const [totalLines, setTotalLines]         = useState(0);
  const [processedLines, setProcessedLines] = useState(0);
  const [processingLog, setProcessingLog]   = useState([]);

  const [streamRunning, setStreamRunning]   = useState(false);
  const [streamLines, setStreamLines]       = useState([]);
  const [linesIngested, setLinesIngested]   = useState(0);
  const [anomaliesFound, setAnomaliesFound] = useState(0);
  const [streamRate, setStreamRate]         = useState(0);
  const streamContainerRef = useRef(null);

  const fileInputRef = useRef(null);
  const [fileDetails, setFileDetails] = useState('');

  useEffect(() => {
    if (detectStatus === 'anomaly') {
      const canvas = document.getElementById('detectCanvas');
      if (!canvas) return;
      canvas.width = canvas.offsetWidth || 400;
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      const pts = [];
      for (let x = 0; x < W; x++) pts.push({ x, y: H / 2 + (Math.random() - 0.5) * 20 });
      pts[W - 40] = { x: W - 40, y: H * 0.1 };
      pts[W - 35] = { x: W - 35, y: H * 0.05 };
      pts[W - 30] = { x: W - 30, y: H * 0.15 };
      ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
      pts.forEach(p => ctx.lineTo(p.x, p.y));
      ctx.strokeStyle = '#e74c3c'; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.beginPath(); ctx.setLineDash([4, 4]);
      ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2);
      ctx.strokeStyle = '#c8bfaf'; ctx.lineWidth = 1; ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [detectStatus]);

  useEffect(() => {
    let iv;
    if (streamRunning) {
      const rateStart = Date.now();
      let rateLines = 0;
      iv = setInterval(() => {
        const numLines = Math.floor(Math.random() * 3) + 1;
        setStreamLines(prev => {
          let next = [...prev];
          for (let i = 0; i < numLines; i++) {
            const isAnomaly = Math.random() < 0.07;
            const ip   = isAnomaly ? ANOMALY_IPS[Math.floor(Math.random() * ANOMALY_IPS.length)] : NORMAL_IPS[Math.floor(Math.random() * NORMAL_IPS.length)];
            const ts   = TS();
            const tmpl = isAnomaly ? ANOMALY_TEMPLATES[Math.floor(Math.random() * ANOMALY_TEMPLATES.length)] : STREAM_TEMPLATES[Math.floor(Math.random() * STREAM_TEMPLATES.length)];
            const text = tmpl(ts, ip);
            next.push({ id: Date.now() + Math.random(), text, anomaly: isAnomaly });
            rateLines++;
            if (isAnomaly) {
              setAnomaliesFound(c => c + 1);
              if (backendMode) {
                setTimeout(() => triggerDetectionBackend(text, ip, true), 200);
              } else {
                setTimeout(() => triggerDetectionLocal(text), 200);
              }
            }
          }
          if (next.length > 80) next = next.slice(next.length - 80);
          return next;
        });
        setLinesIngested(p => p + numLines);
        const elapsed = (Date.now() - rateStart) / 1000;
        setStreamRate((rateLines / elapsed).toFixed(1));
        if (streamContainerRef.current) streamContainerRef.current.scrollTop = streamContainerRef.current.scrollHeight;
      }, 700);
    } else {
      setStreamRate(0);
    }
    return () => clearInterval(iv);
  }, [streamRunning, backendMode]);

  // ── Backend detection ───────────────────────────────────────────────────────
  const triggerDetectionBackend = async (overrideText, srcIp, fromStream = false) => {
    const rawText = overrideText !== undefined ? overrideText : logInput;
    if (!rawText.trim()) return;

    const lines = rawText.split('\n').filter(l => l.trim());
    setTotalLines(lines.length);
    setProcessedLines(0);
    setProcessingLog([]);
    setDetectStatus('parsing');
    setBackendError('');
    setPipelineResult(null);

    // Show line-by-line parse animation while real request runs
    let idx = 0;
    const animInterval = setInterval(() => {
      if (idx < lines.length) {
        const line = lines[idx];
        const isAnomLine = /alert|anomaly|alrt|failed_auth|attack|malicious|exploit|brute|union|sql|c2|beacon|ransomware|mimikatz|lsass/i.test(line);
        setProcessingLog(prev => {
          const next = [...prev, { text: line.slice(0, 90) + (line.length > 90 ? '…' : ''), flag: isAnomLine }];
          return next.slice(-12);
        });
        setProcessedLines(idx + 1);
        idx++;
      } else {
        clearInterval(animInterval);
      }
    }, Math.max(40, Math.min(120, 800 / lines.length)));

    try {
      const logLines = rawText.split('\n').filter(l => l.trim());
      const anomalyLines = logLines.filter(l =>
        /alert|anomaly|alrt|failed_auth|attack|malicious|exploit|brute|union|sql|c2|beacon|ransomware|mimikatz|lsass|scan|flood/i.test(l)
      );
      // Use anomalous lines if any, otherwise use full text
      const targetLines = anomalyLines.length > 0 ? anomalyLines : logLines;
      const representativeLine = targetLines[0] || rawText;
      const alert = buildAlertFromLog(overrideText || rawText, srcIp || '0.0.0.0');
      let result;
      if (fromStream) {
        // Stream mode: call real L1 detect directly for varied results
        try {
          result = await runFullPipelineFrontend(alert);
        } catch {
          result = await runDemoInjection({ logs: representativeLine });
        }
      } else {
        try {
          result = await runDemoInjection({ logs: representativeLine });
        } catch {
          try { result = await runPipeline(alert); }
          catch { result = await runFullPipelineFrontend(alert); }
        }
      }
      clearInterval(animInterval);
      setProcessedLines(lines.length);
      setPipelineResult(result);
      if (onPipelineResult) onPipelineResult(result);
      // Show per-line summary in processing log
      const totalAnom = anomalyLines.length;
      const totalNorm = logLines.length - totalAnom;
      setProcessingLog(prev => [
        ...prev,
        { text: `── ${logLines.length} lines processed: ${totalAnom} anomalous, ${totalNorm} normal`, flag: totalAnom > 0 },
      ]);

      const score = result.anomaly_score ?? result.layer1?.anomaly_score ?? 0;
      const isAnomaly = result.verdict === 'ANOMALOUS' || result.layer1?.is_anomalous || score > 0.5;
      const l1 = result.layer1 || {};

      if (isAnomaly) {
        setDetectStatus('anomaly');
        setAnomalyClass(l1.attack_type || result.layer2?.attack_type || 'Unknown Anomaly');
        setAnomalyScore(parseFloat((score * 100).toFixed(1)));
        const conf = l1.confidence ?? l1.cicids_score ?? score;
        setConfidence(parseFloat((conf * 100).toFixed(1)));
        // sigma: distance based on anomaly score — higher score = further from baseline
        const sigmaVal = (2.0 + score * 5.0).toFixed(2);
        setSigma(sigmaVal);
      } else {
        setDetectStatus('clean');
        setAnomalyScore(0);
      }
    } catch (e) {
      clearInterval(animInterval);
      setBackendError(e.message);
      // Fall back to local heuristic
      triggerDetectionLocal(rawText);
    }
  };

  // ── Local heuristic detection (offline fallback) ───────────────────────────
  const triggerDetectionLocal = (overrideText) => {
    const rawText = overrideText !== undefined ? overrideText : logInput;
    if (!rawText.trim()) return;
    const lines = rawText.split('\n').filter(l => l.trim());
    setTotalLines(lines.length);
    setProcessedLines(0);
    setProcessingLog([]);
    setDetectStatus('parsing');

    let idx = 0;
    const interval = setInterval(() => {
      if (idx < lines.length) {
        const line = lines[idx];
        const isAnomLine = /alert|anomaly|alrt|failed_auth|attack|malicious|exploit|brute|union|sql|c2|beacon|ransomware|mimikatz|lsass/i.test(line);
        setProcessingLog(prev => {
          const next = [...prev, { text: line.slice(0, 90) + (line.length > 90 ? '…' : ''), flag: isAnomLine }];
          return next.slice(-12);
        });
        setProcessedLines(idx + 1);
        idx++;
      } else {
        clearInterval(interval);
        const txt = rawText.toLowerCase();
        const isSQL    = txt.includes('sql') || txt.includes('union') || txt.includes('injection');
        const isBrute  = txt.includes('brute') || txt.includes('failed_auth') || txt.includes('attempts=');
        const isC2     = txt.includes('beacon') || txt.includes('c2') || txt.includes('194.28');
        const isAnomaly = isSQL || isBrute || isC2 || txt.includes('anomaly') || txt.includes('alert') || txt.includes('alrt') || txt.includes('mimikatz') || txt.includes('ransomware');
        if (isAnomaly) {
          setDetectStatus('anomaly');
          const classes = ['Lateral Movement', 'Credential Stuffing', 'SQL Injection', 'Privilege Escalation', 'C2 Beacon', 'Ransomware', 'Port Scan'];
          setAnomalyClass(isSQL ? 'SQL Injection' : isBrute ? 'Credential Stuffing' : isC2 ? 'C2 Beacon' : classes[Math.floor(Math.random() * classes.length)]);
          setAnomalyScore(78 + Math.random() * 20);
          setConfidence(85 + Math.random() * 12);
          setSigma((3.5 + Math.random() * 2.5).toFixed(2));
        } else {
          setDetectStatus('clean');
        }
      }
    }, Math.max(40, Math.min(120, 800 / lines.length)));
  };

  const triggerDetection = (overrideText) => {
    if (backendMode) triggerDetectionBackend(overrideText);
    else triggerDetectionLocal(overrideText);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFileDetails(`${file.name} · ${(file.size / 1024).toFixed(1)} KB`);
      const reader = new FileReader();
      reader.onload = (ev) => setLogInput(ev.target.result);
      reader.readAsText(file);
    }
  };

  const clearAll = () => {
    setLogInput(''); setFileDetails(''); setDetectStatus('idle');
    setProcessingLog([]); setTotalLines(0); setProcessedLines(0);
    setPipelineResult(null); setBackendError('');
  };

  const progressPct = totalLines > 0 ? Math.round((processedLines / totalLines) * 100) : 0;

  const activeTabStyle = { flex: 1, padding: '8px 0', fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: 1.5, textTransform: 'uppercase', fontWeight: 600, border: 'none', cursor: 'pointer', background: 'var(--accentbg)', color: '#fff', transition: 'all .2s' };
  const idleTabStyle   = { ...activeTabStyle, background: 'var(--bg2)', color: 'var(--text2)' };

  return (
    <div className="section-page active" style={{ display: 'block' }}>
      <div className="section-page-header" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 40, alignItems: 'center' }}>
        <div>
          <div className="section-eyebrow">Anomaly Detection Engine</div>
          <h2 className="section-title">Inject logs. Get answers instantly.</h2>
          <p className="section-sub">Feed raw telemetry into the neural archive. IMMUNEX matches patterns across all 5 layers to isolate deviant behaviour in milliseconds.</p>
          {/* Backend toggle */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 14, fontFamily: 'var(--mono)', fontSize: 11 }}>
            <button
              onClick={() => setBackendMode(b => !b)}
              style={{
                padding: '5px 12px', borderRadius: 6, border: '1px solid var(--border)',
                background: backendMode ? 'rgba(61,90,62,0.15)' : 'var(--bg2)',
                color: backendMode ? 'var(--accentbg)' : 'var(--text3)',
                fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: 1, cursor: 'pointer',
                fontWeight: 600,
              }}
            >
              {backendMode ? '● LIVE BACKEND' : '○ LOCAL MODE'}
            </button>
            {backendMode && (
              <span style={{ color: 'var(--text3)', fontSize: 10 }}>
                → {LAYER1}/analyze
              </span>
            )}
          </div>
        </div>
        <div className="live-capture-card" style={{ margin: 0 }}>
          <div className="capture-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="card-label">Live Capture</span>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 9, fontWeight: 700, letterSpacing: 1.5, textTransform: 'uppercase', padding: '3px 9px', borderRadius: 5, background: 'rgba(231,76,60,0.12)', color: '#e74c3c', border: '1px solid rgba(231,76,60,0.35)', display: 'flex', alignItems: 'center', gap: 5, cursor: 'default', userSelect: 'none', whiteSpace: 'nowrap' }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#e74c3c', display: 'inline-block', boxShadow: '0 0 5px #e74c3c' }}></span>
              ANOMALY<span style={{ fontSize: 8, opacity: 0.55, fontWeight: 400, marginLeft: 2 }}>[DEMO]</span>
            </span>
          </div>
          <div className="capture-request">POST /api/v1/auth/token HTTP/1.1</div>
          <div className="capture-metrics">
            <div><div className="capture-metric-label">Threat Score</div><div className="capture-metric-value">98.4<span style={{ fontSize: 13, opacity: .6 }}>/100</span></div></div>
            <div><div className="capture-metric-label">Euclidean Distance</div><div className="capture-metric-value">14.22</div></div>
            <div><div className="capture-metric-label">Classification</div><div className="capture-metric-sub">Credential Stuffing</div></div>
            <div><div className="capture-metric-label">Confidence</div><div className="capture-metric-sub">High (92%)</div></div>
          </div>
        </div>
      </div>

      <div className="detect-layout">
        {/* ── Left: Input Panel ── */}
        <div className="detect-left">
          <div className="card-label" style={{ marginBottom: 12 }}>Log Input Panel</div>

          <div style={{ display: 'flex', gap: 0, marginBottom: 14, border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden' }}>
            <button style={logMode === 'paste'  ? activeTabStyle : idleTabStyle} onClick={() => setLogMode('paste')}>PASTE</button>
            <button style={logMode === 'upload' ? activeTabStyle : idleTabStyle} onClick={() => setLogMode('upload')}>UPLOAD FILE</button>
            <button style={logMode === 'stream' ? activeTabStyle : idleTabStyle} onClick={() => setLogMode('stream')}>LIVE STREAM</button>
          </div>

          {logMode === 'paste' && (
            <div>
              <div className="log-sample-btns">
                <button className="log-sample-btn" onClick={() => setLogInput(SAMPLES.brute)}>→ Sample: Brute Force</button>
                <button className="log-sample-btn" onClick={() => setLogInput(SAMPLES.sqli)}>→ Sample: SQL Injection</button>
              </div>
              <textarea className="log-textarea" placeholder={'Paste log data here (JSON, Syslog, or CSV)...\n\nExample:\n[2026-03-28 14:22:01] FAILED_AUTH user=root src=45.12.98.221\n{"event":"auth_failure","src_ip":"45.12.98.221","count":847}'} value={logInput} onChange={e => setLogInput(e.target.value)}></textarea>
              {logInput && (
                <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', marginBottom: 8, letterSpacing: 1 }}>
                  {logInput.split('\n').filter(l => l.trim()).length} lines ready to inject
                </div>
              )}
              <div className="detect-btns">
                <button className="btn btn-primary" onClick={() => triggerDetection()}>Inject &amp; Analyze</button>
                <button className="btn btn-secondary" onClick={clearAll}>Clear</button>
              </div>
            </div>
          )}

          {logMode === 'upload' && (
            <div>
              <div style={{ border: '2px dashed var(--border)', borderRadius: 10, padding: '36px 20px', textAlign: 'center', cursor: 'pointer', marginBottom: 14 }} onClick={() => fileInputRef.current?.click()}>
                <div style={{ fontSize: 28, marginBottom: 10 }}>📄</div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 600, color: 'var(--text2)', marginBottom: 6 }}>DROP LOG FILE HERE</div>
                <div style={{ fontSize: 12, color: 'var(--text3)' }}>or click to browse · .log .txt .json .csv supported</div>
              </div>
              <input type="file" ref={fileInputRef} accept=".log,.txt,.json,.csv" style={{ display: 'none' }} onChange={handleFileUpload} />
              {fileDetails && (
                <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 14px', marginBottom: 10 }}>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 600, color: 'var(--text)', marginBottom: 2 }}>📄 {fileDetails}</div>
                  {logInput && <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)' }}>{logInput.split('\n').filter(l => l.trim()).length} lines detected</div>}
                </div>
              )}
              <textarea className="log-textarea" placeholder="File contents will appear here..." value={logInput} readOnly style={{ height: 140 }}></textarea>
              <div className="detect-btns">
                <button className="btn btn-primary" onClick={() => triggerDetection()}>Inject &amp; Analyze</button>
                <button className="btn btn-secondary" onClick={clearAll}>Clear</button>
              </div>
            </div>
          )}

          {logMode === 'stream' && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
                <div>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: 1.5, textTransform: 'uppercase', color: 'var(--text3)', marginBottom: 4 }}>Stream Source</div>
                  <select style={{ padding: '8px 12px', border: '1px solid var(--border)', borderRadius: 7, background: 'var(--bg2)', color: 'var(--text)', fontSize: 13, fontFamily: 'var(--sans)', outline: 'none' }}>
                    <option value="simulation">▶ Simulate Stream (Demo)</option>
                    <option value="syslog">Syslog (UDP 514)</option>
                  </select>
                </div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <div style={{ width: 10, height: 10, borderRadius: '50%', background: streamRunning ? '#27ae60' : 'var(--border)', boxShadow: streamRunning ? '0 0 6px #27ae60' : 'none' }}></div>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text3)' }}>{streamRunning ? 'STREAMING' : 'IDLE'}</span>
                </div>
              </div>

              <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 7, padding: '8px 12px', marginBottom: 10, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', lineHeight: 1.7 }}>
                <div style={{ fontWeight: 600, color: 'var(--text2)', marginBottom: 3, letterSpacing: 1, textTransform: 'uppercase' }}>Log Formats in Stream</div>
                <div><span style={{ color: '#a8ff78' }}>●</span> Syslog: <span style={{ color: 'var(--text)' }}>[TIMESTAMP] EVENT user=x src=ip</span></div>
                <div><span style={{ color: '#a8ff78' }}>●</span> JSON:   <span style={{ color: 'var(--text)' }}>{`{"event":"type","src_ip":"x"}`}</span></div>
                <div><span style={{ color: '#ff6b6b' }}>▲</span> Anomaly lines highlighted in <span style={{ color: '#ff6b6b' }}>red</span></div>
              </div>

              <div ref={streamContainerRef} style={{ background: '#1a1a1a', borderRadius: 10, padding: 14, height: 240, overflowY: 'auto', fontFamily: 'var(--mono)', fontSize: 10.5, lineHeight: 1.75, color: '#a8ff78', border: '1px solid var(--border)' }}>
                {!streamRunning && streamLines.length === 0 && (
                  <div style={{ color: '#555' }}>
                    <div>// IMMUNEX Live Log Stream — waiting for connection...</div>
                    <div style={{ marginTop: 6 }}>// Press ▶ Start Stream to begin simulation</div>
                  </div>
                )}
                {streamLines.map(line => (
                  <div key={line.id} style={{ color: line.anomaly ? '#ff6b6b' : '#a8ff78', borderLeft: line.anomaly ? '3px solid #e74c3c' : '3px solid transparent', paddingLeft: 8, background: line.anomaly ? 'rgba(231,76,60,0.06)' : 'transparent' }}>
                    {line.anomaly && <span style={{ color: '#e74c3c', marginRight: 6, fontWeight: 700 }}>[ALRT]</span>}
                    {line.text}
                  </div>
                ))}
              </div>

              <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 8 }}>
                {[
                  { label: 'Lines Ingested', value: linesIngested, color: 'var(--accentbg)' },
                  { label: 'Anomalies',      value: anomaliesFound, color: 'var(--red)' },
                  { label: 'Rate',           value: `${streamRate}/s`, color: 'var(--text2)' },
                  { label: 'Normal',         value: linesIngested - anomaliesFound, color: '#27ae60' },
                ].map(({ label, value, color }) => (
                  <div key={label} style={{ background: 'var(--bg2)', borderRadius: 7, padding: '10px 12px', border: '1px solid var(--border)' }}>
                    <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>{label}</div>
                    <div style={{ fontFamily: 'var(--mono)', fontSize: 16, fontWeight: 700, color }}>{value}</div>
                  </div>
                ))}
              </div>

              <div className="detect-btns" style={{ marginTop: 12 }}>
                <button className="btn btn-primary" onClick={() => setStreamRunning(!streamRunning)}>
                  {streamRunning ? '⏹ Stop Stream' : '▶ Start Stream'}
                </button>
                <button className="btn btn-secondary" onClick={() => { setStreamRunning(false); setStreamLines([]); setLinesIngested(0); setAnomaliesFound(0); setStreamRate(0); }}>Clear</button>
              </div>
            </div>
          )}
        </div>

        {/* ── Right: Results Panel ── */}
        <div className="detect-right">

          {backendError && (
            <div style={{ background: 'rgba(192,57,43,0.08)', border: '1px solid rgba(192,57,43,0.3)', borderRadius: 8, padding: '10px 14px', marginBottom: 12, fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--red)' }}>
              ⚠ Backend error (showing local result): {backendError}
            </div>
          )}

          {detectStatus === 'parsing' && totalLines > 0 && (
            <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 10, padding: '14px 16px', marginBottom: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1.5, color: 'var(--text3)' }}>Processing Logs</span>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 700, color: 'var(--accentbg)' }}>{processedLines}/{totalLines} lines</span>
              </div>
              <div style={{ height: 6, background: 'var(--border)', borderRadius: 4, marginBottom: 10, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${progressPct}%`, background: 'var(--accentbg)', borderRadius: 4, transition: 'width .1s linear' }}></div>
              </div>
              <div style={{ background: '#1a1a1a', borderRadius: 7, padding: '10px 12px', maxHeight: 140, overflowY: 'auto', fontFamily: 'var(--mono)', fontSize: 10, lineHeight: 1.65 }}>
                {processingLog.map((l, i) => (
                  <div key={i} style={{ color: l.flag ? '#ff6b6b' : '#a8ff78', paddingLeft: 4, borderLeft: l.flag ? '2px solid #e74c3c' : '2px solid transparent' }}>
                    {l.flag && '▲ '}{l.text}
                  </div>
                ))}
              </div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text3)', marginTop: 6, textAlign: 'right' }}>{progressPct}% complete</div>
            </div>
          )}

          <div className="detection-result-card">
            <div className="card-label">Detection Result</div>
            {detectStatus === 'idle'    && <div style={{ color: 'var(--text3)', fontSize: 13 }}>Waiting for log input...</div>}
            {detectStatus === 'parsing' && totalLines === 0 && <div style={{ color: 'var(--text3)', fontSize: 13 }}>Analysing stream...</div>}
            {detectStatus === 'clean'   && <div style={{ color: 'var(--green)', fontWeight: 600, fontFamily: 'var(--mono)', fontSize: 13 }}>✓ NORMAL TRAFFIC — NO ANOMALY DETECTED</div>}
            {detectStatus === 'anomaly' && (
              <div className="anomaly-banner">
                <span className="anomaly-icon">⚠</span>
                <span className="anomaly-text">ANOMALY DETECTED</span>
                <span className="anomaly-class">{anomalyClass}</span>
              </div>
            )}
          </div>

          <div className="detection-result-card">
            <div className="card-label">Scores</div>
            <div className="progress-row">
              <div className="progress-header">
                <span className="progress-label">Anomaly Score</span>
                <span className="progress-val">{detectStatus === 'anomaly' ? anomalyScore.toFixed(1) + '%' : '—'}</span>
              </div>
              <div className="progress-bar"><div className="progress-fill" style={{ width: detectStatus === 'anomaly' ? `${anomalyScore}%` : '0%' }}></div></div>
            </div>
            <div className="progress-row">
              <div className="progress-header">
                <span className="progress-label">Model Confidence</span>
                <span className="progress-val">{detectStatus === 'anomaly' ? confidence.toFixed(1) + '%' : '—'}</span>
              </div>
              <div className="progress-bar"><div className="progress-fill blue" style={{ width: detectStatus === 'anomaly' ? `${confidence}%` : '0%', background: '#2980b9' }}></div></div>
            </div>
          </div>

          {/* Backend pipeline result if available */}
          {pipelineResult && (
            <div className="detection-result-card" style={{ background: 'rgba(61,90,62,0.04)' }}>
              <div className="card-label" style={{ marginBottom: 10 }}>Pipeline Result (Live Backend)</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6, marginBottom: 10 }}>
                {[1,2,3,4,5].map(n => {
                  const layer = pipelineResult[`layer${n}`];
                  const ok    = layer && !layer.error;
                  return (
                    <div key={n} style={{ background: ok ? 'rgba(61,90,62,0.1)' : 'var(--bg2)', border: `1px solid ${ok ? 'rgba(61,90,62,0.4)' : 'var(--border)'}`, borderRadius: 6, padding: '8px 6px', textAlign: 'center' }}>
                      <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text3)', marginBottom: 2 }}>L{n}</div>
                      <div style={{ fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 700, color: ok ? 'var(--accentbg)' : 'var(--text3)' }}>{ok ? '✓' : layer?.error ? '✗' : '—'}</div>
                    </div>
                  );
                })}
              </div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', lineHeight: 1.6 }}>
                {pipelineResult.layer2?.attack_type && <div>L2 match: <span style={{ color: 'var(--text)' }}>{pipelineResult.layer2.attack_type}</span></div>}
                {pipelineResult.layer3?.decision?.action_name && <div>L3 action: <span style={{ color: 'var(--text)' }}>{pipelineResult.layer3.decision.action_name}</span></div>}
                {pipelineResult.layer5?.playbook && <div>Playbook: <span style={{ color: 'var(--text)' }}>{String(pipelineResult.layer5.playbook).slice(0, 80)}</span></div>}
              </div>
            </div>
          )}

          <div className="detection-result-card">
            <div className="card-label">Distance from Baseline</div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div className="sigma-display">σ {detectStatus === 'anomaly' ? sigma : '—'}</div>
              {detectStatus === 'anomaly' && <span className="anomaly-badge-sm">OUTLIER DETECTED</span>}
            </div>
            <canvas id="detectCanvas" height="80" style={{ marginTop: 12, width: '100%' }}></canvas>
          </div>

          {(detectStatus === 'anomaly' || detectStatus === 'clean') && totalLines > 0 && (
            <div className="detection-result-card" style={{ background: 'var(--bg2)' }}>
              <div className="card-label">Ingestion Summary</div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 11, lineHeight: 2, color: 'var(--text2)' }}>
                <div>Lines processed: <span style={{ color: 'var(--text)', fontWeight: 600 }}>{totalLines}</span></div>
                <div>Anomalous lines: <span style={{ color: detectStatus === 'anomaly' ? '#e74c3c' : '#27ae60', fontWeight: 600 }}>{detectStatus === 'anomaly' ? processingLog.filter(l => l.flag).length : 0}</span></div>
                <div>Normal lines: <span style={{ color: '#27ae60', fontWeight: 600 }}>{totalLines - processingLog.filter(l => l.flag).length}</span></div>
                <div>Mode: <span style={{ color: backendMode ? 'var(--accentbg)' : 'var(--text3)', fontWeight: 600 }}>{backendMode ? 'Live Backend' : 'Local Heuristic'}</span></div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
