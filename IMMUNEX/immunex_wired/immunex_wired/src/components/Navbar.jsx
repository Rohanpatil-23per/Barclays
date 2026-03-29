import React, { useState, useEffect } from 'react';
import { useToast } from '../hooks/useToast';
import { getOrchestratorHealth, ORCHESTRATOR } from '../api/immunexApi';

export default function Navbar({ currentSection, setCurrentSection }) {
  const showToast = useToast();
  const [notifOpen, setNotifOpen]     = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [orchOnline, setOrchOnline]   = useState(null); // null = checking

  const navLinks = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'detect',    label: 'Detect' },
    { id: 'playbook',  label: 'Playbook' },
    { id: 'ais',       label: 'AIS Engine' },
  ];

  const handleNav = (id) => { setCurrentSection(id); setNotifOpen(false); };

  // Live health dot
  useEffect(() => {
    let prev = null;
    const check = async () => {
      try {
        await getOrchestratorHealth();
        if (prev !== true) { setOrchOnline(true); prev = true; }
      } catch {
        if (prev !== false) { setOrchOnline(false); prev = false; }
      }
    };
    check();
    const iv = setInterval(check, 30000);
    return () => clearInterval(iv);
  }, []);

  const dotColor = orchOnline === null ? '#b8860b' : orchOnline ? '#27ae60' : '#c0392b';
  const dotLabel = orchOnline === null ? 'Connecting…' : orchOnline ? 'Backend Online' : 'Backend Offline';

  return (
    <>
      <nav className="topnav">
        <div className="topnav-logo">
          <div className="logo-icon">
            <svg viewBox="0 0 18 18" fill="none">
              <path d="M9 2L15 5.5V12.5L9 16L3 12.5V5.5L9 2Z" stroke="#fff" strokeWidth="1.5" fill="none"/>
              <circle cx="9" cy="9" r="2.5" fill="#fff"/>
            </svg>
          </div>
          <span>IMMUNEX</span>
        </div>

        <div className="topnav-links">
          {navLinks.map(link => (
            <button key={link.id} className={`topnav-link ${currentSection === link.id ? 'active' : ''}`} onClick={() => handleNav(link.id)}>
              {link.label}
            </button>
          ))}
        </div>

        <div className="topnav-right">
          {/* Live backend status */}
          <div
            title={`${dotLabel} — ${ORCHESTRATOR}`}
            style={{ display: 'flex', alignItems: 'center', gap: 5, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', cursor: 'default', padding: '4px 10px', border: '1px solid var(--border)', borderRadius: 6, background: 'var(--bg2)' }}
          >
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: dotColor, boxShadow: orchOnline ? `0 0 5px ${dotColor}` : 'none' }}></div>
            {orchOnline === null ? 'Connecting' : orchOnline ? 'Online' : 'Offline'}
          </div>

          <div className="topnav-search" onClick={() => showToast('Search coming soon', 'info')} style={{ cursor: 'pointer' }}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>
            <input type="text" placeholder="Search telemetry..." readOnly style={{ cursor: 'pointer' }} />
          </div>
          <button className="topnav-icon" onClick={() => { setNotifOpen(!notifOpen); setSettingsOpen(false); }} style={{ position: 'relative' }}>
            🔔<span className="notif-badge" style={{ display: 'block' }}></span>
          </button>
          <button className="topnav-icon" onClick={() => { setSettingsOpen(!settingsOpen); setNotifOpen(false); }}>⚙️</button>
          <div className="topnav-avatar" onClick={() => showToast('Logged in as admin · Level 3 Clearance', 'info')} style={{ cursor: 'pointer' }}>TD</div>
        </div>
      </nav>

      {notifOpen && (
        <div className="notif-panel open">
          <div className="notif-header">
            <span className="notif-title">Alerts · 3 New</span>
            <button className="notif-close" onClick={() => setNotifOpen(false)}>✕</button>
          </div>
          <div className="notif-item" onClick={() => handleNav('attack-graph')}>
            <div className="notif-dot red"></div>
            <div><div className="notif-text">C2 beacon detected — 194.28.115.42 actively beaconing via HTTPS</div><div className="notif-ts">14:22:01 · CRITICAL</div></div>
          </div>
          <div className="notif-item" onClick={() => handleNav('playbook')}>
            <div className="notif-dot yellow"></div>
            <div><div className="notif-text">Playbook action pending — ISOLATE HOST DC-PRIMARY-01</div><div className="notif-ts">14:18:45 · URGENT</div></div>
          </div>
          <div className="notif-item" onClick={() => handleNav('ais')}>
            <div className="notif-dot green"></div>
            <div><div className="notif-text">AIS retraining complete — blindspot patched in 2.3 min</div><div className="notif-ts">13:55:00 · INFO</div></div>
          </div>
        </div>
      )}

      {settingsOpen && (
        <div className="settings-panel open">
          <div className="settings-header">
            <span className="settings-title">Settings</span>
            <button className="notif-close" onClick={() => setSettingsOpen(false)}>✕</button>
          </div>
          <div className="settings-item">
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', marginBottom: 6, letterSpacing: 1 }}>BACKEND</div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text2)', wordBreak: 'break-all' }}>{ORCHESTRATOR}</div>
          </div>
          <div className="settings-item">
            <div className="settings-row">
              <div><div className="settings-label">Live Terminal</div><div className="settings-sub">Auto-scroll terminal</div></div>
              <label className="tog"><input type="checkbox" defaultChecked onChange={() => showToast('Setting saved', 'success')} /><span className="tog-track"></span></label>
            </div>
          </div>
          <div className="settings-item">
            <div className="settings-row">
              <div><div className="settings-label">Dark Mode</div><div className="settings-sub">Toggle dark theme</div></div>
              <label className="tog"><input type="checkbox" onChange={(e) => {
                if (e.target.checked) document.documentElement.style.setProperty('--bg', '#111');
                else document.documentElement.style.setProperty('--bg', '#f5f0e8');
              }} /><span className="tog-track"></span></label>
            </div>
          </div>
          <div className="settings-item">
            <button className="btn btn-danger btn-sm" style={{ width: '100%' }} onClick={() => setCurrentSection('login')}>Log Out</button>
          </div>
        </div>
      )}
    </>
  );
}
