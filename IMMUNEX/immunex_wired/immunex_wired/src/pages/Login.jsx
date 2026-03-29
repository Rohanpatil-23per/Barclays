import React, { useState } from 'react';
import { useToast } from '../hooks/useToast';

export default function Login({ onLogin }) {
  const showToast = useToast();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [errorVisible, setErrorVisible] = useState(false);
  const [isShaking, setIsShaking] = useState(false);

  const handleLogin = (e) => {
    e.preventDefault();
    if (username === 'admin' && password === 'immunex123') {
      onLogin(); // transition to dashboard
    } else {
      setErrorVisible(true);
      setIsShaking(true);
      setTimeout(() => setIsShaking(false), 500);
    }
  };

  return (
    <div id="login-page" className="page active" style={{flexDirection:'column'}}>
      <div className="login-bg-dots"></div>

      <nav className="login-nav">
        <div className="login-nav-logo">
          <div className="logo-icon">
            <svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M9 2L15 5.5V12.5L9 16L3 12.5V5.5L9 2Z" stroke="#fff" strokeWidth="1.5" fill="none"/>
              <circle cx="9" cy="9" r="2.5" fill="#fff"/>
            </svg>
          </div>
          <span>IMMUNEX</span>
        </div>
      </nav>

      <div className="login-main">
        <div className="login-eyebrow">Threat Intelligence Platform</div>
        <h1 className="login-headline">Detect. Respond. Adapt.</h1>
        <p className="login-sub">IMMUNEX is the adaptive immune system for your infrastructure. Advanced threat intelligence, curated with editorial precision.</p>

        <form className={`login-card ${isShaking ? 'shake' : ''}`} id="login-card" onSubmit={handleLogin}>
          {errorVisible && (
            <div className="login-error show">⛔ ACCESS DENIED — INVALID CREDENTIALS</div>
          )}
          <div className="form-group">
            <label className="form-label">Username (try: admin)</label>
            <input 
              className="form-input" 
              type="text" 
              placeholder="admin" 
              value={username} 
              onChange={e => setUsername(e.target.value)} 
            />
          </div>
          <div className="form-group">
            <label className="form-label">Password (try: immunex123)</label>
            <input 
              className="form-input" 
              type="password" 
              placeholder="••••••••" 
              value={password} 
              onChange={e => setPassword(e.target.value)} 
            />
          </div>
          <button className="btn-auth" type="submit">Authenticate</button>
        </form>

        <div className="login-stats">
          <div className="login-stat">
            <span className="login-stat-num">99.999%</span>
            <span className="login-stat-label">Uptime</span>
          </div>
          <div className="login-stat">
            <span className="login-stat-num">&lt; 2ms</span>
            <span className="login-stat-label">Detection Latency</span>
          </div>
          <div className="login-stat">
            <span className="login-stat-num">0</span>
            <span className="login-stat-label">False Negatives in Production</span>
          </div>
        </div>
      </div>
    </div>
  );
}
