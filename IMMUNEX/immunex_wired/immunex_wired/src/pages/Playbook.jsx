import React, { useState, useEffect, useRef } from 'react';
import { getPendingApprovals, approveAction, rejectAction } from '../api/immunexApi';
import { runDemoInjection } from '../api/immunexApi';

// ── All 50 override actions — sourced from action_registry.py ────────────────
const OVERRIDE_OPTIONS = [
  // Monitoring (0–9)
  { index: 0,  name: 'do_nothing',                    category: 'monitoring' },
  { index: 1,  name: 'increase_log_verbosity',        category: 'monitoring' },
  { index: 2,  name: 'trigger_soc_alert',             category: 'monitoring' },
  { index: 3,  name: 'snapshot_memory',               category: 'monitoring' },
  { index: 4,  name: 'capture_network_traffic',       category: 'monitoring' },
  { index: 5,  name: 'enable_deep_packet_inspection', category: 'monitoring' },
  { index: 6,  name: 'flag_for_human_review',         category: 'monitoring' },
  { index: 7,  name: 'escalate_to_tier2',             category: 'monitoring' },
  { index: 8,  name: 'notify_incident_manager',       category: 'monitoring' },
  { index: 9,  name: 'generate_soc_report',           category: 'monitoring' },
  // Network (10–19)
  { index: 10, name: 'block_source_ip',               category: 'network' },
  { index: 11, name: 'block_destination_ip',          category: 'network' },
  { index: 12, name: 'isolate_endpoint',              category: 'network' },
  { index: 13, name: 'quarantine_subnet',             category: 'network' },
  { index: 14, name: 'disable_external_routing',      category: 'network' },
  { index: 15, name: 'block_suspicious_port',         category: 'network' },
  { index: 16, name: 'rate_limit_source',             category: 'network' },
  { index: 17, name: 'drop_malicious_packets',        category: 'network' },
  { index: 18, name: 'null_route_attacker',           category: 'network' },
  { index: 19, name: 'block_c2_domain',               category: 'network' },
  // Credential & access (20–29)
  { index: 20, name: 'revoke_user_session',           category: 'credential' },
  { index: 21, name: 'force_mfa_reauthentication',    category: 'credential' },
  { index: 22, name: 'disable_compromised_account',   category: 'credential' },
  { index: 23, name: 'reset_service_account_password',category: 'credential' },
  { index: 24, name: 'revoke_api_key',                category: 'credential' },
  { index: 25, name: 'restrict_admin_privileges',     category: 'credential' },
  { index: 26, name: 'enforce_least_privilege',       category: 'credential' },
  { index: 27, name: 'disable_lateral_movement_path', category: 'credential' },
  { index: 28, name: 'lock_privileged_account',       category: 'credential' },
  { index: 29, name: 'audit_active_sessions',         category: 'credential' },
  // Process & endpoint (30–39)
  { index: 30, name: 'kill_malicious_process',        category: 'process' },
  { index: 31, name: 'suspend_suspicious_process',    category: 'process' },
  { index: 32, name: 'quarantine_malicious_file',     category: 'process' },
  { index: 33, name: 'rollback_filesystem_changes',   category: 'process' },
  { index: 34, name: 'restore_from_clean_snapshot',   category: 'process' },
  { index: 35, name: 'run_edr_deep_scan',             category: 'process' },
  { index: 36, name: 'patch_vulnerable_service',      category: 'process' },
  { index: 37, name: 'disable_autorun_mechanisms',    category: 'process' },
  { index: 38, name: 'terminate_remote_desktop_session', category: 'process' },
  { index: 39, name: 'sandbox_suspicious_binary',     category: 'process' },
  // Data protection (40–49)
  { index: 40, name: 'encrypt_sensitive_data_at_rest',category: 'data_protection' },
  { index: 41, name: 'revoke_data_export_permissions',category: 'data_protection' },
  { index: 42, name: 'block_usb_exfiltration',        category: 'data_protection' },
  { index: 43, name: 'disable_email_forwarding_rules',category: 'data_protection' },
  { index: 44, name: 'watermark_sensitive_documents', category: 'data_protection' },
  { index: 45, name: 'enable_dlp_policy',             category: 'data_protection' },
  { index: 46, name: 'freeze_database_writes',        category: 'data_protection' },
  { index: 47, name: 'backup_critical_data',          category: 'data_protection' },
  { index: 48, name: 'rotate_encryption_keys',        category: 'data_protection' },
  { index: 49, name: 'activate_honeypot',             category: 'data_protection' },
];

const CATEGORY_COLORS = {
  monitoring:      { bg: 'rgba(52,152,219,0.10)', border: 'rgba(52,152,219,0.35)', text: '#3498db' },
  network:         { bg: 'rgba(231,76,60,0.10)',  border: 'rgba(231,76,60,0.35)',  text: '#e74c3c' },
  credential:      { bg: 'rgba(230,126,34,0.10)', border: 'rgba(230,126,34,0.35)', text: '#e67e22' },
  process:         { bg: 'rgba(155,89,182,0.10)', border: 'rgba(155,89,182,0.35)', text: '#9b59b6' },
  data_protection: { bg: 'rgba(39,174,96,0.10)',  border: 'rgba(39,174,96,0.35)',  text: '#27ae60' },
};

// ── Seed queue — backend will push anomalies here ────────────────────────────
const INITIAL_QUEUE = [
  {
    id: 'AX-2091', priority: 1, sev: 'critical', icon: '🖥',
    title: 'ISOLATE HOST',
    target: 'ENDPOINT: DC-PRIMARY-01',
    mitre: 'T1003.001 — OS Credential Dumping',
    desc: 'Unauthorized LSASS memory dump attempt detected. Immediate isolation prevents credential harvesting and lateral spread.',
    action: 'Block IP on Perimeter Firewall',
    ts: '14:22:01',
  },
  {
    id: 'AX-2089', priority: 2, sev: 'critical', icon: '🚫',
    title: 'BLOCK C2',
    target: 'IP: 194.28.115.42',
    mitre: 'T1071 — Application Layer Protocol',
    desc: 'Outbound HTTPS beaconing detected on port 443. Fixed 30s heartbeat interval matches known C2 profile. Sever at perimeter.',
    action: 'Sinkhole C2 Domain',
    ts: '13:40:02',
  },
  {
    id: 'AX-2087', priority: 3, sev: 'critical', icon: '☁',
    title: 'BLOCK DNS EXFIL',
    target: 'DOMAIN: exfil.attacker.io',
    mitre: 'T1048.003 — DNS Exfiltration',
    desc: '4.2GB staged for exfiltration via DNS tunneling. Base64-encoded subdomains detected. Block domain and isolate source host.',
    action: 'Block Domain on Proxy',
    ts: '12:58:44',
  },
  {
    id: 'AX-2086', priority: 4, sev: 'high', icon: '🔑',
    title: 'ROTATE CREDENTIALS',
    target: 'USER: SVC_DATABASE_PROD',
    mitre: 'T1078 — Valid Accounts',
    desc: 'Anomalous login from outside business hours. Service account used interactively. Rotate credentials and revoke all active sessions.',
    action: 'Rotate Cloud Access Keys',
    ts: '12:34:01',
  },
  {
    id: 'AX-2084', priority: 5, sev: 'high', icon: '🛡',
    title: 'PATCH & ISOLATE',
    target: 'HOST: WORKSTATION-07',
    mitre: 'T1203 — Exploitation for Client Execution',
    desc: 'Browser RCE via unpatched zero-day. Spawned child shell process detected. Isolate endpoint and initiate emergency patch cycle.',
    action: 'Isolate Endpoint from Network',
    ts: '11:48:33',
  },
];

export default function Playbook({ lastPipelineResult }) {
  const [queue, setQueue]                     = useState([]);
  const [resolved, setResolved]               = useState([]);
  const [overrideModalFor, setOverrideModalFor] = useState(null);
  const [overrideSearch, setOverrideSearch]   = useState('');
  const [overrideSelections, setOverrideSelections] = useState([]);
  const [countdown, setCountdown]             = useState(685);
  const [newAlertFlash, setNewAlertFlash]     = useState(false);
  const bottomRef = useRef(null);

  // Countdown timer
  useEffect(() => {
    if (countdown <= 0) return;
    const t = setInterval(() => setCountdown(c => Math.max(0, c - 1)), 1000);
    return () => clearInterval(t);
  }, []);

  // Seed queue from real backend on mount
  useEffect(() => {
    const seedFromBackend = async () => {
      try {
        const result = await runDemoInjection({});
        if (result?.verdict !== 'ANOMALOUS') return;
        const l1 = result.layer1 || {};
        const l3 = result.layer3 || {};
        const l5 = result.layer5 || {};
        const score = result.anomaly_score || 0.8;
        const attackType = l1.attack_type || 'Zeus_Banking_Trojan';
        const ACTION_MAP = {
          Zeus_Banking_Trojan: { title: 'BLOCK & ISOLATE',  action: 'Block Source IP + Isolate Endpoint', icon: '🏦', mitre: 'T1071 — C2 Communication' },
          BruteForce:          { title: 'RATE LIMIT',       action: 'Rate Limit + Force MFA',             icon: '🔑', mitre: 'T1110 — Brute Force' },
          SQLInjection:        { title: 'FREEZE DB',        action: 'Freeze Database Writes',             icon: '💉', mitre: 'T1190 — Exploit Public App' },
          PortScan:            { title: 'MONITOR & BLOCK',  action: 'Block Port Scan Source',             icon: '🔗', mitre: 'T1046 — Network Discovery' },
          DDoS:                { title: 'NULL ROUTE',       action: 'Null Route Attacker IP',             icon: '🌊', mitre: 'T1498 — Network DoS' },
          C2Beacon:            { title: 'SINKHOLE C2',      action: 'Block C2 Domain + Kill Process',     icon: '📡', mitre: 'T1071 — App Layer Protocol' },
          Ransomware:          { title: 'ISOLATE HOST',     action: 'Network Isolation + Backup Restore', icon: '🔒', mitre: 'T1486 — Data Encrypted' },
        };
        const mapped = ACTION_MAP[attackType] || { title: 'INVESTIGATE', action: 'Escalate to SOC Tier 2', icon: '⚠', mitre: 'T1059 — Command Execution' };
        const playbook = l5.playbook ? String(l5.playbook).slice(0, 150) : mapped.action;
        const srcIp = l1.source_ip || '203.0.113.99';
        setQueue([
          {
            id: result.alert_id || 'AX-LIVE-001',
            priority: 1, sev: score > 0.8 ? 'critical' : 'high',
            icon: mapped.icon, title: mapped.title,
            target: `IP: ${srcIp}`,
            mitre: l3?.layer2?.mitre_stage || mapped.mitre,
            desc: playbook,
            action: l3?.decision?.action_name || mapped.action,
            ts: new Date().toLocaleTimeString('en', { hour12: false }),
            _backendId: result.alert_id,
          }
        ]);
      } catch { /* backend offline — queue stays empty */ }
    };
    seedFromBackend();
  }, []);

  useEffect(() => {
    if (!lastPipelineResult) return;
    const l1 = lastPipelineResult.layer1 || {};
    const l3 = lastPipelineResult.layer3 || {};
    const l5 = lastPipelineResult.layer5 || {};
    const attackType = l1.attack_type || 'Unknown';
    const score = lastPipelineResult.anomaly_score || 0;
    if (!lastPipelineResult.verdict || lastPipelineResult.verdict !== 'ANOMALOUS') return;

    const ACTION_MAP = {
      Zeus_Banking_Trojan: { title: 'BLOCK & ISOLATE', action: 'Block Source IP + Isolate Endpoint', icon: '🏦', mitre: 'T1071 — C2 Communication' },
      BruteForce:          { title: 'RATE LIMIT',      action: 'Rate Limit + Force MFA',             icon: '🔑', mitre: 'T1110 — Brute Force' },
      SQLInjection:        { title: 'FREEZE DB',       action: 'Freeze Database Writes',             icon: '💉', mitre: 'T1190 — Exploit Public App' },
      PortScan:            { title: 'MONITOR & BLOCK', action: 'Block Port Scan Source',             icon: '🔗', mitre: 'T1046 — Network Discovery' },
      DDoS:                { title: 'NULL ROUTE',      action: 'Null Route Attacker IP',             icon: '🌊', mitre: 'T1498 — Network DoS' },
      C2Beacon:            { title: 'SINKHOLE C2',     action: 'Block C2 Domain + Kill Process',     icon: '📡', mitre: 'T1071 — App Layer Protocol' },
      Ransomware:          { title: 'ISOLATE HOST',    action: 'Network Isolation + Backup Restore', icon: '🔒', mitre: 'T1486 — Data Encrypted' },
    };
    const mapped = ACTION_MAP[attackType] || { title: 'INVESTIGATE', action: 'Escalate to SOC Tier 2', icon: '⚠', mitre: 'T1059 — Command Execution' };

    const playbook = l5.playbook ? String(l5.playbook).slice(0, 120) : mapped.action;
    const newItem = {
      id: 'AX-' + lastPipelineResult.alert_id?.slice(-6) || Math.random().toString(36).slice(2,8).toUpperCase(),
      priority: score > 0.8 ? 1 : score > 0.6 ? 2 : 3,
      sev: score > 0.8 ? 'critical' : score > 0.5 ? 'high' : 'medium',
      icon: mapped.icon,
      title: mapped.title,
      target: `IP: ${l1.source_ip || lastPipelineResult.source_ip || '—'}`,
      mitre: mapped.mitre,
      desc: playbook,
      action: mapped.action,
      ts: new Date().toLocaleTimeString('en', { hour12: false }),
      fromDetect: true,
      _l3: l3,
    };
    setQueue(prev => [newItem, ...prev]);
    setNewAlertFlash(true);
    setTimeout(() => setNewAlertFlash(false), 3000);
  }, [lastPipelineResult]);

  // Poll real L3 pending queue and merge into local queue
  useEffect(() => {
    const ICON_MAP = { DDoS: '🌊', BruteForce: '🔑', SQLInjection: '💉', PortScan: '🔗', C2Beacon: '📡', Ransomware: '🔒', default: '⚠' };
    const syncBackend = async () => {
      try {
        const raw = await getPendingApprovals();
        const items = Array.isArray(raw) ? raw : (raw.pending || raw.items || []);
        if (!items.length) return;
        setQueue(prev => {
          const existingIds = new Set(prev.map(q => q.id));
          const fresh = items
            .filter(r => !existingIds.has(r.alert_id))
            .map((r, i) => ({
              id: r.alert_id,
              priority: prev.length + i + 1,
              sev: r.anomaly_score > 0.8 ? 'critical' : r.anomaly_score > 0.5 ? 'high' : 'medium',
              icon: ICON_MAP[r.attack_type] || ICON_MAP.default,
              title: (r.attack_type || 'ANOMALY').toUpperCase().replace('_', ' '),
              target: `IP: ${r.source_ip || '—'}`,
              mitre: r.layer2?.mitre_stage || '—',
              desc: `L3 recommended action: ${r.decision?.action_name || '—'}. Score: ${r.anomaly_score?.toFixed(2) ?? '—'}`,
              action: r.decision?.action_name || 'flag_for_human_review',
              ts: new Date().toLocaleTimeString('en', { hour12: false }),
              _backendId: r.alert_id,
            }));
          if (fresh.length > 0) {
            setNewAlertFlash(true);
            setTimeout(() => setNewAlertFlash(false), 2000);
          }
          return [...prev, ...fresh];
        });
      } catch { /* backend offline — silently ignore */ }
    };
    syncBackend();
    const iv = setInterval(syncBackend, 15000);
    return () => clearInterval(iv);
  }, []);

  // Real backend approve/reject (wraps local state update)
  const handleApproveWithBackend = async (item) => {
    if (item._backendId) {
      try { await approveAction(item._backendId); } catch { /* ignore — local state still updates */ }
    }
    handleApprove(item);
  };

  const handleRejectWithBackend = async (item) => {
    if (item._backendId) {
      try { await rejectAction(item._backendId); } catch { /* ignore */ }
    }
    handleDismiss(item);
  };

  const mm = String(Math.floor(countdown / 60)).padStart(2, '0');
  const ss = String(countdown % 60).padStart(2, '0');

  // Approve top action
  const handleApprove = (item) => {
    setResolved(prev => [...prev, { ...item, outcome: 'approved', outMsg: `✅ ${item.action} executed — ${item.target} secured.` }]);
    setQueue(prev => prev.filter(q => q.id !== item.id));
  };

  // Dismiss
  const handleDismiss = (item) => {
    setResolved(prev => [...prev, { ...item, outcome: 'dismissed', outMsg: '🚫 Dismissed — logged for RLHF review.' }]);
    setQueue(prev => prev.filter(q => q.id !== item.id));
  };

  // Override modal
  const openOverride = (item) => {
    setOverrideModalFor(item);
    setOverrideSelections([]);
    setOverrideSearch('');
  };
  const closeOverride = () => { setOverrideModalFor(null); setOverrideSelections([]); setOverrideSearch(''); };

  const toggleOverride = (opt) =>
    setOverrideSelections(prev =>
      prev.some(o => o.index === opt.index)
        ? prev.filter(o => o.index !== opt.index)
        : [...prev, opt]
    );

  const applyOverride = async () => {
    if (!overrideModalFor || overrideSelections.length === 0) return;

    const ts = new Date().toISOString();

    // Build override payload JSON
    const payload = {
      generated_at: ts,
      incident_id: overrideModalFor.id,
      target: overrideModalFor.target,
      mitre: overrideModalFor.mitre,
      override_actions: overrideSelections.map(o => ({
        action_index: o.index,
        action_name: o.name,
        category: o.category,
      })),
    };

    // Build action_registry Python snippet
    const registryLines = overrideSelections
      .map(o => `    ${o.index}: "${o.name}",  # ${o.category}`)
      .join('\n');
    const registryPy =
`# IMMUNEX — Manual Override Registry Snapshot
# Incident : ${overrideModalFor.id}
# Target   : ${overrideModalFor.target}
# Generated: ${ts}

SELECTED_ACTIONS: dict[int, str] = {
${registryLines}
}
`;

    // Build README
    const readme =
`IMMUNEX Manual Override Package
================================
Incident : ${overrideModalFor.id}
Target   : ${overrideModalFor.target}
MITRE    : ${overrideModalFor.mitre}
Generated: ${ts}

Selected Actions (${overrideSelections.length})
${'─'.repeat(44)}
${overrideSelections.map((o, i) => `${String(i + 1).padStart(2, ' ')}. [ACT_${String(o.index).padStart(2, '0')}] ${o.name}  (${o.category})`).join('\n')}

Files in this package
─────────────────────
• README.txt                   — This file
• override_payload.json        — Machine-readable action list (IMMUNEX API ready)
• action_registry_snippet.py   — Python dict of selected actions
`;

    // Load JSZip dynamically and generate zip
    const loadJSZip = () => new Promise((resolve, reject) => {
      if (window.JSZip) { resolve(window.JSZip); return; }
      const s = document.createElement('script');
      s.src = 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js';
      s.onload = () => resolve(window.JSZip);
      s.onerror = reject;
      document.head.appendChild(s);
    });

    try {
      const JSZip = await loadJSZip();
      const zip = new JSZip();
      zip.file('README.txt', readme);
      zip.file('override_payload.json', JSON.stringify(payload, null, 2));
      zip.file('action_registry_snippet.py', registryPy);
      const blob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `IMMUNEX_override_${overrideModalFor.id}_${ts.slice(0, 10)}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      // Fallback: download payload JSON directly
      console.error('ZIP generation failed, falling back to JSON:', err);
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `IMMUNEX_override_${overrideModalFor.id}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }

    setResolved(prev => [...prev, {
      ...overrideModalFor,
      outcome: 'override',
      outMsg: `⛭ Manual override — ${overrideSelections.length} action${overrideSelections.length > 1 ? 's' : ''} applied & ZIP downloaded: ${overrideSelections.slice(0, 2).map(o => o.name).join(', ')}${overrideSelections.length > 2 ? ` +${overrideSelections.length - 2} more` : ''}.`,
    }]);
    setQueue(prev => prev.filter(q => q.id !== overrideModalFor.id));
    closeOverride();
  };

  const filteredOpts = OVERRIDE_OPTIONS.filter(o =>
    o.name.toLowerCase().includes(overrideSearch.toLowerCase()) ||
    o.category.toLowerCase().includes(overrideSearch.toLowerCase())
  );

  const groupedOpts = filteredOpts.reduce((acc, o) => {
    if (!acc[o.category]) acc[o.category] = [];
    acc[o.category].push(o);
    return acc;
  }, {});

  const CATEGORY_LABELS = {
    monitoring: 'Monitoring',
    network: 'Network Containment',
    credential: 'Credential & Access',
    process: 'Process & Endpoint',
    data_protection: 'Data Protection',
  };

  const sevColor = { critical: '#c0392b', high: '#e67e22', medium: '#f39c12', low: '#27ae60' };

  return (
    <div className="section-page active" style={{ display: 'block', position: 'relative' }}>

      {/* ── Override Modal ── */}
      {overrideModalFor && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(5px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
          <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 14, width: '100%', maxWidth: 580, maxHeight: '82vh', display: 'flex', flexDirection: 'column', boxShadow: '0 24px 48px rgba(0,0,0,0.5)', overflow: 'hidden' }}>

            {/* Modal header */}
            <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: 2, color: 'var(--text3)', textTransform: 'uppercase', marginBottom: 4 }}>Manual Override — {overrideModalFor.id}</div>
                <h3 style={{ margin: 0, fontFamily: 'var(--serif)', fontSize: 20, fontWeight: 500 }}>Select Response Actions</h3>
              </div>
              <button onClick={closeOverride} style={{ background: 'none', border: 'none', color: 'var(--text2)', cursor: 'pointer', fontSize: 22, lineHeight: 1 }}>&times;</button>
            </div>

            {/* Search */}
            <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
              <input
                type="text"
                placeholder="Search actions..."
                value={overrideSearch}
                onChange={e => setOverrideSearch(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', border: '1px solid var(--border)', borderRadius: 7, background: 'var(--bg2)', color: 'var(--text)', fontFamily: 'var(--mono)', fontSize: 12, outline: 'none', boxSizing: 'border-box' }}
              />
            </div>

            {/* Options list - grouped by category */}
            <div style={{ overflowY: 'auto', padding: 12, flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
              {filteredOpts.length === 0 && (
                <div style={{ textAlign: 'center', color: 'var(--text3)', fontFamily: 'var(--mono)', fontSize: 12, padding: 20 }}>No actions match "{overrideSearch}"</div>
              )}
              {Object.keys(groupedOpts).map(cat => (
                <div key={cat}>
                  {/* Category header */}
                  <div style={{
                    fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: 1.8, textTransform: 'uppercase',
                    color: CATEGORY_COLORS[cat]?.text || 'var(--text3)',
                    padding: '8px 4px 4px',
                    borderBottom: `1px solid ${CATEGORY_COLORS[cat]?.border || 'var(--border)'}`,
                    marginBottom: 4,
                    display: 'flex', alignItems: 'center', gap: 6,
                  }}>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: CATEGORY_COLORS[cat]?.text, display: 'inline-block', flexShrink: 0 }}></span>
                    {CATEGORY_LABELS[cat] || cat}
                    <span style={{ marginLeft: 'auto', opacity: 0.6 }}>{groupedOpts[cat].length}</span>
                  </div>
                  {/* Actions in category */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 8 }}>
                    {groupedOpts[cat].map(opt => {
                      const isSel = overrideSelections.some(o => o.index === opt.index);
                      const cc = CATEGORY_COLORS[opt.category];
                      return (
                        <button
                          key={opt.index}
                          onClick={() => toggleOverride(opt)}
                          style={{
                            textAlign: 'left', padding: '8px 12px',
                            background: isSel ? cc?.bg : 'var(--card)',
                            border: isSel ? `1px solid ${cc?.border}` : '1px solid var(--border)',
                            borderRadius: 7, color: 'var(--text)', fontSize: 12, cursor: 'pointer',
                            display: 'flex', alignItems: 'center', gap: 10, transition: 'all 0.12s',
                          }}
                        >
                          <div style={{
                            width: 15, height: 15, border: isSel ? `1.5px solid ${cc?.text}` : '1px solid var(--border2)',
                            borderRadius: 3, background: isSel ? cc?.text : 'transparent',
                            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                          }}>
                            {isSel && <span style={{ color: '#fff', fontSize: 10, fontWeight: 700, lineHeight: 1 }}>✓</span>}
                          </div>
                          <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text3)', minWidth: 40 }}>
                            ACT_{String(opt.index).padStart(2, '0')}
                          </span>
                          <span style={{ flex: 1, fontFamily: 'var(--mono)', fontSize: 11 }}>
                            {opt.name}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>

            {/* Footer */}
            <div style={{ padding: '14px 20px', borderTop: '1px solid var(--border)', background: 'var(--card)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text3)' }}>
                {overrideSelections.length} / {OVERRIDE_OPTIONS.length} selected
                {overrideSelections.length > 0 && (
                  <span style={{ marginLeft: 8, color: 'var(--text2)' }}>
                    · {[...new Set(overrideSelections.map(o => o.category))].length} categories
                  </span>
                )}
              </span>
              <div style={{ display: 'flex', gap: 10 }}>
                <button onClick={closeOverride} className="btn btn-secondary btn-sm">Cancel</button>
                <button
                  className="btn btn-primary"
                  disabled={overrideSelections.length === 0}
                  style={{ opacity: overrideSelections.length === 0 ? 0.5 : 1, display: 'flex', alignItems: 'center', gap: 6 }}
                  onClick={applyOverride}
                >
                  <span>⬇ Apply & Download ZIP</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Page Header ── */}
      <div className="section-page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div className="section-eyebrow">Response Playbook</div>
            <h2 className="section-title">Anomaly Queue — Decide fast. Act with precision.</h2>
            <p className="section-sub">Anomalies detected by the pipeline are queued here in priority order. Each requires exactly one response action.</p>
          </div>
          <div style={{ textAlign: 'right', flexShrink: 0, marginLeft: 24 }}>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: 1.5, marginBottom: 4 }}>Response Window</div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 32, fontWeight: 700, color: countdown < 120 ? '#c0392b' : 'var(--text)', letterSpacing: 2 }}>{mm}:{ss}</div>
          </div>
        </div>
      </div>

      {/* ── Stats bar ── */}
      <div className="playbook-stats" style={{ marginBottom: 24 }}>
        <div className="pb-stat">
          <div className="pb-stat-label">In Queue</div>
          <div className="pb-stat-val" style={{ color: queue.length > 0 ? '#c0392b' : 'var(--green)' }}>
            {String(queue.length).padStart(2, '0')}
          </div>
        </div>
        <div className="pb-stat">
          <div className="pb-stat-label">Resolved</div>
          <div className="pb-stat-val accent">{String(resolved.length).padStart(2, '0')}</div>
        </div>
        <div className="pb-stat">
          <div className="pb-stat-label">Critical Pending</div>
          <div className="pb-stat-val red">{queue.filter(q => q.sev === 'critical').length}</div>
        </div>
        <div className="pb-stat">
          <div className="pb-stat-label">Override Actions</div>
          <div className="pb-stat-val">{OVERRIDE_OPTIONS.length}</div>
        </div>
      </div>

      {/* ── Queue ── */}
      <div style={{ marginBottom: 32 }}>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: 2, textTransform: 'uppercase', color: 'var(--text3)', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 10 }}>
          <span>Pending Anomalies ({queue.length})</span>
          {newAlertFlash && (
            <span style={{ background: '#c0392b', color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 9, animation: 'none' }}>⚡ NEW ALERT</span>
          )}
        </div>

        {queue.length === 0 && (
          <div style={{ textAlign: 'center', padding: '48px 24px', color: 'var(--text3)', fontFamily: 'var(--mono)', fontSize: 13, background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 12 }}>
            ✓ All anomalies resolved — queue is empty
          </div>
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {queue.map((item, idx) => (
            <div
              key={item.id}
              style={{
                background: 'var(--card)',
                border: `1px solid ${idx === 0 ? sevColor[item.sev] + '88' : 'var(--border)'}`,
                borderLeft: `4px solid ${sevColor[item.sev] || 'var(--border)'}`,
                borderRadius: 10,
                overflow: 'hidden',
                boxShadow: idx === 0 ? `0 0 0 1px ${sevColor[item.sev]}22` : 'none',
                transition: 'all .2s',
              }}
            >
              <div style={{ padding: '14px 18px' }}>
                {/* Top row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <div style={{ width: 36, height: 36, borderRadius: 8, background: 'var(--bg2)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18, flexShrink: 0 }}>{item.icon}</div>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
                        <span style={{ fontFamily: 'var(--mono)', fontSize: 13, fontWeight: 700, letterSpacing: 1, color: 'var(--text)' }}>{item.title}</span>
                        <span style={{ fontFamily: 'var(--mono)', fontSize: 9, textTransform: 'uppercase', letterSpacing: 1.5, color: sevColor[item.sev], background: sevColor[item.sev] + '18', border: `1px solid ${sevColor[item.sev]}44`, borderRadius: 4, padding: '2px 7px' }}>{item.sev}</span>
                        {idx === 0 && <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: '#fff', background: '#c0392b', borderRadius: 4, padding: '2px 7px', letterSpacing: 1 }}>NEXT IN QUEUE</span>}
                      </div>
                      <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text3)' }}>{item.target}</div>
                    </div>
                  </div>
                  <div style={{ textAlign: 'right', flexShrink: 0 }}>
                    <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', marginBottom: 2 }}>{item.ts}</div>
                    <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)' }}>{item.id}</div>
                    <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--accentbg)', marginTop: 2 }}>{item.mitre}</div>
                  </div>
                </div>

                {/* Description */}
                <p style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.6, margin: '0 0 12px 0' }}>{item.desc}</p>

                {/* Recommended action tag */}
                <div style={{ marginBottom: 12 }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', marginRight: 8 }}>RECOMMENDED:</span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--accentbg)', background: 'rgba(61,90,62,0.1)', border: '1px solid rgba(61,90,62,0.3)', borderRadius: 4, padding: '2px 8px' }}>{item.action}</span>
                </div>

                {/* Three action buttons */}
                <div style={{ display: 'flex', gap: 8 }}>
                  <button
                    className="pb-approve"
                    style={{ flex: 1 }}
                    onClick={() => handleApprove(item)}
                  >
                    ✓ APPROVE
                  </button>
                  <button
                    className="pb-dismiss"
                    style={{ flex: 1 }}
                    onClick={() => handleDismiss(item)}
                  >
                    ✕ DISMISS
                  </button>
                  <button
                    className="pb-dismiss"
                    style={{ flex: 1, color: '#3498db', borderColor: '#3498db44' }}
                    onClick={() => openOverride(item)}
                  >
                    ⛭ OVERRIDE
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Resolved log ── */}
      {resolved.length > 0 && (
        <div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: 2, textTransform: 'uppercase', color: 'var(--text3)', marginBottom: 12 }}>Resolved ({resolved.length})</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[...resolved].reverse().map((item) => (
              <div
                key={item.id + item.outcome}
                style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderLeft: `4px solid ${item.outcome === 'approved' ? '#27ae60' : item.outcome === 'override' ? '#3498db' : '#7f8c8d'}`, borderRadius: 8, padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 14, opacity: 0.8 }}
              >
                <span style={{ fontSize: 18 }}>{item.icon}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 600, color: 'var(--text)', marginBottom: 2 }}>{item.title} — {item.target}</div>
                  <div style={{ fontSize: 11, color: 'var(--text3)' }}>{item.outMsg}</div>
                </div>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', flexShrink: 0 }}>{item.id}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div ref={bottomRef}></div>
    </div>
  );
}
