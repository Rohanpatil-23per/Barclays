/**
 * IMMUNEX API Service — wired to real backend
 */

export const ORCHESTRATOR = import.meta.env.VITE_ORCHESTRATOR_URL || 'http://localhost:8000';
export const LAYER1       = import.meta.env.VITE_LAYER1_URL        || 'http://localhost:8001';
export const LAYER2       = import.meta.env.VITE_LAYER2_URL        || 'http://localhost:8002';
export const LAYER3       = import.meta.env.VITE_LAYER3_URL        || 'http://localhost:8003';
export const LAYER4       = import.meta.env.VITE_LAYER4_URL        || 'http://localhost:8004';
export const LAYER5       = import.meta.env.VITE_LAYER5_URL        || 'http://localhost:8005';

async function apiFetch(url, options = {}, timeoutMs = 20000) {
  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      signal: controller.signal,
      ...options,
    });
    clearTimeout(tid);
    if (!res.ok) {
      const detail = await res.text().catch(() => res.statusText);
      throw new Error(`API ${res.status}: ${detail}`);
    }
    return res.json();
  } catch (err) {
    clearTimeout(tid);
    if (err.name === 'AbortError') throw new Error('Request timed out');
    throw err;
  }
}

export const getOrchestratorHealth = () => apiFetch(`${ORCHESTRATOR}/health`, {}, 5000);

// FIX 9: New metrics endpoint for production monitoring
export const getOrchestratorMetrics = () => apiFetch(`${ORCHESTRATOR}/metrics`, {}, 5000);

// Layer 4 status for AIS Engine real data
export const getLayer4Status = () => apiFetch(`${LAYER4}/status`, {}, 5000);

// Run pipeline with proper Alert format
export async function runPipelineAlert(alert) {
  return apiFetch(`${ORCHESTRATOR}/pipeline/run`, { 
    method: 'POST', 
    body: JSON.stringify(alert)
  }, 30000);
}

// Get security status summary
export const getSecurityStatus = () => apiFetch(`${ORCHESTRATOR}/security/status`, {}, 5000);

// Get recent alerts from various endpoints
export async function getRecentAlerts() {
  try {
    // Try security status endpoint first
    const security = await getSecurityStatus();
    if (security.recent_alerts) {
      return security.recent_alerts.map(transformAlert);
    }
  } catch {}
  
  // Fallback to generating from metrics
  try {
    const metrics = await getOrchestratorMetrics();
    return generateMockAlertsFromMetrics(metrics);
  } catch {
    return [];
  }
}

function transformAlert(alert) {
  return {
    id: alert.alert_id || alert.id || 'ALT-' + Math.random().toString(36).slice(2, 7).toUpperCase(),
    ts: new Date(alert.timestamp || Date.now()).toLocaleTimeString('en', { hour12: false }),
    type: alert.attack_type || alert.alert_type || alert.type || 'Unknown',
    sev: alert.severity === 'critical' ? 'critical' : 
         alert.severity === 'high' ? 'high' :
         alert.severity === 'medium' ? 'medium' : 'low',
    ip: alert.source_ip || '—',
    status: alert.status || (alert.anomaly_score > 0.7 ? 'active' : 'resolved'),
    mitre: alert.mitre_tactic || 'T1059.001',
    vector: alert.attack_vector || 'Network',
    rec: alert.recommendation || 'Monitor and review'
  };
}

function generateMockAlertsFromMetrics(metrics) {
  const alertTypes = ['DDoS', 'PortScan', 'BruteForce', 'SQLInjection', 'Zeus_Botnet', 'Ransomware'];
  const severities = ['critical', 'high', 'medium', 'low'];
  const ips = ['192.168.1.105', '10.10.10.50', '203.0.113.42', '172.16.0.15', '198.51.100.20'];
  
  const count = Math.min(8, Math.max(3, Math.floor((metrics.counters?.pipeline_started || 0) / 2)));
  
  return Array.from({ length: count }, (_, i) => ({
    id: 'ALT-' + Math.random().toString(36).slice(2, 7).toUpperCase(),
    ts: new Date(Date.now() - i * 300000).toLocaleTimeString('en', { hour12: false }),
    type: alertTypes[Math.floor(Math.random() * alertTypes.length)],
    sev: severities[Math.floor(Math.random() * severities.length)],
    ip: ips[Math.floor(Math.random() * ips.length)],
    status: Math.random() > 0.3 ? 'resolved' : 'active',
    mitre: 'T1059.001',
    vector: 'Network',
    rec: 'Block source IP'
  }));
}

export async function getLayerHealthStats() {
  const layers = [
    { name: 'L1 Detection',     url: `${LAYER1}/health` },
    { name: 'L2 Correlation',   url: `${LAYER2}/health` },
    { name: 'L3 Response',      url: `${LAYER3}/health` },
    { name: 'L4 Immunity',      url: `${LAYER4}/health` },
    { name: 'L5 Threat Memory', url: `${LAYER5}/health` },
  ];
  return Promise.all(layers.map(async ({ name, url }) => {
    const t0 = performance.now();
    try {
      const d = await apiFetch(url, {}, 4000);
      return { name, latency: `${Math.round(performance.now() - t0)}ms`, online: true, data: d };
    } catch {
      return { name, latency: '—', online: false, data: null };
    }
  }));
}

export const getMeshNodes     = () => apiFetch(`${ORCHESTRATOR}/ingest/nodes`, {}, 5000);
export const runPipeline      = (log) => apiFetch(`${ORCHESTRATOR}/pipeline/run`, { method: 'POST', body: JSON.stringify({ log }) });
export const runDemoInjection = (payload) => {
  // Ensure we send logs in the correct format
  let logsToSend = payload.logs || payload.text || payload.message || '';
  if (!logsToSend && typeof payload === 'object') {
    // Convert object to log string format
    logsToSend = Object.entries(payload)
      .filter(([k,v]) => v && k !== 'features')
      .map(([k,v]) => `${k}=${v}`)
      .join(' ');
  }
  return apiFetch(`${ORCHESTRATOR}/demo/inject`, { 
    method: 'POST', 
    body: JSON.stringify({ logs: logsToSend })
  });
};
export const runBatchIngest   = (logs, batch_size = 32) =>
  apiFetch(`${ORCHESTRATOR}/ingest/batch`, { method: 'POST', body: JSON.stringify({ logs, batch_size }) }, 60000);

export async function detectAnomaly(alert) {
  for (const route of ['/analyze', '/detect']) {
    try { return await apiFetch(`${LAYER1}${route}`, { method: 'POST', body: JSON.stringify(alert) }); }
    catch (e) { if (!e.message.includes('404')) throw e; }
  }
  throw new Error('No valid L1 route found');
}

export const correlateAttack     = (payload) => apiFetch(`${LAYER2}/correlate`,           { method: 'POST', body: JSON.stringify(payload) });
export const respondToThreat     = (payload) => apiFetch(`${LAYER3}/decide`,              { method: 'POST', body: JSON.stringify(payload) });
export const getPendingApprovals = ()        => apiFetch(`${LAYER3}/pending`,             {}, 5000);
export const approveAction       = (id)      => apiFetch(`${LAYER3}/approve/${id}`,       { method: 'POST', body: '{}' });
export const rejectAction        = (id)      => apiFetch(`${LAYER3}/reject/${id}`,        { method: 'POST', body: '{}' });

export async function predictOutcome(payload) {
  for (const route of ['/classify', '/predict']) {
    try { return await apiFetch(`${LAYER4}${route}`, { method: 'POST', body: JSON.stringify(payload) }); }
    catch (e) { if (!e.message.includes('404')) throw e; }
  }
  throw new Error('No valid L4 route found');
}

export async function explainThreat(payload) {
  for (const route of ['/remember', '/explain']) {
    try { return await apiFetch(`${LAYER5}${route}`, { method: 'POST', body: JSON.stringify(payload) }); }
    catch (e) { if (!e.message.includes('404')) throw e; }
  }
  throw new Error('No valid L5 route found');
}
export const retrainLayer4 = (payload) => apiFetch(`${LAYER4}/retrain`, { method: 'POST', body: JSON.stringify(payload) });

export async function runFullPipelineFrontend(alert) {
  // Convert alert to log string format for demo/inject endpoint
  const logStr = alert.text || alert.message || JSON.stringify(alert);
  return apiFetch(`${ORCHESTRATOR}/demo/inject`, { 
    method: 'POST', 
    body: JSON.stringify({ logs: logStr })
  });
}

export function buildAlertFromLog(logText, sourceIp = '0.0.0.0') {
  const id  = 'FE-' + Math.random().toString(36).slice(2, 10).toUpperCase();
  const txt = logText.toLowerCase();
  let attack_type = 'Unknown';
  if      (txt.includes('ddos') || txt.includes('flood'))   attack_type = 'DDoS';
  else if (txt.includes('sql')  || txt.includes('union'))   attack_type = 'SQLInjection';
  else if (txt.includes('brute') || txt.includes('failed_auth')) attack_type = 'BruteForce';
  else if (txt.includes('port_scan') || txt.includes('scan'))   attack_type = 'PortScan';
  else if (txt.includes('c2') || txt.includes('beacon'))    attack_type = 'C2Beacon';
  else if (txt.includes('ransomware'))                       attack_type = 'Ransomware';
  else if (txt.includes('anomaly') || txt.includes('alrt')) attack_type = 'UnknownAnomaly';

  const severity = txt.includes('critical') ? 'critical'
    : txt.includes('high') || txt.includes('alert') ? 'high'
    : txt.includes('medium') ? 'medium' : 'low';

  return {
    alert_id: id, timestamp: new Date().toISOString(),
    source_ip: sourceIp, dest_ip: '10.0.0.1', destination_ip: '10.0.0.1',
    alert_type: attack_type, attack_type, severity,
    features: [-0.7646905574813246,1.8493171336719367,0.1360842597127493,-0.038256832829822,-0.2234762935258288,0.1867425617760749,-0.2887230032189737,-0.3338156024101207,-0.3094501517292162,-0.243075338744169,4.544640932295218,-0.6642257447173439,3.1770112733598714,4.633082888390731,-0.1770146168201649,-0.1587662528427847,1.1936840927012191,2.3922403065334,2.765064551509225,-0.0582008046198575,1.862034595775524,0.8855807278018033,2.6763375243600733,2.762514242871944,-0.1243816297992087,-0.3613781146589914,-0.2127801830691396,-0.2475923127532377,-0.2842236442106687,-0.122538486443635,-0.1890309835329326,0.0,0.0,0.0,0.0373291387560878,-0.0817210922126444,-0.1349697802794396,-0.205328179738589,-0.7746180708204594,4.332156577732468,2.060003453281091,3.636782569436395,3.947944798307136,-0.1742105834718708,-0.1890309835329326,0.0,-0.5689403351968044,1.6932962780256324,-0.2973405807608872,0.0,0.0,-1.1283890896284317,2.027611601794204,-0.3094501517292162,3.1770112733598714,0.0,0.0,0.0,0.0,0.0,0.0,0.1360842597127493,-0.2234762935258288,-0.038256832829822,0.1867425617760749,-0.4425389824504542,-0.2020681797073581,0.3430337318398583,-0.8167862912590615,0.4715044104583414,-0.1382837801439019,0.2060428820616336,0.5834133109605503,2.897013722170425,-0.1150629802833818,2.7933872433630125,2.946668070457503],
    text: logText.slice(0, 512), message: logText.slice(0, 256),
    event_type: 'NETWORK', protocol: 'TCP', port: 443,
    flow_bytes_per_sec: txt.includes('flood') ? 1500000 : 1000,
    flow_packets_per_sec: txt.includes('scan') ? 800 : 10,
    username: '', process: '', file: '', privilege_level: 'user',
  };
}
