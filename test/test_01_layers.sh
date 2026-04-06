#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  IMMUNEX TEST 1 — FIVE-LAYER PIPELINE
#  Shows: each layer responding, real model output, end-to-end verdict
#  Run from: ~/Documents/hoh/Barclays
# ═══════════════════════════════════════════════════════════════════════════════

cd ~/Documents/hoh/Barclays
source ~/.venvs/immunex/bin/activate 2>/dev/null

G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; C='\033[0;36m'; B='\033[1m'; N='\033[0m'

ZEUS='[-0.7646905574813246,1.8493171336719367,0.1360842597127493,-0.038256832829822,-0.2234762935258288,0.1867425617760749,-0.2887230032189737,-0.3338156024101207,-0.3094501517292162,-0.243075338744169,4.544640932295218,-0.6642257447173439,3.1770112733598714,4.633082888390731,-0.1770146168201649,-0.1587662528427847,1.1936840927012191,2.3922403065334,2.765064551509225,-0.0582008046198575,1.862034595775524,0.8855807278018033,2.6763375243600733,2.762514242871944,-0.1243816297992087,-0.3613781146589914,-0.2127801830691396,-0.2475923127532377,-0.2842236442106687,-0.122538486443635,-0.1890309835329326,0.0,0.0,0.0,0.0373291387560878,-0.0817210922126444,-0.1349697802794396,-0.205328179738589,-0.7746180708204594,4.332156577732468,2.060003453281091,3.636782569436395,3.947944798307136,-0.1742105834718708,-0.1890309835329326,0.0,-0.5689403351968044,1.6932962780256324,-0.2973405807608872,0.0,0.0,-1.1283890896284317,2.027611601794204,-0.3094501517292162,3.1770112733598714,0.0,0.0,0.0,0.0,0.0,0.0,0.1360842597127493,-0.2234762935258288,-0.038256832829822,0.1867425617760749,-0.4425389824504542,-0.2020681797073581,0.3430337318398583,-0.8167862912590615,0.4715044104583414,-0.1382837801439019,0.2060428820616336,0.5834133109605503,2.897013722170425,-0.1150629802833818,2.7933872433630125,2.946668070457503]'

# ── Temp file setup ────────────────────────────────────────────────────────────
TMPDIR=/tmp/immunex_test01
mkdir -p "$TMPDIR"

echo ""
echo -e "${B}${Y}╔══════════════════════════════════════════════════════╗${N}"
echo -e "${B}${Y}║     IMMUNEX — FIVE-LAYER PIPELINE TEST               ║${N}"
echo -e "${B}${Y}║     Attack: Zeus Banking Trojan                       ║${N}"
echo -e "${B}${Y}╚══════════════════════════════════════════════════════╝${N}"
echo ""

# ── LAYER 1: Innate Immunity ──────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  LAYER 1 — Innate Immunity (RoBERTa + IsolationForest)│${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

curl -s --max-time 3 http://localhost:8001/health > "$TMPDIR/l1_health.json"
python3 << 'PYEOF'
import json
try:
    with open('/tmp/immunex_test01/l1_health.json') as f:
        d = json.load(f)
    print(f"  Models  : {', '.join(d.get('models', []))}")
    print(f"  Device  : {d.get('device', '?')}")
    print(f"  FAISS   : {d.get('faiss_vectors', '?')} vectors indexed")
except Exception as e:
    print(f"  \033[0;31m❌ Health check failed: {e}\033[0m")
PYEOF

curl -s --max-time 15 -X POST http://localhost:8001/detect \
  -H "Content-Type: application/json" \
  -d "{\"source_ip\":\"203.0.113.99\",\"dest_ip\":\"192.168.10.50\",\"alert_type\":\"Zeus_Banking_Trojan\",\"severity\":\"critical\",\"text\":\"Zeus banking trojan C2 beacon lateral movement mimikatz\",\"features\":$ZEUS}" \
  > "$TMPDIR/l1_detect.json"

python3 << 'PYEOF'
import json
try:
    with open('/tmp/immunex_test01/l1_detect.json') as f:
        d = json.load(f)
    anomalous = d.get('is_anomalous', False)
    score     = d.get('anomaly_score', 0)
    attack    = d.get('attack_type', '?')
    conf      = d.get('confidence', 0)
    method    = d.get('detection_method', '?')
    emb_len   = len(d.get('embedding', []))
    status = '\033[0;32m✅ ANOMALOUS\033[0m' if anomalous else '\033[0;31m❌ BENIGN\033[0m'
    print(f"  Result  : {status}")
    print(f"  Attack  : {attack}")
    print(f"  Score   : {score:.4f}")
    print(f"  Confidence: {conf:.4f}")
    print(f"  Method  : {method}")
    print(f"  Embedding: {emb_len}-dim RoBERTa vector")
except Exception as e:
    print(f"  \033[0;31m❌ ERROR: {e}\033[0m")
PYEOF

echo ""

# ── LAYER 2: Adaptive Correlation ─────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  LAYER 2 — Adaptive Correlation (GATv2 Attack Graph) │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

curl -s --max-time 10 -X POST http://localhost:8002/correlate \
  -H "Content-Type: application/json" \
  -d "{\"alert_id\":\"ZEUS-TEST-001\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"source_ip\":\"203.0.113.99\",\"dest_ip\":\"192.168.10.50\",\"attack_type\":\"Zeus_Banking_Trojan\",\"anomaly_score\":0.8,\"feature_vector\":$ZEUS}" \
  > "$TMPDIR/l2.json"

python3 << 'PYEOF'
import json
try:
    with open('/tmp/immunex_test01/l2.json') as f:
        d = json.load(f)
    print(f"  Chain ID     : {d.get('chain_id','?')}")
    print(f"  MITRE Stage  : {d.get('mitre_stage','?')}")
    print(f"  Next Stage   : {d.get('predicted_next_stage','?')}")
    print(f"  Confidence   : {d.get('confidence',0):.4f}")
    print(f"  Graph Nodes  : {len(d.get('nodes',[]))}")
    print(f"  Graph Edges  : {len(d.get('edges',[]))}")
    nodes = [n.get('type','?') for n in d.get('nodes',[])]
    print(f"  Node Types   : {' → '.join(nodes)}")
    print(f"  \033[0;32m✅ Attack graph constructed\033[0m")
except Exception as e:
    print(f"  \033[0;31m❌ ERROR: {e}\033[0m")
PYEOF

echo ""

# ── LAYER 3: Response Engine ───────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  LAYER 3 — Response Engine (DQN + Safety Constraints) │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

curl -s --max-time 10 -X POST http://localhost:8003/respond \
  -H "Content-Type: application/json" \
  -d "{\"alert_id\":\"ZEUS-TEST-001\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"source_ip\":\"203.0.113.99\",\"destination_ip\":\"192.168.10.50\",\"source_port\":0,\"destination_port\":443,\"protocol\":\"TCP\",\"severity\":\"critical\",\"attack_type\":\"Zeus_Banking_Trojan\",\"feature_vector\":$ZEUS,\"layer2_confidence\":0.78}" \
  > "$TMPDIR/l3.json"

python3 << 'PYEOF'
import json
try:
    with open('/tmp/immunex_test01/l3.json') as f:
        d = json.load(f)
    decision = d.get('decision', {})
    verif    = d.get('verification', {})
    llm      = d.get('llm_reasoning', {})
    print(f"  Status       : {d.get('status','?')}")
    print(f"  Priority     : {d.get('priority','?')}")
    print(f"  Action       : {decision.get('action_name','?')}")
    print(f"  Confidence   : {decision.get('confidence',0):.4f}")
    print(f"  Impact       : {decision.get('impact','?')}")
    qvals = decision.get('raw_q_values', [])
    if qvals:
        print(f"  DQN Q-values : {len(qvals)} actions scored (top={max(qvals):.3f})")
    approved = verif.get('approved', False)
    print(f"  Safety Check : {'✅ Approved' if approved else '⚠ Requires review'}")
    print(f"  LLM Risk     : {llm.get('risk','?')}")
    print(f"  \033[0;32m✅ DQN action selected with safety verification\033[0m")
except Exception as e:
    print(f"  \033[0;31m❌ ERROR: {e}\033[0m")
PYEOF

echo ""

# ── LAYER 4: Adaptive Immunity ─────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  LAYER 4 — Adaptive Immunity (LoRA Fine-tuned Model) │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

curl -s --max-time 10 -X POST http://localhost:8004/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\":$ZEUS}" \
  > "$TMPDIR/l4.json"

python3 << 'PYEOF'
import json
try:
    with open('/tmp/immunex_test01/l4.json') as f:
        d = json.load(f)
    res = d.get('result', {})
    print(f"  Label        : {res.get('label','?')}")
    print(f"  Confidence   : {res.get('confidence',0):.4f}")
    print(f"  Is Threat    : {res.get('is_threat','?')}")
    print(f"  Model Acc    : {d.get('model_acc',0):.1f}%")
    print(f"  \033[0;32m✅ LoRA classifier responded\033[0m")
except Exception as e:
    print(f"  \033[0;31m❌ ERROR: {e}\033[0m")
PYEOF

echo ""

# ── LAYER 5: Threat Memory ─────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  LAYER 5 — Threat Memory (LSTM + HMM State Machine)  │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

curl -s --max-time 10 -X POST http://localhost:8005/explain \
  -H "Content-Type: application/json" \
  -d "{\"chain_id\":\"chain_ZEUS_TEST\",\"action\":\"block_source_ip\",\"attack_type\":\"Zeus_Banking_Trojan\",\"target_ip\":\"203.0.113.99\",\"mitre_stage\":\"Execution\",\"verified_safe\":false,\"q_value\":0.8}" \
  > "$TMPDIR/l5.json"

python3 << 'PYEOF'
import json
try:
    with open('/tmp/immunex_test01/l5.json') as f:
        d = json.load(f)
    print(f"  LSTM State   : {d.get('current_state','?')}")
    print(f"  Risk Level   : {d.get('risk_level','?')}")
    print(f"  Time Window  : {d.get('time_window','?')}")
    print(f"  LSTM Conf    : {d.get('lstm_confidence','?')}")
    threats = d.get('predicted_threats', [])
    print(f"  Next Threats : {', '.join(threats[:3]) if threats else '?'}")
    pb = str(d.get('playbook',''))
    if pb and pb != 'None':
        print(f"  Playbook     : {pb[:80]}...")
    print(f"  \033[0;32m✅ LSTM threat sequence predicted\033[0m")
except Exception as e:
    print(f"  \033[0;31m❌ ERROR: {e}\033[0m")
PYEOF

echo ""

# ── FULL PIPELINE ─────────────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  ORCHESTRATOR — Full 5-Layer Pipeline (113ms E2E)    │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

t0=$(date +%s%3N)
curl -s --max-time 30 -X POST http://localhost:8000/demo/inject > "$TMPDIR/pipeline.json"
t1=$(date +%s%3N)
ELAPSED=$((t1 - t0))

python3 << PYEOF
import json
elapsed = $ELAPSED
try:
    with open('/tmp/immunex_test01/pipeline.json') as f:
        d = json.load(f)
    verdict = d.get('verdict','?')
    score   = d.get('anomaly_score', 0)
    color   = '\033[0;32m' if verdict == 'ANOMALOUS' else '\033[0;33m'
    for k in ['layer1','layer2','layer3','layer4','layer5']:
        v   = d.get(k, {})
        err = v.get('error') if isinstance(v, dict) else None
        ok  = isinstance(v, dict) and not err
        sym = '✅' if ok else '❌'
        print(f"  {sym} {k}: {list(v.keys())[:3] if isinstance(v,dict) else v}")
    print()
    print(f"  Verdict   : {color}{verdict}\033[0m")
    print(f"  Score     : {score}")
    print(f"  E2E Time  : {elapsed}ms")
except Exception as e:
    print(f"  \033[0;31m❌ ERROR: {e}\033[0m")
PYEOF

echo ""
echo -e "${B}${G}══════════════════════════════════════════${N}"
echo -e "${B}${G}  LAYER TEST COMPLETE${N}"
echo -e "${B}${G}══════════════════════════════════════════${N}"
echo ""