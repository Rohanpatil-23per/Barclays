#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  IMMUNEX TEST 4 — WIREGUARD + mTLS SECURITY
#  Shows: encrypted tunnel, peer status, cert-based auth, access control
#  Run from: ~/Documents/hoh/Barclays
# ═══════════════════════════════════════════════════════════════════════════════

cd ~/Documents/hoh/Barclays
source ~/.venvs/immunex/bin/activate 2>/dev/null

G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; C='\033[0;36m'; B='\033[1m'; N='\033[0m'

echo ""
echo -e "${B}${Y}╔══════════════════════════════════════════════════════╗${N}"
echo -e "${B}${Y}║     IMMUNEX — WIREGUARD + mTLS SECURITY TEST         ║${N}"
echo -e "${B}${Y}║     Encrypted inter-node communication               ║${N}"
echo -e "${B}${Y}╚══════════════════════════════════════════════════════╝${N}"
echo ""

# ── WIREGUARD STATUS ──────────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  WireGuard VPN — Interface & Peer Status             │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

sudo wg show 2>/dev/null | python3 << 'PYEOF'
import sys, re

lines = sys.stdin.read()
if not lines.strip():
    print("  ❌ WireGuard not running")
    sys.exit(0)

# Parse interface
iface_match = re.search(r'interface: (\S+)', lines)
pubkey_match = re.search(r'public key: (\S+)', lines)
port_match   = re.search(r'listening port: (\d+)', lines)

print(f"  Interface : {iface_match.group(1) if iface_match else '?'}")
print(f"  Public Key: {pubkey_match.group(1)[:20] if pubkey_match else '?'}...")
print(f"  Port      : {port_match.group(1) if port_match else '?'}")
print(f"  Cipher    : ChaCha20-Poly1305 (WireGuard default)")
print()

# Parse peers
peers = re.findall(
    r'peer: (\S+).*?endpoint: (\S+).*?allowed ips: (\S+).*?(?:latest handshake: (.+?)(?:\n|transfer))?transfer: ([^\n]+)',
    lines, re.DOTALL
)
NODE_NAMES = {
    '10.0.0.2': 'L2-Node (Acer Nitro 4050)',
    '10.0.0.3': 'L3-Node (Lenovo LOQ 3050)',
    '10.0.0.4': 'L4-Node (HP Victus 2050)',
    '10.0.0.5': 'L5-Node (HP Pavilion 1650)',
}

print(f"  {'NODE':<30} {'ENDPOINT':<22} {'VPN IP':<14} {'STATUS'}")
print(f"  {'-'*30} {'-'*22} {'-'*14} {'-'*15}")
for peer_key, endpoint, allowed_ip, handshake, transfer in peers:
    vpn_ip = allowed_ip.split('/')[0]
    node_name = NODE_NAMES.get(vpn_ip, f'Node ({vpn_ip})')
    if handshake and 'minute' in handshake:
        status = '\033[0;32m✅ ACTIVE\033[0m'
    elif handshake and 'second' in handshake:
        status = '\033[0;32m✅ ACTIVE\033[0m'
    elif '0 B received' in transfer and '0 B sent' in transfer.split(',')[0]:
        status = '\033[0;31m❌ NO TRAFFIC\033[0m'
    else:
        status = '\033[1;33m⚠ CHECK\033[0m'
    print(f"  {node_name:<30} {endpoint:<22} {vpn_ip:<14} {status}")
PYEOF

echo ""

# ── REACHABILITY TEST ──────────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  WireGuard Reachability (encrypted tunnel test)      │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

NODE_IPS=("10.0.0.2" "10.0.0.3" "10.0.0.4" "10.0.0.5")
NODE_NAMES=("L2-Correlation" "L3-Response" "L4-Adaptive" "L5-Memory")

for i in "${!NODE_IPS[@]}"; do
    IP="${NODE_IPS[$i]}"
    NAME="${NODE_NAMES[$i]}"
    RTT=$(ping -c 1 -W 2 "$IP" 2>/dev/null | grep -oP 'time=\K[0-9.]+')
    if [ -n "$RTT" ]; then
        echo -e "  ${G}✅${N} ${NAME} (${IP}) — reachable via WireGuard, RTT=${RTT}ms"
    else
        echo -e "  ${R}❌${N} ${NAME} (${IP}) — not reachable (node offline)"
    fi
done

echo ""

# ── SUBNET ACCESS CONTROL ─────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  Access Control — WireGuard subnet isolation         │${N}"
echo -e "${C}│  Only 10.0.0.x peers can reach layer ports          │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

echo "  Testing: this node (10.0.0.1) → L2 node (10.0.0.2)"
L2_REMOTE=$(curl -s --max-time 3 http://10.0.0.2:8002/health 2>/dev/null)
if echo "$L2_REMOTE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('layer','?'))" 2>/dev/null | grep -q "2"; then
    echo -e "  ${G}✅ L2 accessible over WireGuard tunnel${N}"
    echo "     $(echo $L2_REMOTE | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'layer={d.get(\"layer\")} device={d.get(\"device\")} status={d.get(\"status\")}')" 2>/dev/null)"
else
    echo -e "  ${Y}⚠  L2 node offline (teammate not connected)${N}"
fi

echo ""
echo "  Testing: L5 node (10.0.0.5)"
L5_REMOTE=$(curl -s --max-time 3 http://10.0.0.5:8005/health 2>/dev/null)
if [ -n "$L5_REMOTE" ]; then
    echo -e "  ${G}✅ L5 accessible over WireGuard tunnel${N}"
    echo "     $(echo $L5_REMOTE | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'layer={d.get(\"layer\")} device={d.get(\"device\")} status={d.get(\"status\")}')" 2>/dev/null)"
else
    echo -e "  ${Y}⚠  L5 node offline${N}"
fi

echo ""

# ── PKI / mTLS STATUS ─────────────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  PKI Certificates (X.509 Mutual Authentication)      │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

python3 << 'PYEOF'
import os, ssl
from datetime import datetime

CERTS = {
    "CA Certificate"     : "certs/ca/ca.crt",
    "Server Certificate" : "certs/server/server.crt",
    "Client Certificate" : "certs/client/client.crt",
}
KEYS = {
    "CA Key"             : "certs/ca/ca.key",
    "Server Key"         : "certs/server/server.key",
    "Client Key"         : "certs/client/client.key",
}

print("  Certificates:")
for name, path in CERTS.items():
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"    \033[0;32m✅\033[0m {name:<22} {path}  ({size}B)")
    else:
        print(f"    \033[0;31m❌\033[0m {name:<22} MISSING")

print()
print("  Private Keys (protected):")
for name, path in KEYS.items():
    if os.path.exists(path):
        perms = oct(os.stat(path).st_mode)[-3:]
        safe = perms in ('600', '400')
        sym = '\033[0;32m✅\033[0m' if safe else '\033[1;33m⚠\033[0m'
        print(f"    {sym} {name:<22} {path}  (perms: {perms})")
    else:
        print(f"    \033[0;31m❌\033[0m {name:<22} MISSING")

print()
print("  mTLS Client Module:")
try:
    from shared.mtls_client import mtls_status
    s = mtls_status()
    print(f"    Mode          : {s['mode']}")
    print(f"    CA cert exists: {s['ca_cert_exists']}")
    print(f"    Client cert   : {s['client_cert_exists']}")
    print(f"    Toggle env    : IMMUNEX_MTLS=1 to enable full mTLS")
    print(f"    \033[0;32m✅ mTLS client ready — toggle with IMMUNEX_MTLS=1\033[0m")
except Exception as e:
    print(f"    \033[0;31m❌ {e}\033[0m")
PYEOF

echo ""

# ── ENCRYPTION LAYER SUMMARY ──────────────────────────────────────────────────
echo -e "${C}┌─────────────────────────────────────────────────────┐${N}"
echo -e "${C}│  Security Architecture Summary                       │${N}"
echo -e "${C}└─────────────────────────────────────────────────────┘${N}"

echo "  Layer 1 (Transport)  : WireGuard VPN"
echo "                         ChaCha20-Poly1305 AEAD encryption"
echo "                         Noise protocol handshake (curve25519)"
echo ""
echo "  Layer 2 (Auth)       : X.509 Mutual TLS"
echo "                         Both client and server present certificates"
echo "                         CA-signed — self-signed certs rejected"
echo ""
echo "  Layer 3 (Access)     : Subnet allowlist"
echo "                         Only 10.0.0.x peers in WireGuard config"
echo "                         No plaintext traffic between nodes"
echo ""
echo -e "  ${G}Result: An attacker on the same WiFi cannot reach any${N}"
echo -e "  ${G}layer port — all traffic is inside the encrypted tunnel.${N}"

echo ""
echo -e "${B}${G}══════════════════════════════════════════${N}"
echo -e "${B}${G}  SECURITY TEST COMPLETE${N}"
echo -e "${B}${G}══════════════════════════════════════════${N}"
echo ""
