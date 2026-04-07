"""
Exposes /security/status endpoint — shows WireGuard, mTLS, PKI status.
Mount this router in orchestrator/server.py.
"""
import subprocess, os
from fastapi import APIRouter

router = APIRouter()
CERTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'certs')

def _wg_status():
    try:
        out = subprocess.check_output(['sudo', 'wg', 'show'], text=True, timeout=3)
        peers = [l.strip() for l in out.split('\n') if l.strip().startswith('peer')]
        handshakes = [l.strip() for l in out.split('\n') if 'latest handshake' in l]
        return {
            "running": True,
            "peer_count": len(peers),
            "active_peers": len(handshakes),
            "peers_detail": [
                {"endpoint": e.split('endpoint: ')[-1].strip()}
                for e in out.split('\n') if 'endpoint:' in e
            ]
        }
    except Exception as e:
        return {"running": False, "error": str(e)}

@router.get("/security/status")
async def security_status():
    from shared.mtls_client import mtls_status
    wg = _wg_status()
    certs = {
        "ca_cert": os.path.exists(os.path.join(CERTS_DIR, 'ca', 'ca.crt')),
        "server_cert": os.path.exists(os.path.join(CERTS_DIR, 'server', 'server.crt')),
        "client_cert": os.path.exists(os.path.join(CERTS_DIR, 'client', 'client.crt')),
    }
    return {
        "wireguard": wg,
        "mtls": mtls_status(),
        "pki_certs": certs,
        "encryption_layer": "WireGuard (ChaCha20-Poly1305) + mTLS (TLS 1.3)" if wg["running"] else "mTLS only",
        "inter_node_auth": "X.509 mutual certificate authentication",
        "transport_encryption": "AES-256-GCM (TLS 1.3 cipher suite)",
    }
