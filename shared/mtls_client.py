"""
mTLS client for IMMUNEX inter-layer communication.
All orchestrator → layer calls go through this client.
Uses mutual TLS: server verifies client cert, client verifies server cert.
Toggle: set IMMUNEX_MTLS=0 to disable (plain HTTP fallback for demo frontend).
"""
import os
import ssl
import httpx

CERTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'certs')
CA_CERT    = os.path.join(CERTS_DIR, 'ca', 'ca.crt')
CLIENT_CRT = os.path.join(CERTS_DIR, 'client', 'client.crt')
CLIENT_KEY = os.path.join(CERTS_DIR, 'client', 'client.key')

MTLS_ENABLED = os.getenv("IMMUNEX_MTLS", "0") == "1"   # toggle

def make_client(timeout: float = 10.0) -> httpx.AsyncClient:
    """Return an httpx.AsyncClient — mTLS if IMMUNEX_MTLS=1, plain HTTP otherwise."""
    if MTLS_ENABLED and os.path.exists(CA_CERT):
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=CA_CERT)
        ctx.load_cert_chain(certfile=CLIENT_CRT, keyfile=CLIENT_KEY)
        ctx.check_hostname = False          # WireGuard IPs, no hostname
        transport = httpx.AsyncHTTPTransport(ssl=ctx)
        print("[mTLS] client created with mutual TLS")
        return httpx.AsyncClient(transport=transport, timeout=timeout)
    return httpx.AsyncClient(timeout=timeout)


def mtls_status() -> dict:
    return {
        "mtls_enabled": MTLS_ENABLED,
        "ca_cert_exists": os.path.exists(CA_CERT),
        "client_cert_exists": os.path.exists(CLIENT_CRT),
        "mode": "mTLS (encrypted)" if MTLS_ENABLED else "plain HTTP (demo mode)",
    }
