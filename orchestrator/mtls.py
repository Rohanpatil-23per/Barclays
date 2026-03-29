"""
mTLS client and server config for IMMUNEX.
All inter-node communication requires a cert signed by IMMUNEX-CA.
"""
import ssl
import os

CERTS = os.path.join(os.path.dirname(__file__), "..", "certs")

def get_client_ssl_context(node: int = 1) -> ssl.SSLContext:
    """
    Returns an SSL context for outbound httpx requests.
    Sends ROG's cert, verifies remote cert against CA.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_cert_chain(
        certfile=os.path.join(CERTS, f"node{node}.crt"),
        keyfile=os.path.join(CERTS, f"node{node}.key"),
    )
    ctx.load_verify_locations(os.path.join(CERTS, "ca.crt"))
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = False  # We use IPs not hostnames
    return ctx

def get_server_ssl_context(node: int) -> dict:
    """
    Returns uvicorn SSL kwargs for a given node.
    Requires client cert signed by IMMUNEX-CA.
    """
    return {
        "ssl_certfile": os.path.join(CERTS, f"node{node}.crt"),
        "ssl_keyfile":  os.path.join(CERTS, f"node{node}.key"),
        "ssl_ca_certs": os.path.join(CERTS, "ca.crt"),
        "ssl_cert_reqs": "required",
    }
