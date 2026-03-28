"""
log_templatizer.py
Replaces dynamic values with typed tokens before embedding.
Use: templatize(text) → template string for DBSCAN clustering.
"""
import re
from typing import Tuple

# Order matters — more specific patterns first
_PATTERNS = [
    # Network
    (re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),   '<IPv6>'),
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b'),        '<IPv4>'),
    (re.compile(r'\b([0-9a-fA-F]{2}[:\-]){5}[0-9a-fA-F]{2}\b'),      '<MAC>'),
    # Identifiers
    (re.compile(r'\bCVE-\d{4}-\d{4,7}\b', re.I),                      '<CVE>'),
    (re.compile(r'\b[0-9a-f]{64}\b'),                                  '<SHA256>'),
    (re.compile(r'\b[0-9a-f]{40}\b'),                                  '<SHA1>'),
    (re.compile(r'\b[0-9a-f]{32}\b'),                                  '<MD5>'),
    (re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'), '<UUID>'),
    # Paths
    (re.compile(r'[A-Za-z]:\\(?:[^\s,;"\'<>|]+\\)*[^\s,;"\'<>|]*'),   '<WINPATH>'),
    (re.compile(r'(?<!\w)/(?:etc|usr|var|home|tmp|opt|proc|sys|bin|sbin|lib|dev|run)/[^\s,;"\'<>|]*'), '<UNIXPATH>'),
    # Ports and numbers
    (re.compile(r'(?<=port\s)\d{1,5}\b', re.I),                       '<PORT>'),
    (re.compile(r'(?<=:)\d{1,5}\b'),                                   '<PORT>'),
    # Usernames (word before "from" or after "user"/"for user"/"account")
    (re.compile(r'(?<=\buser\s)[\w.\-@]{2,32}', re.I),                '<USER>'),
    (re.compile(r'(?<=\baccount\s)[\w.\-@]{2,32}', re.I),             '<USER>'),
    (re.compile(r'[\w.\-@]{2,32}(?=\s+from\s+<IPv4>)', re.I),        '<USER>'),
    # Finance / Barclays-specific
    (re.compile(r'\b\d{8,12}\b'),                                      '<ACCTNUM>'),  # account numbers
    (re.compile(r'£[\d,]+(?:\.\d{2})?|\$[\d,]+(?:\.\d{2})?'),        '<AMOUNT>'),
    # Generic large numbers (session IDs, sequence numbers)
    (re.compile(r'\b\d{6,}\b'),                                        '<NUMID>'),
    # Timestamps that slipped through (ISO format)
    (re.compile(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?'), '<TIMESTAMP>'),
]

def templatize(text: str) -> Tuple[str, dict]:
    """
    Replace dynamic values with typed tokens.
    Returns (template, extracted_values) where extracted_values maps
    token positions back to original values for lookup.
    
    Example:
        templatize("Failed login for user jsmith from 192.168.1.45 port 52341")
        → ("Failed login for user <USER> from <IPv4> port <PORT>", {...})
    """
    extracted = {}
    result = str(text)
    for pattern, token in _PATTERNS:
        matches = list(pattern.finditer(result))
        for m in reversed(matches):  # reverse to preserve offsets
            key = f"{token}_{m.start()}"
            extracted[key] = m.group(0)
        result = pattern.sub(token, result)
    return result, extracted


def templatize_event(event: dict) -> str:
    """
    Build a templatized string from a normalized event dict for embedding.
    Combines finding title + key metadata into a single template string.
    """
    parts = []
    finding = event.get("finding", {}).get("title", "")
    if finding:
        parts.append(finding)
    
    # Add protocol context
    proto = event.get("connection_info", {}).get("protocol_name", "")
    dst_port = event.get("dst_endpoint", {}).get("port") or \
               event.get("connection_info", {}).get("dst_port", "")
    if proto and dst_port:
        parts.append(f"proto={proto} port={dst_port}")
    elif proto:
        parts.append(f"proto={proto}")
    
    # Add severity
    sev = event.get("severity", "")
    if sev and sev != "Unknown":
        parts.append(f"severity={sev}")
    
    raw_text = " | ".join(parts)
    template, _ = templatize(raw_text)
    return template


if __name__ == "__main__":
    tests = [
        "Failed login for user jsmith from 192.168.1.45 at port 52341",
        "Failed login for user admin from 10.0.0.8 at port 49122",
        "CVE-2021-44228 exploit attempt from 45.33.99.1 to 10.0.0.5",
        "Ransomware file C:\\Windows\\Temp\\update.exe written by BARCLAYS\\svc_backup",
        "SHA256: a3f5c1e2d4b6789012345678901234567890123456789012345678901234abcd",
        "Account 00124567890 transfer £50,000.00 flagged",
    ]
    print("Templatization tests:\n")
    for t in tests:
        template, _ = templatize(t)
        print(f"  IN:  {t}")
        print(f"  OUT: {template}\n")
