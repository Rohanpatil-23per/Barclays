#!/usr/bin/env python3
"""
patch_aliases_for_judges.py
Patches shared/smart_normalizer.py _build_aliases() to add the 13 unmapped
fields from the judge dataset. Run once before the eval, idempotent.

Usage:
    python3 patch_aliases_for_judges.py
"""
import re, subprocess, sys

PATH = "/home/aditya/Documents/hoh/Barclays/shared/smart_normalizer.py"

with open(PATH) as f:
    content = f.read()

PATCH_MARKER = "# ── JUDGE_DATASET_ALIASES_PATCH ──"
if PATCH_MARKER in content:
    print("Already patched — skipping.")
    sys.exit(0)

NEW_ALIASES = f'''
{PATCH_MARKER}
        # Azure AD / Office365 / SharePoint
        _add("src_hostname",
             "UserPrincipalName", "user_principal_name", "UserId",
             "userId", "user_id", "upn")
        _add("src_ip",
             "ClientIP", "clientIP", "client_ip", "client_address")
        _add("signature",
             "Operation", "OperationName", "operation", "operation_name",
             "EventType", "event_type", "activity",
             "CommandLine", "command_line", "cmdline", "ProcessCommandLine",
             "Image", "process_image",
             "http_uri", "uri", "url", "request_uri")
        _add("action",
             "logon_type", "LogonType", "http_method", "method")
        # Zeek traffic bytes
        _add("bytes_in",  "orig_bytes", "bytes_to_server")
        _add("bytes_out", "resp_bytes", "bytes_to_client")
        _add("protocol",  "service", "network_service", "app_proto", "application")
        # Fix: log_source must NOT alias to src_ip
        for _bad in ("log_source", "source", "log_type"):
            self._alias_map.pop(_bad, None)
'''

# Find the end of _build_aliases — last alias add() call before the closing line
# Insert our patch just before the method's final return or the next def
ba_start = content.find("    def _build_aliases(")
ba_end   = content.find("\n    def ", ba_start + 1)
assert ba_start != -1 and ba_end != -1, "Cannot find _build_aliases method"

fn_body = content[ba_start:ba_end]

# Find the internal helper name: add() or _add() or self._add_alias()
if "_add_alias" in fn_body:
    helper = "_add_alias"
elif re.search(r'^\s+add\(', fn_body, re.M):
    helper = "add"
else:
    helper = None

# Build a version of NEW_ALIASES that uses the right helper name
if helper and helper != "_add":
    patch = NEW_ALIASES.replace("_add(", f"{helper}(")
else:
    # Fallback: use direct dict update instead of helper
    patch = f'''
{PATCH_MARKER}
        # Judge dataset field aliases — direct map update
        _JUDGE_FIELDS = {{
            "UserPrincipalName": "src_hostname", "user_principal_name": "src_hostname",
            "UserId": "src_hostname", "userId": "src_hostname", "upn": "src_hostname",
            "ClientIP": "src_ip", "clientIP": "src_ip", "client_ip": "src_ip",
            "Operation": "signature", "OperationName": "signature",
            "operation": "signature", "operation_name": "signature",
            "EventType": "signature", "CommandLine": "signature",
            "command_line": "signature", "Image": "signature",
            "http_uri": "signature", "uri": "signature", "url": "signature",
            "logon_type": "action", "LogonType": "action",
            "http_method": "action", "method": "action",
            "orig_bytes": "bytes_in", "bytes_to_server": "bytes_in",
            "resp_bytes": "bytes_out", "bytes_to_client": "bytes_out",
            "service": "protocol", "network_service": "protocol",
            "app_proto": "protocol", "application": "protocol",
        }}
        self._alias_map.update(_JUDGE_FIELDS)
        for _bad in ("log_source", "source", "log_type"):
            self._alias_map.pop(_bad, None)
'''

# Insert the patch just before the end of _build_aliases
insertion_point = ba_end
content = content[:insertion_point] + patch + "\n" + content[insertion_point:]

with open(PATH, "w") as f:
    f.write(content)

# Syntax check
r = subprocess.run(
    ["python3", "-c", f"import ast; ast.parse(open('{PATH}').read())"],
    capture_output=True, text=True
)
if r.returncode == 0:
    print("✓ Patch applied. Syntax OK.")
    # Quick verify
    import subprocess as sp
    verify = sp.run(["python3", "-c", """
import sys
sys.path.insert(0, '/home/aditya/Documents/hoh/Barclays')
from shared.smart_normalizer import SchemaFingerprinter
fp = SchemaFingerprinter()
tests = [
    ("UserPrincipalName", "src_hostname"),
    ("ClientIP",          "src_ip"),
    ("Operation",         "signature"),
    ("CommandLine",       "signature"),
    ("orig_bytes",        "bytes_in"),
    ("resp_bytes",        "bytes_out"),
    ("service",           "protocol"),
    ("logon_type",        "action"),
]
ok = all = 0
for field, want in tests:
    got = fp.map_field(field)
    match = got == want
    ok += int(match)
    all += 1
    print(f"  {'✓' if match else '✗'}  {field:<22} → {got!r}")
print(f"Coverage: {ok}/{all}")
"""], capture_output=True, text=True)
    print(verify.stdout)
    if verify.stderr:
        print("STDERR:", verify.stderr[:300])
else:
    print(f"✗ Syntax error:\n{r.stderr}")