"""
Rigorous normalizer test suite — 60+ assertions across 8 sections.
Run: python3 tests/test_normalizer.py
"""
import sys, time, json
sys.path.append('/home/aditya/Documents/hoh/Barclays')
from shared.smart_normalizer import SmartNormalizer, SchemaFingerprinter

PASS = FAIL = 0
def chk(label, got, exp, partial=False):
    global PASS, FAIL
    ok = (str(exp).lower() in str(got).lower()) if partial else (str(got) == str(exp))
    if ok: PASS += 1; print(f"  ✓  {label}")
    else:  FAIL += 1; print(f"  ✗  {label}\n       got={got!r}  want={exp!r}")

def n1(n, raw):
    evs, _ = n.normalize_batch([raw]); return evs[0] if evs else None

def src(e):     return e['src_endpoint']['ip']
def dst(e):     return e['dst_endpoint']['ip']
def sport(e):   return e['src_endpoint']['port']
def dport(e):   return e['dst_endpoint']['port']
def proto(e):   return e['connection_info']['protocol_name']
def sev(e):     return e['severity']
def find(e):    return e['finding']['title']
def bin_(e):    return e['traffic']['bytes_in']
def bout(e):    return e['traffic']['bytes_out']

NOW_MS = int(time.time() * 1000)

# ═══════════════════════════════════════════════════════════
print("\n═"*35)
print("SECTION 1 — Format Detection & Field Extraction")
print("═"*35)
n = SmartNormalizer()

e = n1(n, {"src":"45.33.1.1","dest":"10.0.0.5","transport":"tcp","dest_port":443,
           "src_port":54321,"severity":"high","signature":"BruteForce",
           "_time":1774717200,"bytes_in":5000,"bytes_out":200,"action":"blocked"})
chk("1.1 Splunk CIM src_ip",    src(e),   "45.33.1.1")
chk("1.1 Splunk CIM dst_ip",    dst(e),   "10.0.0.5")
chk("1.1 Splunk CIM protocol",  proto(e), "TCP")
chk("1.1 Splunk CIM dst_port",  dport(e), 443)
chk("1.1 Splunk CIM severity",  sev(e),   "High")
chk("1.1 Splunk CIM finding",   find(e),  "BruteForce")
chk("1.1 Splunk CIM bytes_in",  bin_(e),  5000)
chk("1.1 Splunk CIM bytes_out", bout(e),  200)

e = n1(n, '{"timestamp":"2026-03-28T17:00:08Z","src_ip":"45.33.32.156","src_port":54321,"dest_ip":"10.0.0.20","dest_port":80,"proto":"TCP","alert":{"signature":"ET SCAN Nmap","severity":1},"flow":{"bytes_toserver":328,"bytes_toclient":1240}}')
chk("1.2 Suricata src_ip",      src(e),   "45.33.32.156")
chk("1.2 Suricata dst_ip",      dst(e),   "10.0.0.20")
chk("1.2 Suricata finding",     find(e),  "ET SCAN Nmap")
chk("1.2 Suricata bytes_in",    bin_(e),  328)
chk("1.2 Suricata bytes_out",   bout(e),  1240)

e = n1(n, {"@timestamp":"2026-03-28T17:00:15Z","event.severity":3,"source.ip":"203.78.90.1",
           "source.port":41234,"destination.ip":"10.0.0.5","destination.port":443,
           "network.transport":"tcp","rule.name":"SQLInjection"})
chk("1.3 ECS src_ip",           src(e),   "203.78.90.1")
chk("1.3 ECS dst_ip",           dst(e),   "10.0.0.5")
chk("1.3 ECS protocol",         proto(e), "TCP")
chk("1.3 ECS finding",          find(e),  "SQLInjection")
chk("1.3 ECS severity (3=Med)", sev(e),   "Medium")

e = n1(n, {"IN_BYTES":150000,"PROTOCOL":6,"L4_SRC_PORT":80,"IPV4_SRC_ADDR":"10.0.0.80",
           "L4_DST_PORT":54000,"IPV4_DST_ADDR":"185.220.101.99","OUT_BYTES":200,
           "LAST_SWITCHED":1774717200})
chk("1.4 NetFlow src_ip",       src(e),   "10.0.0.80")
chk("1.4 NetFlow dst_ip",       dst(e),   "185.220.101.99")
chk("1.4 NetFlow src_port",     sport(e), 80)
chk("1.4 NetFlow dst_port",     dport(e), 54000)
chk("1.4 NetFlow protocol",     proto(e), "TCP")
chk("1.4 NetFlow bytes_in",     bin_(e),  150000)

e = n1(n, "2 123456789012 eni-abc123 10.0.0.1 203.0.113.5 49152 22 6 15 7800 1774717200 1774717260 REJECT OK")
chk("1.5 AWS VPC src_ip",       src(e),   "10.0.0.1")
chk("1.5 AWS VPC dst_ip",       dst(e),   "203.0.113.5")
chk("1.5 AWS VPC src_port",     sport(e), 49152)
chk("1.5 AWS VPC dst_port",     dport(e), 22)
chk("1.5 AWS VPC protocol",     proto(e), "TCP")
chk("1.5 AWS VPC finding",      find(e),  "FirewallReject")

e = n1(n, "1774717210.123456\tCHO123\t10.0.0.5\t54321\t185.220.101.5\t4444\ttcp\t-\t3600\t512000\t1024000\t12\t24\tSF")
chk("1.6 Zeek src_ip",          src(e),   "10.0.0.5")
chk("1.6 Zeek dst_ip",          dst(e),   "185.220.101.5")
chk("1.6 Zeek bytes_in",        bin_(e),  512000)

e = n1(n, "<14>Dec 12 11:43:52 pafw LEEF:1.0|Palo Alto Networks|PAN-OS|8.1|allow|cat=TRAFFIC|ReceiveTime=2026/03/28 17:00:00|src=192.168.1.3|dst=10.0.2.21|srcPort=39936|dstPort=443|proto=tcp|action=allow|totalBytes=553|srcBytes=479|dstBytes=74|")
chk("1.7 LEEF src_ip",          src(e),   "192.168.1.3")
chk("1.7 LEEF dst_ip",          dst(e),   "10.0.2.21")
chk("1.7 LEEF src_port",        sport(e), 39936)
chk("1.7 LEEF dst_port",        dport(e), 443)
chk("1.7 LEEF protocol",        proto(e), "TCP")

e = n1(n, "1,2026/03/28 17:00:01,007200001165,THREAT,vulnerability,2560,2026/03/28 17:00:01,192.168.1.105,10.0.0.3,0.0.0.0,0.0.0.0,rule1,domain\\jsmith,vsys1,trust,untrust,eth1/1,eth1/2,threat-log,1234,1,63221,443,0,0,0x402000,tcp,alert,CVE-2021-44228,Log4Shell,critical,client-to-server,12345")
chk("1.8 PAN-OS THREAT src_ip", src(e),   "192.168.1.105")
chk("1.8 PAN-OS THREAT dst_ip", dst(e),   "10.0.0.3")
chk("1.8 PAN-OS THREAT sport",  sport(e), 63221)
chk("1.8 PAN-OS THREAT dport",  dport(e), 443)
chk("1.8 PAN-OS THREAT proto",  proto(e), "TCP")
chk("1.8 PAN-OS THREAT sev",    sev(e),   "Critical")

e = n1(n, "1,2026/03/28 17:01:00,007200001165,TRAFFIC,end,2560,2026/03/28 17:01:00,10.0.0.5,8.8.8.8,0.0.0.0,0.0.0.0,allow-out,jsmith,,,dns,vsys1,trust,untrust,eth1/1,eth1/2,log-fwd,0,12345,1,54123,53,0,0,0x0,udp,allow,1200,500,700,8,2026/03/28 17:00:55,5,any,0,123456,0x0,US,US,0,5,3,aged-out")
chk("1.9 PAN-OS TRAFFIC src_ip",src(e),   "10.0.0.5")
chk("1.9 PAN-OS TRAFFIC dst_ip",dst(e),   "8.8.8.8")
chk("1.9 PAN-OS TRAFFIC sport", sport(e), 54123)
chk("1.9 PAN-OS TRAFFIC dport", dport(e), 53)
chk("1.9 PAN-OS TRAFFIC proto", proto(e), "UDP")

e = n1(n, '{"EventID":4625,"TimeCreated":"2026-03-28T17:00:05.123Z","IpAddress":"185.220.101.5","IpPort":"52341","TargetUserName":"administrator","LogonType":3}')
chk("1.10 Win4625 src_ip",      src(e),   "185.220.101.5")
chk("1.10 Win4625 finding",     find(e),  "WindowsLogonFailure")

e = n1(n, {"flow_initiator_v4_addr":"172.16.50.1","flow_responder_v4_addr":"10.0.0.99",
           "l4_xport_src":55123,"l4_xport_dst":3389,"ip_nexthdr":"TCP",
           "obs_point_severity_rank":"HIGH","threat_classification_label":"LateralMovement",
           "octets_initiated":204800,"octets_responded":512})
chk("1.11 Proprietary src_ip",  src(e),   "172.16.50.1")
chk("1.11 Proprietary dst_ip",  dst(e),   "10.0.0.99")
chk("1.11 Proprietary sev",     sev(e),   "High")
chk("1.11 Proprietary finding", find(e),  "LateralMovement")
chk("1.11 Proprietary bytes_in",bin_(e),  204800)

# ═══════════════════════════════════════════════════════════
print("\n═"*35)
print("SECTION 2 — Cross-Vendor Field Collisions")
print("═"*35)
n2 = SmartNormalizer()

e = n2.normalize_batch([{"src":"1.1.1.1","source.ip":"2.2.2.2","dest":"10.0.0.1",
                          "destination.ip":"10.0.0.2","transport":"tcp","network.transport":"udp"}])[0]
e = e[0] if e else None
chk("2.1 src vs source.ip: one wins", src(e) in ("1.1.1.1","2.2.2.2"), True)
chk("2.1 protocol resolves",          proto(e) in ("TCP","UDP"), True)

e = n1(n2, {"IPV4_SRC_ADDR":"3.3.3.3","IPV4_DST_ADDR":"4.4.4.4","PROTOCOL":17,
             "transport":"tcp","L4_SRC_PORT":5000,"L4_DST_PORT":53})
chk("2.2 PROTOCOL=17(UDP) vs transport=tcp: one wins", proto(e) in ("UDP","TCP"), True)
chk("2.2 src from IPV4_SRC_ADDR", src(e), "3.3.3.3")

e = n1(n2, {"src_ip":"6.6.6.6","dst_ip":"7.7.7.7","bytes_in":1000,"IN_BYTES":9999})
chk("2.3 bytes_in collision: non-zero", bin_(e) > 0, True)
chk("2.3 bytes_in collision: one of two values", bin_(e) in (1000,9999), True)

# ═══════════════════════════════════════════════════════════
print("\n═"*35)
print("SECTION 3 — Adversarial / Edge Cases")
print("═"*35)
n3 = SmartNormalizer()

e = n1(n3, {"src_ip":None,"dst_ip":None,"severity":None})
chk("3.1 None values: no crash", e is not None, True)
chk("3.1 None src_ip → 0.0.0.0", src(e), "0.0.0.0")

e = n1(n3, {"src_ip":"","dst_ip":"","severity":""})
chk("3.2 Empty strings: no crash", e is not None, True)

e = n1(n3, {"src_ip":"8.8.8.8","dst_ip":"1.1.1.1","src_port":999999,"dst_port":-1})
chk("3.3 Port overflow: no crash", e is not None, True)

e = n1(n3, {"src_ip":"9.9.9.9","dst_ip":"10.0.0.1","severity":0.95})
chk("3.4 Float sev 0.95 → Critical", sev(e), "Critical")
e = n1(n3, {"src_ip":"9.9.9.8","dst_ip":"10.0.0.1","severity":0.45})
chk("3.4 Float sev 0.45 → Medium",   sev(e), "Medium")
e = n1(n3, {"src_ip":"9.9.9.7","dst_ip":"10.0.0.1","severity":0.15})
chk("3.4 Float sev 0.15 → Low",      sev(e), "Low")

e = n1(n3, {"src_ip":"11.0.0.1","dst_ip":"10.0.0.1","timestamp":"not-a-date"})
chk("3.5 Garbage timestamp: time > 0", e['time'] > 0, True)
e = n1(n3, {"src_ip":"11.0.0.3","dst_ip":"10.0.0.1","timestamp":"2026-03-28T17:00:00Z"})
chk("3.5 ISO 8601 timestamp",          e['time'], 1774717200000)
e = n1(n3, {"src_ip":"11.0.0.4","dst_ip":"10.0.0.1","timestamp":1774717200})
chk("3.5 Epoch seconds → ms",          e['time'], 1774717200000)
e = n1(n3, {"src_ip":"11.0.0.5","dst_ip":"10.0.0.1","timestamp":1774717200000})
chk("3.5 Epoch ms passthrough",        e['time'], 1774717200000)

e = n1(n3, "CORRUPTED\x00\xff LOG DATA src=1.2.3.4 garbage=@#$% dst=broken")
chk("3.6 Corrupted binary: no crash",    e is not None, True)
chk("3.6 Corrupted binary: src_ip",      src(e), "1.2.3.4")

e = n1(n3, {"src_ip":"2001:db8::1","dst_ip":"2001:db8::2","severity":"high"})
chk("3.7 IPv6 src",  src(e), "2001:db8::1")
chk("3.7 IPv6 dst",  dst(e), "2001:db8::2")

e = n1(n3, {"dst_ip":"10.0.0.1","severity":"critical","signature":"Ransomware"})
chk("3.8 Missing src_ip → 0.0.0.0",    src(e), "0.0.0.0")
chk("3.8 Finding still extracted",      find(e), "Ransomware")

# ═══════════════════════════════════════════════════════════
print("\n═"*35)
print("SECTION 4 — Bloom Filter Deduplication")
print("═"*35)
n4 = SmartNormalizer()

LOG = {"src_ip":"20.0.0.1","dst_ip":"10.0.0.1","protocol":"TCP",
       "severity":"high","signature":"PortScan","timestamp":1774717200000}
evs, stats = n4.normalize_batch([LOG, LOG])
chk("4.1 Exact dup: 1 event",       len(evs), 1)
chk("4.1 Exact dup: stats deduped", stats['deduped_bloom'], 1)

evs, _ = n4.normalize_batch([
    {"src_ip":"20.0.0.3","dst_ip":"10.0.0.1","signature":"PortScan","timestamp":1774717300000},
    {"src_ip":"20.0.0.4","dst_ip":"10.0.0.1","signature":"PortScan","timestamp":1774717300000},
])
chk("4.2 Different src: both kept", len(evs), 2)

evs, stats = n4.normalize_batch([{"src_ip":"20.0.0.6","dst_ip":"10.0.0.1",
                                    "signature":"Noise","timestamp":1774717500000}] * 1000)
chk("4.3 1000 identical → 1",      len(evs), 1)
chk("4.3 999 deduped",             stats['deduped_bloom'], 999)

# ═══════════════════════════════════════════════════════════
print("\n═"*35)
print("SECTION 5 — Throughput Benchmark")
print("═"*35)
n5 = SmartNormalizer()
bench = []
for i in range(2500):
    bench.append({"src":f"{i//256}.{i%256}.0.1","dest":"10.0.0.1","transport":"tcp",
                   "dest_port":443,"severity":"high","signature":"BruteForce","_time":NOW_MS+i*10})
for i in range(2500):
    bench.append(f'{{"src_ip":"{i//256}.{i%256}.1.1","dest_ip":"10.0.0.2","proto":"TCP","dest_port":80,"severity":"medium","signature":"PortScan","timestamp":{NOW_MS+25000+i*10}}}')
for i in range(2500):
    bench.append({"IPV4_SRC_ADDR":f"{i//256}.{i%256}.2.1","IPV4_DST_ADDR":"10.0.0.3",
                   "PROTOCOL":6,"L4_SRC_PORT":i%65534+1,"L4_DST_PORT":443,
                   "IN_BYTES":i*100,"OUT_BYTES":i*10,"LAST_SWITCHED":NOW_MS//1000+i})
for i in range(2500):
    bench.append({"src":f"{i//256}.{i%256}.3.1","dest":"10.0.0.4","transport":"udp",
                   "dest_port":53,"severity":"low","signature":"DNS","_time":NOW_MS//1000+10000+i})

t0 = time.perf_counter()
_, bstats = n5.normalize_batch(bench)
elapsed = time.perf_counter() - t0
tput = int(10000 / elapsed)
chk("5.1 10k logs: no crash",         bstats['total'], 10000)
chk("5.2 Throughput > 10k/sec",       tput > 10000, True)
chk("5.3 Coverage > 90%",             bstats['avg_mapping_coverage'] > 0.90, True)
print(f"  📊 {tput:,} logs/sec | {bstats['avg_mapping_coverage']*100:.0f}% coverage | {elapsed*1000:.0f}ms")

# ═══════════════════════════════════════════════════════════
total = PASS + FAIL
print(f"\n{'═'*35}")
print(f"RESULTS: {PASS}/{total} passed ({100*PASS//total if total else 0}%)")
if FAIL:
    print(f"         {FAIL} FAILED")
print("═"*35)
