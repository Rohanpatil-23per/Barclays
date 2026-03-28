import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests, json, random

BASE = "http://localhost:8001"

def test_health():
    r = requests.get(f"{BASE}/health")
    print("Health:", r.json())

def test_detect():
    payload = {
        "alert_id":   "test-001",
        "timestamp":  "2026-03-19T21:00:00",
        "source_ip":  "192.168.1.10",
        "dest_ip":    "10.0.0.1",
        "alert_type": "unknown",
        "severity":   0.5,
        "features":   [random.uniform(-1, 2) for _ in range(77)]
    }
    r = requests.post(f"{BASE}/detect", json=payload)
    print("Detect:", json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    test_health()
    test_detect()
