# IMMUNEX Node Startup Script
# Run as: .\start_node.ps1 -node acer
# or:     .\start_node.ps1 -node lenovo
# etc.

param([string]$node)

$base = $PSScriptRoot
$venv = "$base\.venv\Scripts\activate"

function Start-Layer {
    param([string]$module, [int]$port, [string]$workdir)
    Start-Process powershell -ArgumentList "-NoExit", "-Command", 
        "cd '$workdir'; & '$venv'; python -m uvicorn $module --host 0.0.0.0 --port $port --no-access-log"
}

switch ($node) {
    "acer" {
        Write-Host "Starting Acer (10.0.0.2): L1 + L2"
        Start-Layer "layer1_detection.server:app" 8001 $base
        Start-Layer "layer2_correlation.server:app" 8002 $base
    }
    "lenovo" {
        Write-Host "Starting Lenovo (10.0.0.3): L1 + L3"
        Start-Layer "layer1_detection.server:app" 8001 $base
        $l3path = "$base\Layer3 Response Engine\Response_engine"
        Start-Layer "main:app" 8003 $l3path
    }
    "victus" {
        Write-Host "Starting Victus (10.0.0.4): L1 + L4"
        Start-Layer "layer1_detection.server:app" 8001 $base
        Start-Layer "layer4_immunity.server:app" 8004 $base
    }
    "pavilion" {
        Write-Host "Starting Pavilion (10.0.0.5): L1 + L5"
        Start-Layer "layer1_detection.server:app" 8001 $base
        $l5path = "$base\Layer5_Threat Memory"
        Start-Layer "server:app" 8005 $l5path
    }
    default {
        Write-Host "Usage: .\start_node.ps1 -node [acer|lenovo|victus|pavilion]"
    }
}
