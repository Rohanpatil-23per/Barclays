# IMMUNEX Node Startup Script with mTLS
param([string]$node)

$base = $PSScriptRoot
$venv = "$base\.venv\Scripts\activate.ps1"
$certs = "$base\certs"

$nodeMap = @{
    "acer"    = 2
    "lenovo"  = 3
    "victus"  = 4
    "pavilion"= 5
}

function Start-Layer {
    param([string]$module, [int]$port, [string]$workdir, [int]$nodeNum)
    $certfile = "$certs\node${nodeNum}.crt"
    $keyfile  = "$certs\node${nodeNum}.key"
    $cafile   = "$certs\ca.crt"
    $cmd = "cd '$workdir'; & '$venv'; " +
           "python -m uvicorn $module " +
           "--host 0.0.0.0 --port $port --no-access-log " +
           "--ssl-certfile '$certfile' " +
           "--ssl-keyfile '$keyfile' " +
           "--ssl-ca-certs '$cafile'"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd
}

if (-not $nodeMap.ContainsKey($node)) {
    Write-Host "Usage: .\start_node_mtls.ps1 -node [acer|lenovo|victus|pavilion]"
    exit
}

$num = $nodeMap[$node]
Write-Host "Starting $node (10.0.0.$num) with mTLS"

switch ($node) {
    "acer" {
        Start-Layer "layer1_detection.server:app" 8001 $base $num
        Start-Layer "layer2_correlation.server:app" 8002 $base $num
    }
    "lenovo" {
        Start-Layer "layer1_detection.server:app" 8001 $base $num
        $l3 = "$base\Layer3 Response Engine\Response_engine"
        Start-Layer "main:app" 8003 $l3 $num
    }
    "victus" {
        Start-Layer "layer1_detection.server:app" 8001 $base $num
        Start-Layer "layer4_immunity.server:app" 8004 $base $num
    }
    "pavilion" {
        Start-Layer "layer1_detection.server:app" 8001 $base $num
        $l5 = "$base\Layer5_Threat Memory"
        Start-Layer "server:app" 8005 $l5 $num
    }
}
