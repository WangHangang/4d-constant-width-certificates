param(
    [int]$RadiusExp = 24,
    [int]$Workers = 16,
    [string]$SharedCache = "cache/AMO_shared_ray_cache_r24.json"
)

$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

New-Item -ItemType Directory -Force -Path "certificates" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "cache" | Out-Null

function Run-Checked {
    param(
        [string[]]$Args,
        [string]$Log
    )
    & python @Args 2>&1 | Tee-Object -FilePath $Log
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE. See $Log"
    }
}

foreach ($level in 4,5,6) {
    $cert = "certificates/AMO_k$level.json"
    $ck = "certificates/AMO_k$level.checkpoint.json"
    $glog = "logs/AMO_k${level}_generate.log"
    $vlog = "logs/AMO_k${level}_verify.log"

    Write-Host "$(Get-Date -Format o) [AMO] V1 level $level, workers=$Workers"
    Run-Checked -Args @(
        "-u","generate_amo_certificate.py",
        "--level","$level",
        "--radius-exp","$RadiusExp",
        "--workers","$Workers",
        "--shared-cache",$SharedCache,
        "--out",$cert,
        "--checkpoint",$ck
    ) -Log $glog

    Run-Checked -Args @("verify_certificate.py",$cert) -Log $vlog

    & python "check_threshold.py" $cert
    if ($LASTEXITCODE -eq 0) {
        Write-Host "$(Get-Date -Format o) [AMO] COMPLETE at V1 level $level"
        exit 0
    }
}

Write-Host "$(Get-Date -Format o) [AMO] switching to V2 adaptive, workers=$Workers"
$acert = "certificates/AMO_adaptive.json"
$ack = "certificates/AMO_adaptive.checkpoint.json"

Run-Checked -Args @(
    "-u","generate_amo_adaptive_certificate.py",
    "--base-level","5",
    "--fraction","0.20",
    "--max-rounds","6",
    "--radius-exp","$RadiusExp",
    "--workers","$Workers",
    "--shared-cache",$SharedCache,
    "--out",$acert,
    "--checkpoint",$ack
) -Log "logs/AMO_adaptive_generate.log"

Run-Checked -Args @("verify_certificate.py",$acert) -Log "logs/AMO_adaptive_verify.log"

& python "check_threshold.py" $acert
if ($LASTEXITCODE -ne 0) {
    Write-Host "AMO adaptive certificate is valid but still below the separator."
    exit 2
}
Write-Host "$(Get-Date -Format o) [AMO] COMPLETE with V2 adaptive"
exit 0
