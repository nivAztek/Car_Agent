<#
.SYNOPSIS
  Start the Cosmos DB vNext emulator (docker compose) and wait until it's ready.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .github/skills/cosmos-emulator/scripts/start_emulator.ps1
  powershell -ExecutionPolicy Bypass -File .github/skills/cosmos-emulator/scripts/start_emulator.ps1 -Stop
  powershell -ExecutionPolicy Bypass -File .github/skills/cosmos-emulator/scripts/start_emulator.ps1 -Wipe   # delete persisted data
#>
param(
    [switch]$Stop,
    [switch]$Wipe,
    [int]$TimeoutSec = 180
)

# PS 5.1 turns native stderr into terminating errors under "Stop"; docker writes progress to stderr.
# Use "Continue" and check $LASTEXITCODE explicitly instead.
$ErrorActionPreference = "Continue"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Push-Location $repo
try {
    function Test-DockerEngine {
        cmd /c "docker info --format {{.ServerVersion}} >nul 2>&1"
        return ($LASTEXITCODE -eq 0)
    }

    if (-not (Test-DockerEngine)) {
        $dd = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
        if (-not (Test-Path $dd)) { throw "Docker engine not reachable and Docker Desktop not found at $dd" }
        Write-Host "Starting Docker Desktop..."
        Start-Process -FilePath $dd | Out-Null
        $deadline = (Get-Date).AddSeconds($TimeoutSec)
        while (-not (Test-DockerEngine)) {
            if ((Get-Date) -gt $deadline) { throw "Docker engine did not start within $TimeoutSec s" }
            Start-Sleep -Seconds 3
        }
        Write-Host "Docker engine is up."
    }

    if ($Stop -or $Wipe) {
        if ($Wipe) {
            cmd /c "docker compose down --volumes 2>&1"   # drops the autochat-cosmos-data volume
            Write-Host "Wiped emulator data volume"
        } else {
            cmd /c "docker compose down 2>&1"
        }
        if (Test-Path ".cosmos-data") {   # legacy bind-mount folder (pre named volume)
            Remove-Item -Recurse -Force ".cosmos-data"
        }
        if ($Stop) { return }
    }

    cmd /c "docker compose up -d cosmos 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }

    Write-Host "Waiting for emulator readiness on http://localhost:8080/ready ..."
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ($true) {
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:8080/ready" -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -eq 200) { break }
        } catch { }
        if ((Get-Date) -gt $deadline) {
            cmd /c "docker compose logs --tail 40 cosmos 2>&1"
            throw "Emulator not ready within $TimeoutSec s"
        }
        Start-Sleep -Seconds 3
    }
    Write-Host "Cosmos emulator ready: endpoint http://localhost:8081  | Data Explorer http://localhost:1234"
}
finally {
    Pop-Location
}
