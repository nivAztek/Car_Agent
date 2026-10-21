<#
.SYNOPSIS
  autochat task runner (Windows PowerShell 5.1 compatible).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/dev.ps1 setup
  powershell -ExecutionPolicy Bypass -File scripts/dev.ps1 smoke
  powershell -ExecutionPolicy Bypass -File scripts/dev.ps1 ingest --stage embed --force
  (extra args pass straight through; do NOT use a bare `--`, PowerShell -File rejects it)
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("help", "up", "down", "logs", "setup", "emulator", "emulator-stop", "emulator-wipe", "smoke", "smoke-cosmos", "smoke-foundry",
                 "smoke-mlflow", "mlflow", "probe", "ingest", "reset-db", "db-stats", "retrieve", "api", "ui", "chat",
                 "test", "eval", "lint", "graph")]
    [string]$Task = "help",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Continue"   # PS 5.1: native stderr must not abort the script
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
$skills = Join-Path $repo ".github\skills"
$env:PYTHONPATH = Join-Path $repo "src"
$env:PYTHONUTF8 = "1"

function Invoke-Py([string[]]$PyArgs) {
    if (-not (Test-Path $py)) { Write-Error "Missing .venv - run: scripts/dev.ps1 setup"; exit 1 }
    & $py @PyArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Task) {
    "help" {
        @"
Docker (whole stack, no Python needed on the host):
  up             docker compose up -d --build  -> UI :8501, API :8000, MLflow :5000, Data Explorer :1234
  down           docker compose down (add -v to also wipe the Cosmos + MLflow volumes)
  logs           docker compose logs -f (optionally a service: api, ui, mlflow, cosmos)

Local venv development:
  setup          create .venv (uv) and install project + dev deps; create .env from example
  emulator       start Docker Desktop (if needed) + Cosmos vNext emulator, wait until ready
  emulator-stop  stop emulator          | emulator-wipe  stop + delete persisted data
  smoke          smoke-cosmos + smoke-foundry + smoke-mlflow
  smoke-mlflow   MLflow tracing (LangGraph + retriever span + feedback) + genai.evaluate with Foundry judges
  mlflow         local MLflow UI/tracking server on http://localhost:5000 (store: mlflow.db)
  probe          download/cache the 8 articles and report structure
  ingest         run ingestion pipeline (extra args passed through, e.g. ingest --stage chunk --force)
  reset-db       drop + recreate cars/chunks containers (needed after EMBED_DIMENSIONS change)
  db-stats       container doc counts + vector policy
  retrieve       retrieval check: filter cases + hit@5 vector/bm25/hybrid (or: retrieve -q "query")
  api            uvicorn autochat.api.main:app --reload (port 8000)
  ui             streamlit run src/autochat/ui/app.py (port 8501)
  chat           terminal chat (in-process graph); add --api http://localhost:8000 for API mode
  test           pytest + eval scoring selftest
  eval           golden-set eval logged to MLflow (add --judge for Foundry LLM judges, --no-mlflow to skip)
  lint           ruff check + format check
  graph          print LangGraph as Mermaid
"@
    }
    "setup" {
        python -m uv --version 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { python -m pip install --user -q uv }
        if (-not (Test-Path $py)) { python -m uv venv .venv --python 3.13 }
        python -m uv pip install --python $py -e ".[dev]"
        if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Write-Host "Created .env - fill FOUNDRY_* values." }
    }
    "up"   { docker compose up -d --build @Rest }
    "down" { docker compose down @Rest }
    "logs" { docker compose logs -f --tail 100 @Rest }
    "emulator"      { powershell -ExecutionPolicy Bypass -File "$skills\cosmos-emulator\scripts\start_emulator.ps1" @Rest }
    "emulator-stop" { powershell -ExecutionPolicy Bypass -File "$skills\cosmos-emulator\scripts\start_emulator.ps1" -Stop }
    "emulator-wipe" { powershell -ExecutionPolicy Bypass -File "$skills\cosmos-emulator\scripts\start_emulator.ps1" -Wipe }
    "smoke-cosmos"  { Invoke-Py @("$skills\cosmos-emulator\scripts\cosmos_smoke.py") }
    "smoke-foundry" { Invoke-Py @("$skills\foundry-models\scripts\foundry_smoke.py") }
    "smoke-mlflow"  { Invoke-Py @("$skills\mlflow-observability\scripts\mlflow_smoke.py") }
    "mlflow" {
        $env:MLFLOW_DISABLE_TELEMETRY = "true"
        Invoke-Py (@("-m", "mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db",
                     "--host", "127.0.0.1", "--port", "5000") + $Rest)
    }
    "smoke" {
        Invoke-Py @("$skills\cosmos-emulator\scripts\cosmos_smoke.py")
        Invoke-Py @("$skills\foundry-models\scripts\foundry_smoke.py")
        Invoke-Py @("$skills\mlflow-observability\scripts\mlflow_smoke.py")
    }
    "probe"    { Invoke-Py (@("$skills\autochat-ingestion\scripts\probe_articles.py") + $Rest) }
    "ingest"   { Invoke-Py (@("-m", "autochat.ingestion.pipeline") + $Rest) }
    "reset-db" { Invoke-Py @("$skills\cosmos-emulator\scripts\cosmos_admin.py", "reset") }
    "db-stats" { Invoke-Py @("$skills\cosmos-emulator\scripts\cosmos_admin.py", "stats") }
    "retrieve" { Invoke-Py (@("$skills\autochat-agent\scripts\retrieval_check.py") + $Rest) }
    "api"      { Invoke-Py (@("-m", "uvicorn", "autochat.api.main:app", "--reload", "--port", "8000", "--app-dir", "src") + $Rest) }
    "ui"       { Invoke-Py (@("-m", "streamlit", "run", "src/autochat/ui/app.py", "--server.port", "8501") + $Rest) }
    "chat"     { Invoke-Py (@("$skills\autochat-agent\scripts\chat_cli.py") + $Rest) }
    "graph"    { Invoke-Py @("$skills\autochat-agent\scripts\graph_mermaid.py") }
    "test" {
        Invoke-Py @("$skills\autochat-eval\scripts\selftest_scoring.py")
        if (Test-Path "tests") { Invoke-Py (@("-m", "pytest", "-q") + $Rest) } else { Write-Host "(no tests/ yet)" }
    }
    "eval" { Invoke-Py (@("$skills\autochat-eval\scripts\run_eval.py") + $Rest) }
    "lint" {
        $paths = @("src", "tests", ".github/skills") | Where-Object { Test-Path $_ }
        Invoke-Py (@("-m", "ruff", "check") + $paths)
        Invoke-Py (@("-m", "ruff", "format", "--check") + (@("src", "tests") | Where-Object { Test-Path $_ }))
    }
}
