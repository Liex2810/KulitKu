param([string]$Repo = "Liex2810/kulitku", [switch]$Public)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Check-Command { if ($LASTEXITCODE -ne 0) { throw "Command failed: $LASTEXITCODE" } }
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw "Install GitHub CLI from https://cli.github.com" }
gh auth status; Check-Command
if (-not (Test-Path .git)) { git init -b main; Check-Command }
git add .; Check-Command
git diff --cached --quiet
if ($LASTEXITCODE -eq 1) { git commit -m "Initialize KulitKu monorepo"; Check-Command }
$visibility = if ($Public) { "--public" } else { "--private" }
gh repo create $Repo $visibility --source . --remote origin --push; Check-Command
