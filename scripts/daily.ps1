# Daily job: Yahoo snapshot, DoltHub increment, smile build, screen. Output goes to logs\.
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\daily_$(Get-Date -Format yyyy-MM).log"
"=== $(Get-Date -Format s) ===" | Out-File $log -Append -Encoding utf8
cmd /c ".venv\Scripts\python.exe -m spotvol daily >> $log 2>&1"
