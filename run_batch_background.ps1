$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path (Join-Path $root "..\\.venv311") "python.exe"
$script = Join-Path $root "run_batch_resume.py"
$logDir = Join-Path $root "outputs"
$stdout = Join-Path $logDir "batch_stdout.log"
$stderr = Join-Path $logDir "batch_stderr.log"

New-Item -ItemType Directory -Force $logDir | Out-Null

Start-Process `
  -FilePath $python `
  -ArgumentList @($script, "--task-file", (Join-Path $root "batch_tasks.json"), "--output-dir", $logDir) `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden

Write-Output "Background batch started."
Write-Output "stdout: $stdout"
Write-Output "stderr: $stderr"
