<#
.SYNOPSIS
    1-click PowerShell launcher for Simple HF Trainer.
#>
Set-Location -Path $PSScriptRoot

$python = Join-Path $PSScriptRoot "environments\app\Scripts\python.exe"
if (Test-Path $python) {
    & $python (Join-Path $PSScriptRoot "main.py") @args
} else {
    Write-Error "Application environment not found at environments\app. See docs/INSTALLATION.md."
    Read-Host "Press Enter to exit"
}
