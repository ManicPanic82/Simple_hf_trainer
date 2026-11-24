# Launch the Simple HF Trainer desktop application using the app virtual environment
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$pythonExe = Join-Path $scriptDir 'environments\app\Scripts\python.exe'

if (-not (Test-Path $pythonExe)) {
    Write-Error "Python executable not found at $pythonExe"
    exit 1
}

Write-Host "Starting Simple HF Trainer..."
& $pythonExe "$scriptDir\main.py"