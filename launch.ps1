# Launch the Simple HF Trainer desktop application using the app virtual environment
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$pythonExe = Join-Path $scriptDir 'environments\app\Scripts\python.exe'
& $pythonExe "$scriptDir\main.py"