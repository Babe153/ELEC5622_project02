param(
    [ValidateSet('check','train','test')][string]$Action = 'check',
    [string]$Experiment = 'baseline_30epochs',
    [string]$Config = '',
    [string]$RunDir = '',
    [string]$PythonExe = ''
)
$ErrorActionPreference = 'Stop'
if (-not $PythonExe) { $PythonExe = $env:PROJECT02_PYTHON }
if (-not $PythonExe) {
    $projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    $existingPython = Join-Path $env:USERPROFILE 'Documents/Codex/2026-11-03/w-x/work/benchmark-env/Scripts/python.exe'
    if (Test-Path -LiteralPath $projectPython) { $PythonExe = $projectPython }
    elseif (Test-Path -LiteralPath $existingPython) { $PythonExe = $existingPython }
    else { $PythonExe = (Get-Command python -ErrorAction Stop).Source }
}
if (-not (Test-Path -LiteralPath $PythonExe)) { throw 'Python executable not found. Supply -PythonExe with its full path.' }
if (-not $Config) { $Config = Join-Path $PSScriptRoot 'config.json' }
$arguments = @('-u', (Join-Path $PSScriptRoot 'train.py'), $Action, '--config', $Config)
if ($Action -eq 'train') { $arguments += @('--experiment', $Experiment) }
if ($Action -eq 'test') {
    if (-not $RunDir) {
        $settings = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
        $resultsPath = $settings.results_dir
        if (-not [IO.Path]::IsPathRooted($resultsPath)) { $resultsPath = Join-Path $PSScriptRoot $resultsPath }
        $RunDir = Join-Path $resultsPath $Experiment
    }
    $arguments += @('--run-dir', $RunDir)
}
& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
