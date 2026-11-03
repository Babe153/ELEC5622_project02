param(
    [ValidateSet('check','train','test')][string]$Action = 'check',
    [string]$Experiment = 'baseline_30epochs',
    [string]$Config = '',
    [string]$RunDir = '',
    [string]$PythonExe = ''
)
$ErrorActionPreference = 'Stop'
if (-not $PythonExe) {
    $PythonExe = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw 'Project Python environment not found. Create .venv and install requirements.txt as described in README.md, or supply -PythonExe explicitly.'
}
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
