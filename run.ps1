param(
    [ValidateSet('check','train','resume','test')][string]$Action = 'check',
    [string]$Experiment = 'baseline_60epochs',
    [string]$Config = '',
    [string]$RunDir = '',
    [string]$PythonExe = '',
    [int]$Epochs = 0,
    [switch]$UpgradeLoader
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
if ($Action -in @('test', 'resume')) {
    if (-not $RunDir) {
        $settings = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
        $resultsPath = $settings.results_dir
        if (-not [IO.Path]::IsPathRooted($resultsPath)) { $resultsPath = Join-Path $PSScriptRoot $resultsPath }
        $RunDir = Join-Path $resultsPath $Experiment
    }
    $arguments += @('--run-dir', $RunDir)
}
if ($Epochs -ne 0) {
    if ($Action -ne 'resume') { throw '-Epochs is only valid with -Action resume' }
    $arguments += @('--epochs', $Epochs)
}
if ($UpgradeLoader) { $arguments += '--upgrade-loader' }
& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
