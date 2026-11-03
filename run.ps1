param(
    [ValidateSet('check','train','test')][string]$Action = 'check',
    [string]$Experiment = 'baseline_30epochs',
    [string]$Config = '',
    [string]$RunDir = ''
)
$ErrorActionPreference = 'Stop'
$settingsPath = Join-Path $PSScriptRoot 'environment/local.json'
if (Test-Path -LiteralPath $settingsPath) {
    $localSettings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
    $pythonExe = $localSettings.python
} else {
    $pythonExe = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Python environment missing. See README.md to install dependencies and update environment/local.json.'
}
if (-not $Config) { $Config = Join-Path $PSScriptRoot 'config.json' }
$arguments = @((Join-Path $PSScriptRoot 'train.py'), $Action, '--config', $Config)
if ($Action -eq 'train') { $arguments += @('--experiment', $Experiment) }
if ($Action -eq 'test') {
    if (-not $RunDir) { $RunDir = Join-Path $PSScriptRoot ('results/' + $Experiment) }
    $arguments += @('--run-dir', $RunDir)
}
& $pythonExe @arguments
if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
