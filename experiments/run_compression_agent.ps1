$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project '.venv\Scripts\python.exe'
$scratch = Join-Path $project '.venv\tmp'
New-Item -ItemType Directory -Path $scratch -Force | Out-Null

$env:TEMP = $scratch
$env:TMP = $scratch
$env:HF_HUB_OFFLINE = '1'
$env:HF_HOME = Join-Path $scratch 'hf-awq'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:PYTHONUTF8 = '1'
$env:PYTORCH_ALLOC_CONF = 'backend:native'
$env:TRITON_CACHE_DIR = Join-Path $scratch 'triton-awq'
$env:TORCH_EXTENSIONS_DIR = Join-Path $scratch 'torch-extensions-awq'

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Visual Studio C++ Build Tools are required for the AWQ Triton backend.' }
    $visualStudio = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $visualStudio) { throw 'No Visual Studio C++ toolchain was found.' }
    Import-Module (Join-Path $visualStudio 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $visualStudio -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
}
$env:CC = (Get-Command cl.exe).Source

Push-Location -LiteralPath $project
try {
    & $python -u -m experiments.compression_agent @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
