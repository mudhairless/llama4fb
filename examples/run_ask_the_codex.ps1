# examples/run_ask_the_codex.ps1
#
# Windows PowerShell 5.1 compatible. Launches ask_the_codex.exe with the
# llama.cpp DLL folder on PATH for this process. Safe to re-run.
#
# Usage:
#   .\run_ask_the_codex.ps1 [llama.cpp dir] [build/bin dir] [extra args...]
#   $env:LLAMA_CPP_DIR = 'C:\path\to\llama.cpp'; .\run_ask_the_codex.ps1
#   $env:LLAMA_BUILD_DIR = 'C:\path\to\bin'; .\run_ask_the_codex.ps1

function Fail {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Red
    exit 1
}

$LlamaCpp = $env:LLAMA_CPP_DIR
$BuildDir = $env:LLAMA_BUILD_DIR
$AppArgs = @()
if ($args.Count -gt 0) { $LlamaCpp = $args[0] }
if ($args.Count -gt 1) { $BuildDir = $args[1] }
if ($args.Count -gt 2) { $AppArgs = $args[2..($args.Count - 1)] }
if ($BuildDir -and -not (Test-Path $BuildDir -PathType Container)) {
    $AppArgs = @($BuildDir) + $AppArgs
    $BuildDir = ''
}

$ProjectDir = Split-Path -Parent $PSScriptRoot
$UsingSubmodule = $false

if (-not $LlamaCpp) {
    $LlamaCpp = Join-Path $ProjectDir 'vendor\llama.cpp'
    $UsingSubmodule = $true
    if (-not (Test-Path (Join-Path $LlamaCpp 'CMakeLists.txt'))) {
        Fail "llama.cpp submodule not populated at $LlamaCpp; run: git submodule update --init"
    }
}
if (-not (Test-Path $LlamaCpp -PathType Container)) {
    Fail "llama.cpp directory not found: $LlamaCpp"
}
$LlamaCpp = (Resolve-Path $LlamaCpp).Path

if (-not $BuildDir) {
    $candidates = @(
        (Join-Path $LlamaCpp 'build-windows\bin\Release'),
        (Join-Path $LlamaCpp 'build-windows\bin'),
        (Join-Path $LlamaCpp 'build-cuda\bin\Release'),
        (Join-Path $LlamaCpp 'build-cuda\bin'),
        (Join-Path $LlamaCpp 'build-vulkan\bin\Release'),
        (Join-Path $LlamaCpp 'build-vulkan\bin'),
        (Join-Path $LlamaCpp 'build\bin\Release'),
        (Join-Path $LlamaCpp 'build\bin')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate 'llama.dll')) {
            $BuildDir = $candidate
            break
        }
    }
    if (-not $BuildDir) {
        if ($UsingSubmodule) {
            Fail "No Windows shared build found under $LlamaCpp. From the repo root, run: tools\build-windows.ps1"
        }
        Fail "No folder with llama.dll found under $LlamaCpp; pass the DLL folder as the second argument."
    }
}
if (-not (Test-Path $BuildDir -PathType Container)) {
    Fail "DLL folder not found: $BuildDir"
}
$BuildDir = (Resolve-Path $BuildDir).Path

$required = @('llama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll')
$missing = @()
foreach ($dll in $required) {
    if (-not (Test-Path (Join-Path $BuildDir $dll))) { $missing += $dll }
}
if ($missing.Count -gt 0) {
    Write-Host ("MISSING: " + ($missing -join ', ') + " (looked in $BuildDir)") -ForegroundColor Red
    Fail "Build llama.cpp first. From the repo root: tools\build-windows.ps1"
}

$Exe = Join-Path $PSScriptRoot 'ask_the_codex.exe'
if (-not (Test-Path $Exe)) {
    Write-Host "Binary missing, building..."
    if ($BuildDir) {
        & (Join-Path $PSScriptRoot 'build_ask_the_codex.ps1') $LlamaCpp $BuildDir
    } else {
        & (Join-Path $PSScriptRoot 'build_ask_the_codex.ps1') $LlamaCpp
    }
    if (-not (Test-Path $Exe)) { Fail "Build failed; no $Exe was produced." }
}

$env:PATH = "$BuildDir;$env:PATH"
Set-Location $PSScriptRoot
& $Exe @AppArgs
exit $LASTEXITCODE
