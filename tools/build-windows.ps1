# tools/build-windows.ps1
#
# Windows x64 only. Builds the vendored llama.cpp as shared libraries and
# generates the lib<name>.dll.a import libraries FreeBASIC needs on Windows.
# Safe to re-run: CMake builds incrementally and the import libraries are
# regenerated on every run.
#
# Prerequisites:
#   - git, with the vendor/llama.cpp submodule initialized
#   - CMake, and Visual Studio 2022 with the Desktop development with C++
#     workload; run this from a VS 2022 Developer PowerShell
#   - FreeBASIC for Windows, with fbc on PATH
#   - CUDA toolkit, optional, only when -Cuda is used
# -Vulkan needs the LunarG Vulkan SDK installed, per the fork's build docs.

param(
    # Build the CUDA backend; needs the CUDA toolkit.
    [switch]$Cuda,
    # Build the Vulkan backend; needs the LunarG Vulkan SDK.
    [switch]$Vulkan
)

function Fail {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Red
    exit 1
}

function Find-Tool {
    param(
        [string]$Name,
        [string[]]$Directories
    )
    $exeName = $Name + '.exe'
    foreach ($dir in $Directories) {
        if (-not $dir) { continue }
        $candidate = Join-Path $dir $exeName
        if (Test-Path $candidate) {
            return @{ Path = $candidate; Source = $dir }
        }
    }
    $found = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue
    if ($found) {
        return @{ Path = $found.Source; Source = 'PATH' }
    }
    return $null
}

# Work from the repo root no matter where the script was invoked.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
Write-Host "Repo root: $RepoRoot"

# Preconditions. Each failure prints one line and exits nonzero.
$LlamaDir = Join-Path $RepoRoot 'vendor\llama.cpp'
if (-not (Test-Path (Join-Path $LlamaDir 'CMakeLists.txt'))) {
    Fail "llama.cpp submodule not populated at $LlamaDir; run: git submodule update --init"
}
if (-not (Get-Command cmake -CommandType Application -ErrorAction SilentlyContinue)) {
    Fail "cmake not found on PATH; install Visual Studio 2022 with the Desktop development with C++ workload and use a VS 2022 Developer PowerShell."
}
$FbcCommand = Get-Command fbc -CommandType Application -ErrorAction SilentlyContinue
if (-not $FbcCommand) {
    Fail "fbc not found on PATH; install FreeBASIC for Windows and add its folder to PATH."
}

# Find the GCC tools before the build so a missing tool fails fast.
# Search order: FreeBASIC tool folder derived from fbc, C:\msys64\mingw64\bin, PATH.
$FbcDir = Split-Path -Parent $FbcCommand.Source
$FbToolDirs = @($FbcDir, (Join-Path (Split-Path -Parent $FbcDir) 'bin\win64'))
$ToolDirs = $FbToolDirs + @('C:\msys64\mingw64\bin')
$dllTool = Find-Tool -Name 'dlltool' -Directories $ToolDirs
if (-not $dllTool) {
    Fail "dlltool.exe not found in the FreeBASIC tool folder, C:\msys64\mingw64\bin or PATH. It usually comes with the FreeBASIC Windows distribution; install FreeBASIC for Windows or MSYS2 mingw-w64 binutils."
}
$gendef = Find-Tool -Name 'gendef' -Directories $ToolDirs
if (-not $gendef) {
    Fail "gendef.exe not found in the FreeBASIC tool folder, C:\msys64\mingw64\bin or PATH. It usually comes from MSYS2 mingw-w64 tools; install MSYS2 and mingw-w64-x86_64-tools-git, or put gendef.exe on PATH."
}
Write-Host ("dlltool: " + $dllTool['Path'] + " from " + $dllTool['Source'])
Write-Host ("gendef: " + $gendef['Path'] + " from " + $gendef['Source'])

# Dedicated tree: it cannot collide with the Linux build trees.
$BuildDir = Join-Path $LlamaDir 'build-windows'
$configureArgs = @('-B', $BuildDir, '-DBUILD_SHARED_LIBS=ON', '-A', 'x64')
if ($Cuda) { $configureArgs += '-DGGML_CUDA=ON' } else { $configureArgs += '-DGGML_CUDA=OFF' }
if ($Vulkan) { $configureArgs += '-DGGML_VULKAN=ON' } else { $configureArgs += '-DGGML_VULKAN=OFF' }
Write-Host ("Configuring: cmake " + ($configureArgs -join ' '))
& cmake @configureArgs
if ($LASTEXITCODE -ne 0) { Fail "CMake configure failed with exit code $LASTEXITCODE." }
Write-Host "Building: cmake --build $BuildDir --config Release -j"
& cmake --build $BuildDir --config Release -j
if ($LASTEXITCODE -ne 0) { Fail "CMake build failed with exit code $LASTEXITCODE." }

# Multi-config generators put the DLLs in bin\Release; single-config ones in bin.
$BinDir = Join-Path $BuildDir 'bin'
$ReleaseBinDir = Join-Path $BinDir 'Release'
$DllDir = $null
foreach ($candidate in @($ReleaseBinDir, $BinDir)) {
    if (Test-Path (Join-Path $candidate 'llama.dll')) {
        $DllDir = $candidate
        break
    }
}
if (-not $DllDir) {
    Fail "No llama.dll found in $ReleaseBinDir or $BinDir; the shared build did not produce DLLs."
}
Write-Host "DLL output folder: $DllDir"

# Import libraries go beside the DLLs, so one -p path serves fbc.
$dllNames = @('llama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll')
if ($Cuda) { $dllNames += 'ggml-cuda.dll' }
if ($Vulkan) { $dllNames += 'ggml-vulkan.dll' }

$dllToolExe = $dllTool['Path']
$gendefExe = $gendef['Path']
$createdLibs = @()
Push-Location $DllDir
foreach ($dllName in $dllNames) {
    $dllPath = Join-Path $DllDir $dllName
    if (-not (Test-Path $dllPath)) {
        Fail "Expected DLL not found: $dllPath"
    }
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($dllName)
    $defPath = Join-Path $DllDir ($baseName + '.def')
    $libName = 'lib' + $baseName + '.dll.a'
    $libPath = Join-Path $DllDir $libName

    # gendef writes its .def into the current directory, named after the DLL.
    Write-Host "gendef $dllName"
    & $gendefExe $dllName
    if ($LASTEXITCODE -ne 0) { Fail "gendef failed for $dllName with exit code $LASTEXITCODE." }
    if (-not (Test-Path $defPath)) { Fail "gendef did not create $defPath" }

    Write-Host "dlltool $libName"
    & $dllToolExe -d $defPath -D $dllName -l $libPath
    if ($LASTEXITCODE -ne 0) { Fail "dlltool failed for $dllName with exit code $LASTEXITCODE." }
    if (-not (Test-Path $libPath)) { Fail "dlltool did not create $libPath" }
    $createdLibs += $libPath
}
Pop-Location

Write-Host ""
Write-Host "DLL output folder: $DllDir"
Write-Host "Import libraries created:"
foreach ($libPath in $createdLibs) {
    Write-Host "  $libPath"
}
Write-Host ""
Write-Host "Example compile, from examples/ with the Windows fbc:"
Write-Host "fbc hello_llama.bas -p $DllDir -l llama -l ggml -l ggml-base -l ggml-cpu"
Write-Host "At run time, put the DLLs beside the compiled .exe or add $DllDir to PATH."
