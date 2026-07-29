$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Requirements = Join-Path $RepoRoot "requirements_windows.txt"

if (-not (Test-Path $Requirements -PathType Leaf)) {
    throw "Requirements file not found: $Requirements"
}

& python -c "import sys; assert sys.version_info[:2] == (3, 10), f'Python 3.10 is required, found {sys.version}'"
if ($LASTEXITCODE -ne 0) {
    throw "Activate the documented Python 3.10 Conda environment first."
}

& python -c "import dlib"
if ($LASTEXITCODE -ne 0) {
    throw "dlib is missing. Install it with Conda as documented in the Windows README instructions."
}

Write-Host "Installing Windows dependencies from $Requirements"
& python -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) {
    throw "pip failed with exit code $LASTEXITCODE"
}

& python -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "pip check found an inconsistent environment"
}

Write-Host "Windows dependencies installed successfully."
