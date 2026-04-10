$FILE = "../requirements_windows.txt"

Get-Content $FILE | ForEach-Object {
    $pkg = $_.Trim()
    if (-not [string]::IsNullOrWhiteSpace($pkg) -and -not $pkg.StartsWith("#")) {
        Write-Output "Installing $pkg..."
        pip install $pkg
        if ($LASTEXITCODE -ne 0) {
            Write-Output "Failed to install: $pkg"
        }
    }
}
