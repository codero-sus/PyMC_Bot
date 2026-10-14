# Build the PyMC Playtime Recorder for every Minecraft version in versions.json (Windows).
#
#   .\build-all.ps1                    # all versions -> dist\pymcplaytime-<mod>+<mc>.jar
#   .\build-all.ps1 1.21.1 26.3        # only these
#
# Needs JDK 25 on PATH (or JAVA_HOME). Gradle and Minecraft are downloaded automatically.
# A version that fails does not stop the others; the summary at the end lists it.
param([string[]]$Versions)

$ErrorActionPreference = 'Continue'
Set-Location -Path $PSScriptRoot

if (-not $Versions -or $Versions.Count -eq 0) {
    $table = Get-Content -Raw -Path 'versions.json' | ConvertFrom-Json
    $Versions = $table.versions | ForEach-Object { $_.minecraft }
}

New-Item -ItemType Directory -Force -Path 'dist' | Out-Null
$built = @()
$failed = @()
foreach ($mc in $Versions) {
    Write-Host ""
    Write-Host "==> Minecraft $mc"
    & .\gradlew.bat build "-Pmc=$mc" --console=plain
    if ($LASTEXITCODE -eq 0) {
        $jar = Get-ChildItem -Path 'build\libs' -Filter "pymcplaytime-*+$mc.jar" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($jar) {
            Copy-Item -Path $jar.FullName -Destination 'dist' -Force
            $built += $jar.Name
        } else {
            $failed += "$mc (no jar produced)"
        }
    } else {
        $failed += $mc
    }
}

Write-Host ""
Write-Host "Built $($built.Count) jar(s) into $(Join-Path (Get-Location) 'dist'):"
$built | ForEach-Object { Write-Host "  $_" }
if ($failed.Count -gt 0) {
    Write-Host "Failed: $($failed -join ', ')"
    exit 1
}
