# SPDX-License-Identifier: Apache-2.0
param(
  # Each a directory as package-windows.ps1 leaves it: one package ZIP and its listing.
  [Parameter(Mandatory = $true)][string[]]$Directory,
  # More files to sign in the same pass, such as a release's pinned install.ps1.
  [string[]]$File = @()
)
# Signs the Windows packages a run publishes. Each ZIP is unpacked, all of them are signed by
# sign-windows.ps1 in one pass, so the files the packages share are signed once, and each is packed
# again as package-windows.ps1 packs it, with its listing's size and SHA-256 updated. Runs in Windows
# PowerShell, like package-windows.ps1, so the archive is written the same way.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$utf8 = New-Object Text.UTF8Encoding $false
$work = Join-Path $(if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() }) ('ps-' + [Guid]::NewGuid().ToString('N').Substring(0, 8))
try {
  $index = 0
  $packages = @(foreach ($folder in $Directory) {
    $listings = @(Get-ChildItem -LiteralPath $folder -Filter 'inferssd-*-windows-x64-*.json' -File)
    if ($listings.Count -ne 1) { throw "Expected one Windows package listing in $folder" }
    $listing = Get-Content -LiteralPath $listings[0].FullName -Raw | ConvertFrom-Json
    $archive = Join-Path $folder $listing.file
    if ((Get-Item -LiteralPath $archive).Length -ne $listing.size -or
        (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $listing.sha256) {
      throw "$($listing.file) does not match its listing"
    }
    # A short root: the package's own paths are long.
    $root = Join-Path $work ($index++)
    [IO.Compression.ZipFile]::ExtractToDirectory($archive, $root)
    [pscustomobject]@{Listing = $listings[0].FullName; Data = $listing; Archive = $archive; Root = $root}
  })
  & (Join-Path $PSScriptRoot 'sign-windows.ps1') -Path (@($packages | ForEach-Object Root) + $File)
  foreach ($package in $packages) {
    Remove-Item -LiteralPath $package.Archive
    [IO.Compression.ZipFile]::CreateFromDirectory($package.Root, $package.Archive, [IO.Compression.CompressionLevel]::Optimal, $false)
    $package.Data.size = (Get-Item -LiteralPath $package.Archive).Length
    $package.Data.sha256 = (Get-FileHash -LiteralPath $package.Archive -Algorithm SHA256).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText($package.Listing, ($package.Data | ConvertTo-Json -Depth 5), $utf8)
    Write-Output "Signed $($package.Data.file)"
  }
} finally {
  if (Test-Path -LiteralPath $work) { Remove-Item -LiteralPath $work -Recurse -Force }
}
