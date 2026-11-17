# SPDX-License-Identifier: Apache-2.0
<#
Signs Windows code with WiCi AI's certificate through SSL.com eSigner, the one way a release signs.

Every PE image (exe, dll, pyd, winmd, ...) and PowerShell script under -Path that has no signature
of its own gets WiCi AI's; one that carries its vendor's signature keeps it. A file whose unsigned
bytes were signed before is taken from the signed-file store, where each signed file is named by
the SHA-256 of its unsigned bytes, instead of being signed again: a release spends signings only on
the files that changed. Afterwards every code file must carry a valid, timestamped signature, its
vendor's or WiCi AI's from the expected certificate. -VerifyOnly checks that alone and signs nothing.

Environment:
  ES_USERNAME, ES_PASSWORD, ES_TOTP_SECRET  the eSigner account and its automated-signing secret
  ES_CREDENTIAL_ID            the signing credential (may be left out when the account has one)
  ES_ENVIRONMENT              PROD (the default), or TEST for SSL.com's public sandbox
  WINDOWS_SIGNING_THUMBPRINT  SHA-1 thumbprint of the certificate WiCi AI's signatures come from
  APP_VERSIONING_TOKEN        stores newly signed files for later releases (without it, none are stored)
#>
param(
  [Parameter(Mandatory = $true)][string[]]$Path,
  [switch]$VerifyOnly
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($env:ES_ENVIRONMENT -and $env:ES_ENVIRONMENT -cnotin @('PROD', 'TEST')) { throw 'ES_ENVIRONMENT is PROD or TEST.' }
$test = $env:ES_ENVIRONMENT -ceq 'TEST'
$thumbprint = [string]$env:WINDOWS_SIGNING_THUMBPRINT
if ($thumbprint -notmatch '^[0-9A-Fa-f]{40}$') { throw "WINDOWS_SIGNING_THUMBPRINT must be the signing certificate's SHA-1 thumbprint." }
$thumbprint = $thumbprint.ToUpperInvariant()
# The store is a namespace of WiCi AI's release service; its CDN never gives a name other bytes.
$store = $(if ($test) { 'windows-signed-test' } else { 'windows-signed' })
$service = "https://appapi.wici.ai/versioning/$store"
$cdn = "https://cdn.wici.ai/apps/$store"
$agent = 'windows-signing/1 (+https://github.com/wici-ai/inferssd)'
# CodeSignTool 1.3.2, run on a current Java runtime: the one its Windows download carries (11.0.2,
# from 2019) no longer trusts the roots of SSL.com's TLS certificates.
$toolUrl = 'https://github.com/SSLcom/CodeSignTool/releases/download/v1.3.2/CodeSignTool-v1.3.2.zip'
$toolSha256 = 'f14b1e1ef14bfa1fd00279c363aab0debbf5dcfba0e4bcdce5d22bb771de0e3a'
$temporary = $(if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() })

# A code file: a PE image, by its headers whatever its name, or a PowerShell script. Signed says
# whether it carries an embedded signature (a catalog never counts: users' machines lack it).
function Get-CodeFile([IO.FileInfo]$File) {
  $extension = $File.Extension.ToLowerInvariant()
  if ($extension -in @('.ps1', '.psm1', '.psd1', '.ps1xml')) {
    $signed = [IO.File]::ReadAllText($File.FullName).Contains('SIG # Begin signature block')
    return [pscustomobject]@{Path = $File.FullName; Script = $true; Signed = $signed}
  }
  if ($File.Length -lt 512) { return $null }
  $reader = [IO.BinaryReader]::new([IO.File]::OpenRead($File.FullName))
  try {
    $stream = $reader.BaseStream
    if ($reader.ReadUInt16() -ne 0x5A4D) { return $null }
    $stream.Position = 0x3C
    $header = [long]$reader.ReadUInt32()
    if ($header + 0x100 -gt $stream.Length) { return $null }
    $stream.Position = $header
    if ($reader.ReadUInt32() -ne 0x4550) { return $null }
    $stream.Position = $header + 24
    $magic = $reader.ReadUInt16()
    if ($magic -eq 0x20B) { $directories = $header + 24 + 112 }
    elseif ($magic -eq 0x10B) { $directories = $header + 24 + 96 }
    else { return $null }
    $stream.Position = $directories - 4
    $signed = $false
    if ($reader.ReadUInt32() -gt 4) {
      # The certificate table, data directory 4: its size is nonzero when the image is signed.
      $stream.Position = $directories + 4 * 8 + 4
      $signed = $reader.ReadUInt32() -ne 0
    }
    return [pscustomobject]@{Path = $File.FullName; Script = $false; Signed = $signed}
  } finally { $reader.Dispose() }
}

function Test-Ours([string]$File) {
  $signature = Get-AuthenticodeSignature -LiteralPath $File
  # The sandbox's certificate chains to a root Windows does not trust.
  $trusted = $signature.Status -eq 'Valid' -or ($test -and [string]$signature.Status -in @('UnknownError', 'NotTrusted'))
  return [bool]($trusted -and [string]$signature.SignatureType -eq 'Authenticode' -and $signature.SignerCertificate -and
    $signature.SignerCertificate.Thumbprint -eq $thumbprint -and $signature.TimeStamperCertificate)
}

# Called only for files whose headers carry a signature: Windows may report one it also finds in
# a system catalog (DirectML.dll, an OS component too) as a catalog signature, valid either way.
function Test-Vendor([string]$File) {
  return (Get-AuthenticodeSignature -LiteralPath $File).Status -eq 'Valid'
}

# MSVC command-line quoting, so any password reaches the Java runtime intact.
function Join-CommandLine([string[]]$Arguments) {
  return ($Arguments | ForEach-Object {
    if ($_ -and $_ -notmatch '[\s"]') { $_ }
    else { '"' + ($_ -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"' }
  }) -join ' '
}

# GitHub's Windows images carry several Java runtimes; elsewhere JAVA_HOME names one, 11 or later.
function Get-Java {
  foreach ($javaHome in @($env:JAVA_HOME_17_X64, $env:JAVA_HOME_21_X64, $env:JAVA_HOME)) {
    if ($javaHome -and (Test-Path -LiteralPath (Join-Path $javaHome 'bin\java.exe') -PathType Leaf)) { return Join-Path $javaHome 'bin\java.exe' }
  }
  throw 'Signing needs a Java runtime, 11 or later: set JAVA_HOME.'
}

function Install-CodeSignTool {
  $root = Join-Path $temporary 'CodeSignTool-1.3.2'
  if (!(Test-Path -LiteralPath (Join-Path $root 'jar\code_sign_tool-1.3.2.jar') -PathType Leaf)) {
    $archive = Join-Path $temporary 'CodeSignTool-1.3.2.zip'
    & curl.exe -s -f -L --retry 3 -o $archive $toolUrl
    if ($LASTEXITCODE -ne 0) { throw 'CodeSignTool could not be downloaded.' }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $toolSha256) { throw 'CodeSignTool checksum mismatch' }
    if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Recurse -Force }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($archive, $root)
    Remove-Item -LiteralPath $archive
  }
  $endpoints = $(if ($test) {
    # SSL.com's public sandbox, from its eSigner demo credentials guide.
    @('CLIENT_ID=qOUeZCCzSqgA93acB3LYq6lBNjgZdiOxQc-KayC3UMw', 'OAUTH2_ENDPOINT=https://oauth-sandbox.ssl.com/oauth2/token',
      'CSC_API_ENDPOINT=https://cs-try.ssl.com', 'TSA_URL=http://ts.ssl.com')
  } else {
    @('CLIENT_ID=kaXTRACNijSWsFdRKg_KAfD3fqrBlzMbWs6TwWHwAn8', 'OAUTH2_ENDPOINT=https://login.ssl.com/oauth2/token',
      'CSC_API_ENDPOINT=https://cs.ssl.com', 'TSA_URL=http://ts.ssl.com')
  })
  [IO.File]::WriteAllLines((Join-Path $root 'conf\code_sign_tool.properties'), [string[]]$endpoints)
  return $root
}

# Signs every file in $Directory in place of its unsigned bytes, by one batch of at most 100 files.
# The tool exits 0 whatever happens, so success is judged by its message and then by each file.
function Invoke-BatchSign([string]$Directory, [string]$Output) {
  foreach ($name in 'ES_USERNAME', 'ES_PASSWORD', 'ES_TOTP_SECRET') {
    if (![Environment]::GetEnvironmentVariable($name)) { throw "$name is not set: signing needs the eSigner credential." }
  }
  $tool = Install-CodeSignTool
  $arguments = @('-jar', (Join-Path $tool 'jar\code_sign_tool-1.3.2.jar'), 'batch_sign',
    "-username=$env:ES_USERNAME", "-password=$env:ES_PASSWORD", "-totp_secret=$env:ES_TOTP_SECRET",
    "-input_dir_path=$Directory", "-output_dir_path=$Output")
  if ($env:ES_CREDENTIAL_ID) { $arguments += "-credential_id=$env:ES_CREDENTIAL_ID" }
  $start = [Diagnostics.ProcessStartInfo]::new((Get-Java))
  $start.Arguments = Join-CommandLine $arguments
  $start.WorkingDirectory = $tool
  $start.UseShellExecute = $false
  $start.RedirectStandardInput = $true
  $start.RedirectStandardOutput = $true
  $start.RedirectStandardError = $true
  [void]$start.EnvironmentVariables.Remove('APP_VERSIONING_TOKEN')
  $process = [Diagnostics.Process]::Start($start)
  # No prompt can wait for an answer.
  $process.StandardInput.Close()
  $errors = $process.StandardError.ReadToEndAsync()
  $text = $process.StandardOutput.ReadToEnd() + $errors.Result
  $process.WaitForExit()
  if (!$text.Contains('Batch sign command executed successfully')) { throw "eSigner did not sign the batch: $($text.Trim())" }
}

# The store only saves signings: when it cannot be reached, the file is signed, or not stored, anew.
# The CDN keeps a 404 for an hour, so the lookup carries this run's own query and asks the store
# itself: a file stored since an earlier lookup is found at once.
$lookup = [Guid]::NewGuid().ToString('N')
function Receive-Stored([string]$Name, [string]$Destination) {
  $status = & curl.exe -s -L --retry 3 -A $agent -o $Destination -w '%{http_code}' "$cdn/${Name}?lookup=$lookup"
  if ($LASTEXITCODE -eq 0 -and $status -eq '200') { return $true }
  if (Test-Path -LiteralPath $Destination) { Remove-Item -LiteralPath $Destination }
  if ($LASTEXITCODE -ne 0 -or $status -ne '404') { Write-Warning "The signed-file store did not answer for $Name ($status); signing it." }
  return $false
}

function Send-Stored([string]$Name, [string]$File) {
  try {
    $request = @{name = $Name; length = (Get-Item -LiteralPath $File).Length
      sha256 = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant()} | ConvertTo-Json -Compress
    $upload = Invoke-RestMethod -Method Post -Uri "$service/upload" -UserAgent $agent -ContentType 'application/json' -Body $request `
      -Headers @{Authorization = "Bearer $env:APP_VERSIONING_TOKEN"}
    # Another run may have stored its own signature of the same bytes first; either one serves.
    if ($upload.status -ne 'ready' -or $upload.replacing) { return }
    $headers = @(foreach ($header in $upload.headers.PSObject.Properties) { '-H'; "$($header.Name): $($header.Value)" })
    & curl.exe -s -f --retry 3 -A $agent -X PUT --upload-file $File @headers $upload.upload_url
    if ($LASTEXITCODE -ne 0) { throw "the upload failed ($LASTEXITCODE)" }
  } catch { Write-Warning "The signed-file store did not keep $Name; a later release signs it again: $_" }
}

$files = @(foreach ($item in $Path) {
  $target = Get-Item -LiteralPath $item
  $candidates = $(if ($target.PSIsContainer) { Get-ChildItem -LiteralPath $target.FullName -Recurse -File -Force } else { $target })
  foreach ($file in $candidates) { $code = Get-CodeFile $file; if ($code) { $code } }
})
$vendor = 0
$ours = 0
foreach ($file in $files | Where-Object Signed) {
  if (Test-Ours $file.Path) { $ours++; continue }
  if (!(Test-Vendor $file.Path)) { throw "A signature does not verify: $($file.Path) ($((Get-AuthenticodeSignature -LiteralPath $file.Path).StatusMessage))" }
  $vendor++
}
$unsigned = @($files | Where-Object { !$_.Signed })
if ($VerifyOnly -and $unsigned.Count) { throw "Unsigned code: $(($unsigned | Select-Object -First 20 -ExpandProperty Path) -join ', ')" }

$reused = 0
$signed = 0
$signings = 0
if ($unsigned.Count) {
  # The same unsigned bytes anywhere get the same signed bytes, signed once.
  $byName = [ordered]@{}
  foreach ($file in $unsigned) {
    $name = (Get-FileHash -LiteralPath $file.Path -Algorithm SHA256).Hash.ToLowerInvariant() + [IO.Path]::GetExtension($file.Path).ToLowerInvariant()
    if (!$byName.Contains($name)) { $byName[$name] = [pscustomobject]@{Name = $name; Script = $file.Script; Paths = [Collections.Generic.List[string]]::new()} }
    $byName[$name].Paths.Add($file.Path)
  }
  $work = Join-Path $temporary ('signing-' + [Guid]::NewGuid().ToString('N').Substring(0, 12))
  [IO.Directory]::CreateDirectory((Join-Path $work 'store')) | Out-Null
  try {
    $missing = [Collections.Generic.List[object]]::new()
    foreach ($entry in $byName.Values) {
      $copy = Join-Path $work "store\$($entry.Name)"
      if (Receive-Stored $entry.Name $copy) {
        if (!(Test-Ours $copy)) { throw "The stored signed copy $($entry.Name) does not carry the expected signature." }
        foreach ($target in $entry.Paths) { Copy-Item -LiteralPath $copy -Destination $target -Force }
        $reused += $entry.Paths.Count
      } else { $missing.Add($entry) }
    }
    for ($first = 0; $first -lt $missing.Count; $first += 100) {
      $batch = @($missing | Select-Object -Skip $first -First 100)
      $in = Join-Path $work "in-$first"
      $out = Join-Path $work "out-$first"
      [IO.Directory]::CreateDirectory($in) | Out-Null
      [IO.Directory]::CreateDirectory($out) | Out-Null
      # The tool picks its signer by file name, so each goes by a name it signs: scripts as .ps1
      # (.ps1xml has its own form), images as .exe or .dll; the signature does not depend on the name.
      for ($index = 0; $index -lt $batch.Count; $index++) {
        $original = [IO.Path]::GetExtension($batch[$index].Paths[0]).ToLowerInvariant()
        $extension = $(if ($batch[$index].Script) { $(if ($original -eq '.ps1xml') { '.ps1xml' } else { '.ps1' }) }
          elseif ($original -eq '.exe') { '.exe' } else { '.dll' })
        $batch[$index] | Add-Member -NotePropertyName File -NotePropertyValue "$index$extension" -Force
        Copy-Item -LiteralPath $batch[$index].Paths[0] -Destination (Join-Path $in $batch[$index].File)
      }
      Invoke-BatchSign $in $out
      foreach ($entry in $batch) {
        $result = Join-Path $out $entry.File
        if (!(Test-Path -LiteralPath $result -PathType Leaf)) { throw "eSigner returned no signed copy of $($entry.Paths[0])" }
        $copy = Join-Path $work "store\$($entry.Name)"
        Move-Item -LiteralPath $result -Destination $copy
        if (!(Test-Ours $copy)) { throw "The signed copy of $($entry.Paths[0]) does not verify." }
        foreach ($target in $entry.Paths) { Copy-Item -LiteralPath $copy -Destination $target -Force }
        if ($env:APP_VERSIONING_TOKEN) { Send-Stored $entry.Name $copy }
        $signed += $entry.Paths.Count
        $signings++
      }
    }
  } finally { if (Test-Path -LiteralPath $work) { Remove-Item -LiteralPath $work -Recurse -Force } }
}

# The bytes that ship: every code file verifies.
foreach ($file in $files) {
  if (!(Test-Ours $file.Path) -and !(Test-Vendor $file.Path)) { throw "Not signed as a release requires: $($file.Path)" }
}
Write-Host ("$($files.Count) code files verify: $vendor keep their vendor's signature, $ours already carry WiCi AI's, " +
  "$reused take WiCi AI's from the signed-file store, $signed were signed now, by $signings signings$(if ($test) { ' (sandbox certificate)' }).")
