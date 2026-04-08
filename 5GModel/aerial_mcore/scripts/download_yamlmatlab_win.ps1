# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

$ErrorActionPreference = "Stop"

$scriptDir   = $PSScriptRoot
$projectRoot = Split-Path -Parent $scriptDir
$modelRoot   = Split-Path -Parent $projectRoot
Write-Host "$($MyInvocation.MyCommand.Path) starting..."

$nrMatlabPath   = Join-Path $modelRoot "nr_matlab"
$yamlmatlabPath = Join-Path $nrMatlabPath "yamlmatlab"

# Retry helper: runs a script block up to $MaxAttempts times.
# Handles both thrown exceptions and non-zero $LASTEXITCODE from native tools.
function Invoke-WithRetry {
    param(
        [scriptblock]$ScriptBlock,
        [int]$MaxAttempts = 3,
        [int]$DelaySeconds = 5
    )
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Write-Host "Attempt $attempt of ${MaxAttempts}..."
        $failed = $false
        $err    = $null
        try {
            & $ScriptBlock
        } catch {
            $failed = $true
            $err    = $_
        }
        if (-not $failed -and $LASTEXITCODE -ne 0) {
            $failed = $true
            $err    = "exited with code $LASTEXITCODE"
        }
        if (-not $failed) { return }
        if ($attempt -lt $MaxAttempts) {
            Write-Host "Failed: $err. Retrying in ${DelaySeconds}s..."
            Start-Sleep -Seconds $DelaySeconds
        } else {
            throw "All $MaxAttempts attempts failed. Last error: $err"
        }
    }
}

# Remove existing yamlmatlab directory
if (Test-Path $yamlmatlabPath) {
    Remove-Item -Recurse -Force $yamlmatlabPath
}

# Clone repo
Push-Location $nrMatlabPath
try {
    Invoke-WithRetry { git clone https://github.com/jerelbn/yamlmatlab.git }

    Push-Location yamlmatlab
    try {
        git checkout e011be81a77d2bbcb5a88c244789f2211aadeb59
        if ($LASTEXITCODE -ne 0) { throw "git checkout failed" }

        # Replace snakeyaml jar
        Push-Location external
        try {
            Remove-Item -Force snakeyaml-1.9.jar
            Invoke-WithRetry {
                Invoke-WebRequest -UseBasicParsing `
                    -Uri "https://repo1.maven.org/maven2/org/yaml/snakeyaml/2.5/snakeyaml-2.5.jar" `
                    -OutFile "snakeyaml-2.5.jar"
            }

            # Verify SHA256 checksum (read expected hash from the shared .sha256 reference file)
            $expectedSha256 = ((Get-Content (Join-Path $scriptDir "snakeyaml-2.5.jar.sha256")) -split '\s+')[0].ToLower()
            $actualSha256   = (Get-FileHash -Algorithm SHA256 "snakeyaml-2.5.jar").Hash.ToLower()
            if ($actualSha256 -ne $expectedSha256) {
                throw "SHA256 checksum mismatch for snakeyaml-2.5.jar. Expected: $expectedSha256, Got: $actualSha256"
            }
            Write-Host "snakeyaml-2.5.jar checksum OK"
        } finally {
            Pop-Location
        }
    } finally {
        Pop-Location
    }
} finally {
    Pop-Location
}

# Patch all .m files: replace snakeyaml-1.9.jar -> snakeyaml-2.5.jar
# Use raw byte I/O so existing line endings (LF or CRLF) are preserved exactly.
Get-ChildItem -Path $yamlmatlabPath -Filter "*.m" -Recurse | ForEach-Object {
    $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
    $text  = [System.Text.Encoding]::UTF8.GetString($bytes)
    if ($text -match "snakeyaml-1\.9\.jar") {
        $newBytes = [System.Text.Encoding]::UTF8.GetBytes(
            $text -replace "snakeyaml-1\.9\.jar", "snakeyaml-2.5.jar"
        )
        [System.IO.File]::WriteAllBytes($_.FullName, $newBytes)
    }
}

Write-Host "yamlmatlab setup complete."
