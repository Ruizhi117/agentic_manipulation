$ErrorActionPreference = "Stop"

$pythonCommand = Get-Command python -ErrorAction Stop
$assetTarget = (& $pythonCommand.Source -c "from mani_skill import ASSET_DIR; print(ASSET_DIR / 'robots' / 'xlerobot')").Trim()
$urdfPath = Join-Path $assetTarget "xlerobot.urdf"
if (Test-Path -LiteralPath $urdfPath) {
    Write-Host "xlerobot 资产已存在: $urdfPath"
    exit 0
}

Write-Host "即将下载 ManiSkill 资产 UID: xlerobot"
Write-Host "目标目录: $assetTarget"
& $pythonCommand.Source -m mani_skill.utils.download_asset xlerobot
$downloadExitCode = $LASTEXITCODE

if ((Test-Path -LiteralPath $urdfPath) -and $downloadExitCode -eq 0) {
    Write-Host "xlerobot 资产安装完成: $urdfPath"
    exit 0
}

# ManiSkill 3.0.1 creates the target before renaming the extracted GitHub folder.
# Recover only when the target is empty and the extracted folder is complete.
$assetParent = Split-Path -Parent $assetTarget
$extractedSource = Join-Path $assetParent "ManiSkill-XLeRobot-0.2.1"
$extractedUrdf = Join-Path $extractedSource "xlerobot.urdf"
if (-not (Test-Path -LiteralPath $extractedUrdf)) {
    throw "ManiSkill asset download failed with exit code $downloadExitCode; complete extracted source not found: $extractedSource"
}
if (Test-Path -LiteralPath $assetTarget) {
    if ((Get-ChildItem -LiteralPath $assetTarget -Force | Measure-Object).Count -ne 0) {
        throw "Refusing to remove non-empty xlerobot target: $assetTarget"
    }
    Remove-Item -LiteralPath $assetTarget
}
Move-Item -LiteralPath $extractedSource -Destination $assetTarget
if (-not (Test-Path -LiteralPath $urdfPath)) {
    throw "xlerobot recovery did not produce the expected URDF: $urdfPath"
}
Write-Host "xlerobot 资产安装完成: $urdfPath"
