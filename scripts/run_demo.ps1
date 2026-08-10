param(
    [ValidateSet("mock", "real")]
    [string]$Mode = "mock",

    [ValidateSet("text", "audio")]
    [Alias("Input")]
    [string]$InputMode = "text",

    [string]$Audio,
    [Alias("Command")]
    [string]$CommandText,
    [int]$Seed = 7,
    [string]$Camera = "scene_camera",
    [ValidateSet("cpu", "gpu")]
    [string]$RenderBackend = "cpu"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectSrc = Join-Path $projectRoot "src"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$projectSrc$([IO.Path]::PathSeparator)$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $projectSrc
}

$pythonCommand = Get-Command python -ErrorAction Stop
$cliArgs = @(
    "-m", "agentic_manipulation.cli",
    "--mode", $Mode,
    "--input", $InputMode,
    "--seed", $Seed,
    "--camera", $Camera,
    "--render-backend", $RenderBackend
)
if ($InputMode -eq "audio") {
    if (-not $Audio) {
        throw "-Input audio requires -Audio <path>"
    }
    $cliArgs += @("--audio", $Audio)
}
if ($CommandText) {
    $cliArgs += @("--command", $CommandText)
}

& $pythonCommand.Source @cliArgs
