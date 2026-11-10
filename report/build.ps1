# Build BeeTrack_Report_{EN,ZH}.pdf: fill team placeholders from proposal/team.json into temporary copies,
# print them with headless Edge, then delete the copies (names never land in the HTML sources).
# run: powershell -ExecutionPolicy Bypass -File report\build.ps1
param(
  [string]$Edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
  [string]$Team = "$PSScriptRoot\..\proposal\team.json"
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding($false)

# team.json: [["name","id"], ...] in role order 1..4
$members = [IO.File]::ReadAllText((Resolve-Path $Team), $utf8) | ConvertFrom-Json
if ($members.Count -ne 4) { throw "team.json must list 4 members, found $($members.Count)" }

$edgeProfile = Join-Path $env:TEMP "beetrack_edge_profile"   # separate profile so a running Edge does not swallow the job
foreach ($lang in "en", "zh") {
  $html = [IO.File]::ReadAllText("$here\report_$lang.html", $utf8)
  for ($i = 0; $i -lt 4; $i++) {
    $html = $html.Replace("{{NAME$($i + 1)}}", [Net.WebUtility]::HtmlEncode($members[$i][0]))
    $html = $html.Replace("{{ID$($i + 1)}}", [Net.WebUtility]::HtmlEncode($members[$i][1]))
  }
  if ($html -match "\{\{(NAME|ID)\d\}\}") { throw "unfilled placeholder in report_$lang.html" }
  $tmp = "$here\_build_$lang.html"          # same folder so relative image paths still resolve
  $pdf = "$here\BeeTrack_Report_$($lang.ToUpper()).pdf"
  [IO.File]::WriteAllText($tmp, $html, $utf8)
  try {
    if (Test-Path $pdf) { Remove-Item $pdf }
    $url = "file:///" + ($tmp -replace "\\", "/")
    $edgeArgs = @("--headless", "--disable-gpu", "--no-first-run", "--user-data-dir=$edgeProfile", "--run-all-compositor-stages-before-draw",
              "--virtual-time-budget=5000", "--no-pdf-header-footer", "--print-to-pdf=$pdf", $url)
    Start-Process -FilePath $Edge -ArgumentList $edgeArgs -Wait -WindowStyle Hidden
  } finally {
    Remove-Item $tmp -ErrorAction SilentlyContinue
  }
  if (-not (Test-Path $pdf)) { throw "Edge did not write $pdf" }
  $raw = [IO.File]::ReadAllText($pdf, [Text.Encoding]::GetEncoding(28591))
  $pages = ([regex]::Matches($raw, "/Type\s*/Page[^s]")).Count
  "{0}: {1} pages, {2:N0} KB" -f (Split-Path $pdf -Leaf), $pages, ((Get-Item $pdf).Length / 1KB)
}
