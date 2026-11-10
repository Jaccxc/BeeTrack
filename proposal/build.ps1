# Rebuild the proposal decks and their PDFs.  run: powershell -File courses\dlia\final_project\proposal\build.ps1
# needs: Python with Pillow (make_figs.py), Node with pptxgenjs on NODE_PATH, the pptx skill's apply_theme.js, PowerPoint (PDF export)
param(
  [string]$NodeModules = "$env:LOCALAPPDATA\Temp\claude\d90200169f91afe2\592aba94-8446-4588-8edc-982878d78d00\scratchpad\deck\node_modules",
  [string]$ApplyTheme = "$env:LOCALAPPDATA\Claude-Data\claude_pzs8sxrjxfjjc\local-agent-mode-sessions\skills-plugin\7bde84b8-1975-48a5-b2e2-2babd1b842dc\c9227583-fce5-4fb1-9284-f51fbd8ed745\skills\pptx\scripts\apply_theme.js"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python make_figs.py | Out-Null
$env:NODE_PATH = $NodeModules
node build_deck.js $ApplyTheme
$pp = New-Object -ComObject PowerPoint.Application
foreach ($f in "DLIA_Proposal_BeeTrack_EN", "DLIA_Proposal_BeeTrack_ZH") {
  $p = $pp.Presentations.Open("$PSScriptRoot\$f.pptx", $true, $false, $false)
  $p.SaveAs("$PSScriptRoot\$f.pdf", 32); $p.Close()
}
$pp.Quit()
