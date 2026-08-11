param(
  [string]$Root = "D:\LMO-MOO-runs",
  [string[]]$Methods = @(),
  [string]$Method = ""
)
if ($Method -and $Methods.Count -eq 0) { $Methods = @($Method) }  # backwards-compatible singular form
if (-not (Test-Path $Root)) { Write-Host "No run directory: $Root"; exit 0 }
$cacheDirs = Get-ChildItem -Path $Root -Recurse -Directory -Filter cache
foreach ($cache in $cacheDirs) {
  if ($Methods.Count -gt 0) {
    foreach ($m in $Methods) {
      $safe = $m.Replace('/', '_').Replace(' ', '_')
      $target = Join-Path $cache.FullName $safe
      if (Test-Path $target) { Remove-Item $target -Recurse -Force; Write-Host "Removed $target" }
    }
  } else {
    Remove-Item $cache.FullName -Recurse -Force
    New-Item -ItemType Directory $cache.FullName | Out-Null
    Write-Host "Cleared $($cache.FullName)"
  }
}
