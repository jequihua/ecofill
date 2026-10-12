param([Parameter(Mandatory=$true)][string]$Receipt,
      [Parameter(Mandatory=$true)][string]$Job,
      [Parameter(Mandatory=$true)][string]$Project,
      [string]$Region = 'us-central1')
$ErrorActionPreference = 'Stop'
$runList = Get-Content -LiteralPath $Receipt -Raw | ConvertFrom-Json
if ($runList.status -ne 'staged') { throw 'Actual upload generations are required.' }
foreach ($item in $runList.jobs) {
    if (-not $item.supported) { throw 'Unsupported job requires architect action.' }
    $runArgs = @('run','--config',$item.uri,'--config-generation',[string]$item.generation,
                 '--config-sha256',$item.sha256,'--run-id',$item.run_id) -join ','
    & gcloud run jobs execute $Job "--project=$Project" "--region=$Region" "--args=$runArgs" --wait
    if ($LASTEXITCODE -ne 0) { throw "Execution failed for $($item.run_id); serial list stopped." }
}
