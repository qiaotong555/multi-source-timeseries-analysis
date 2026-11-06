# 用途：把一个版本上传到 PyPI（token 只在这里输入，不落盘）
# 用法：在仓库任意位置执行（版本号默认读 pyproject.toml，也可用 -Version 指定）
#   powershell -ExecutionPolicy Bypass -File scripts\release_pypi.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\release_pypi.ps1 -Version 0.5.0
param([string]$Version = "")
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = "C:\Users\乔佟\.workbuddy\binaries\python\envs\mtsa\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

# twine 装在临时目录，不污染项目环境
$tw = Join-Path $env:TEMP "tw"
if (-not (Test-Path (Join-Path $tw "twine"))) {
    Write-Host "安装 twine 到 $tw ..."
    & $py -m pip install -q --target $tw twine
}
$env:PYTHONPATH = $tw

# 只上传目标版本：dist\ 里常躺着历史版本，全量上传会被 PyPI 拒绝（该版本已存在）
if (-not $Version) {
    $m = Select-String -Path (Join-Path $root "pyproject.toml") -Pattern '^version\s*=\s*"([^"]+)"'
    if (-not $m) {
        Write-Host "读不到 pyproject.toml 里的 version" -ForegroundColor Red
        exit 1
    }
    $Version = $m.Matches[0].Groups[1].Value
}
Write-Host "目标版本：$Version"

$dist = Join-Path $root "dist"
$pat = "^mtsa_agent-" + [regex]::Escape($Version) + "[.-]"
$files = @(Get-ChildItem $dist -Filter "mtsa_agent-*" -ErrorAction SilentlyContinue | Where-Object { $_.Name -match $pat })
if ($files.Count -eq 0) {
    Write-Host "dist\ 下没有 $Version 的产物，先构建（python -m build --sdist --wheel）" -ForegroundColor Red
    exit 1
}

$others = @(Get-ChildItem $dist -Filter "mtsa_agent-*" | Where-Object { $_.Name -notmatch $pat })
if ($others.Count -gt 0) {
    Write-Host ("另有 {0} 个其它版本的产物不会上传：{1}" -f $others.Count, (($others | ForEach-Object { $_.Name }) -join ", ")) -ForegroundColor Yellow
}

$secure = Read-Host "粘贴 PyPI token（pypi-...）" -AsSecureString
$token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))

if ($token -notmatch '^pypi-[\x21-\x7E]+$') {
    Write-Host "token 格式不对：长度=$($token.Length)，前 6 位=$($token.Substring(0,[Math]::Min(6,$token.Length)))" -ForegroundColor Red
    Write-Host "应当以 pypi- 开头、只含 ASCII、长度约 70-90。检查是否粘错或输入法改成了全角。" -ForegroundColor Red
    exit 1
}

$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = $token
try {
    & $py -m twine check $files.FullName
    & $py -m twine upload --repository pypi $files.FullName
} finally {
    Remove-Item Env:\TWINE_PASSWORD -ErrorAction SilentlyContinue
}
