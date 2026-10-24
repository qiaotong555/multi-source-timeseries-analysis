# 用途：把一个版本上传到 PyPI（token 只在这里输入，不落盘）
# 用法：在仓库任意位置执行
#   powershell -ExecutionPolicy Bypass -File scripts\release_pypi.ps1
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

$files = Get-ChildItem (Join-Path $root "dist") -Filter "mtsa_agent-*" -ErrorAction SilentlyContinue
if (-not $files) {
    Write-Host "dist\ 下没有 mtsa_agent-* 产物，先构建：python -m build" -ForegroundColor Red
    exit 1
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
