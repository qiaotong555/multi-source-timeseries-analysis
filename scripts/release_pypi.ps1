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
if ($env:PYTHONPATH) { $env:PYTHONPATH = $tw + ";" + $env:PYTHONPATH } else { $env:PYTHONPATH = $tw }

# 不能只看目录在不在：曾出现「twine\ 只剩 commands/ 与 __pycache__」的半装状态，
# 目录存在但 import 失败，脚本每次都复用它，报 No module named twine.__main__。
function Test-Twine {
    & $py -c "import twine.__main__" 2>&1 | Out-Null
    return ($LASTEXITCODE -eq 0)
}
if (-not (Test-Twine)) {
    Write-Host "安装/修复 twine 到 $tw ..."
    & $py -m pip install -q --upgrade --force-reinstall --target $tw twine
    if (-not (Test-Twine)) {
        Write-Host "twine 仍不可用。删掉临时目录后重试：" -ForegroundColor Red
        Write-Host ("  Remove-Item '" + $tw + "' -Recurse -Force") -ForegroundColor Red
        exit 1
    }
}

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
    # 原生命令失败不会触发 $ErrorActionPreference，必须显式看退出码，
    # 否则上传失败也会静默返回 0，看起来像成功。
    & $py -m twine check $files.FullName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "twine check 未通过，已中止上传" -ForegroundColor Red
        exit 1
    }
    & $py -m twine upload --repository pypi $files.FullName
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("上传失败，twine 退出码 " + $LASTEXITCODE) -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host ("上传完成：" + $Version) -ForegroundColor Green
} finally {
    Remove-Item Env:\TWINE_PASSWORD -ErrorAction SilentlyContinue
}
