# 用途：把 main 推到 GitHub（token 只在本地输入，不落盘、不进命令行参数）
# 用法：powershell -ExecutionPolicy Bypass -File scripts\push_github.ps1
#       powershell -ExecutionPolicy Bypass -File scripts\push_github.ps1 -CheckOnly   # 只测网络，不推送
param([switch]$CheckOnly)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$git = "C:\Program Files\Git\cmd\git.exe"
if (-not (Test-Path $git)) { $git = "git" }

# ---------------------------------------------------------------- 网络预检
# 这台机器上 github.com 时通时不通（api.github.com / pypi.org 一直正常），
# 常见原因是本机代理没开。所以先测直连、再找本机代理，最后才问 token——
# 免得 token 输了半天，才发现根本连不上。
function Test-Tcp {
    param([string]$Target, [int]$Port, [int]$TimeoutMs = 4000)
    $client = New-Object Net.Sockets.TcpClient
    try {
        if (-not $client.ConnectAsync($Target, $Port).Wait($TimeoutMs)) { return $false }
        return $client.Connected
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

$proxy = $null
if (Test-Tcp "github.com" 443) {
    Write-Host "直连 github.com:443：可达" -ForegroundColor Green
} else {
    Write-Host "直连 github.com:443：不可达（表现为连 21 秒后失败）" -ForegroundColor Yellow
    foreach ($port in 10808, 10809, 7890, 7897) {
        if (Test-Tcp "127.0.0.1" $port 800) { $proxy = "http://127.0.0.1:$port"; break }
    }
    if (-not $proxy) {
        Write-Host "本机也没有在跑的代理（试过 10808 / 10809 / 7890 / 7897）。三选一：" -ForegroundColor Red
        Write-Host "  1) 打开平时用的代理，再跑一次这个脚本；" -ForegroundColor Red
        Write-Host "  2) 把 $env:USERPROFILE\.ssh\id_rsa.pub 加到 GitHub → Settings → SSH keys，" -ForegroundColor Red
        Write-Host "     然后走 ssh.github.com:443（实测可达，当前这把 key 未注册）；" -ForegroundColor Red
        Write-Host "  3) 过几分钟再试，这条路有时会自己恢复。" -ForegroundColor Red
        exit 1
    }
    Write-Host "改用本机代理：$proxy" -ForegroundColor Cyan
}

if ($CheckOnly) {
    Write-Host "只做了网络检查，没有推送。" -ForegroundColor Cyan
    exit 0
}

# ---------------------------------------------------------------- 推送
Write-Host "当前分支：$(& $git rev-parse --abbrev-ref HEAD)" -ForegroundColor Cyan
Write-Host "待推送的提交：" -ForegroundColor Cyan
& $git log --oneline "origin/main..HEAD"
if ($LASTEXITCODE -ne 0) { Write-Host "读不到 origin/main，先 git fetch" -ForegroundColor Red; exit 1 }

$secure = Read-Host "粘贴 GitHub token（ghp_... 或 github_pat_...）" -AsSecureString
$token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))

if ($token -notmatch '^(ghp_|github_pat_)[\x21-\x7E]+$') {
    Write-Host "token 格式不对：长度=$($token.Length)，前 6 位=$($token.Substring(0,[Math]::Min(6,$token.Length)))" -ForegroundColor Red
    Write-Host "应当以 ghp_ 或 github_pat_ 开头、只含 ASCII。" -ForegroundColor Red
    exit 1
}

# token 走环境变量交给 credential helper，不写进 git 参数，也不落盘
$env:GIT_PUSH_TOK = $token
try {
    $opts = @("-c", "credential.helper=",
              "-c", 'credential.helper=!f() { echo username=qiaotong555; echo password=$GIT_PUSH_TOK; }; f')
    if ($proxy) { $opts += @("-c", "http.proxy=$proxy") }
    & $git @opts push origin main
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Remove-Item Env:\GIT_PUSH_TOK -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "推送完成。远端 main：" -ForegroundColor Green
& $git --no-pager log --oneline -1 origin/main