# 用途：把 main 推到 GitHub（token 只在本地输入，不落盘、不进命令行参数）
# 用法：powershell -ExecutionPolicy Bypass -File scripts\push_github.ps1
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$git = "C:\Program Files\Git\cmd\git.exe"
if (-not (Test-Path $git)) { $git = "git" }

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
    & $git -c credential.helper= -c 'credential.helper=!f() { echo username=qiaotong555; echo password=$GIT_PUSH_TOK; }; f' push origin main
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Remove-Item Env:\GIT_PUSH_TOK -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "推送完成。远端 main：" -ForegroundColor Green
& $git --no-pager log --oneline -1 origin/main
