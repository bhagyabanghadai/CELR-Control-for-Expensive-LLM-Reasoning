# Rewrite ALL commit history: fix author name + email
# Correct identity:
#   Name:  bhagyabanghadai
#   Email: bhagyaban24523@gmail.com

$CORRECT_NAME = "bhagyabanghadai"
$CORRECT_EMAIL = "bhagyaban24523@gmail.com"

Write-Host "Rewriting git history to use:" -ForegroundColor Cyan
Write-Host "  Name:  $CORRECT_NAME" -ForegroundColor Cyan
Write-Host "  Email: $CORRECT_EMAIL" -ForegroundColor Cyan
Write-Host ""

# Remove old filter-branch backup if it exists (allows re-run)
git for-each-ref --format="delete %(refname)" refs/original 2>$null | git update-ref --stdin 2>$null

# Use filter-branch to rewrite every commit's author + committer
$env_filter = @"
if [ -n "`$GIT_COMMITTER_EMAIL" ]; then
    export GIT_COMMITTER_NAME="$CORRECT_NAME"
    export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"
fi
if [ -n "`$GIT_AUTHOR_EMAIL" ]; then
    export GIT_AUTHOR_NAME="$CORRECT_NAME"
    export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
fi
"@

git filter-branch -f --env-filter @"
export GIT_COMMITTER_NAME="$CORRECT_NAME"
export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"
export GIT_AUTHOR_NAME="$CORRECT_NAME"
export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
"@ --tag-name-filter cat -- --branches --tags

Write-Host ""
if ($LASTEXITCODE -eq 0) {
    Write-Host "History rewritten successfully!" -ForegroundColor Green
    Write-Host "Force-pushing to GitHub..." -ForegroundColor Yellow
    git push origin master --force
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Force push successful!" -ForegroundColor Green
    }
    else {
        Write-Host "Force push failed. Try: git push origin master --force" -ForegroundColor Red
    }
}
else {
    Write-Host "filter-branch failed!" -ForegroundColor Red
}
