@echo off
echo.
echo ===================================
echo  AI Nexus - GitHub Push Script
echo ===================================
echo.
echo This script will push your code to GitHub.
echo.
echo Make sure you have:
echo 1. Created the repository on GitHub (https://github.com/new)
echo 2. Repository name: ai-nexus
echo.
pause

echo.
echo Pushing to GitHub...
git push -u origin master

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ===================================
    echo  Authentication Failed!
    echo ===================================
    echo.
    echo If you see an authentication error, you need to:
    echo 1. Go to https://github.com/settings/tokens
    echo 2. Generate a new token with 'repo' scope
    echo 3. Use your GitHub username and the token as password
    echo.
    echo Or use GitHub CLI: gh auth login
    echo.
) else (
    echo.
    echo ===================================
    echo  Success! Code pushed to GitHub
    echo ===================================
    echo.
    echo Next steps:
    echo 1. Enable GitHub Pages in repository settings
    echo 2. Deploy backend to Render.com
    echo 3. Update config.js with your backend URL
    echo.
    echo See DEPLOYMENT_GUIDE.md for detailed instructions
    echo.
)

pause