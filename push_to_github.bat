@echo off
echo Initializing Git repository for AstroGuard-AI...
git init
git branch -M main
git add .
git commit -m "Initial commit: AstroGuard AI 5-Stage (27-Phase) Spacecraft Telemetry Platform"
git remote remove origin 2>nul
git remote add origin https://github.com/BibinBinoy777/AstroGuard-AI.git
echo Pushing to https://github.com/BibinBinoy777/AstroGuard-AI.git ...
git push -u origin main
echo Done! View your repository at: https://github.com/BibinBinoy777/AstroGuard-AI
