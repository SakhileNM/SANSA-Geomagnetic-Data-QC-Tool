@echo off
echo Creating SANSA Geomagnetic Tool Installer...
echo.

:: Check if Inno Setup is available
where iscc >nul 2>nul
if errorlevel 1 (
    echo Inno Setup not found!
    echo Please install Inno Setup from: http://www.jrsoftware.org/isdl.php
    echo Then run this script again.
    pause
    exit /b 1
)

:: Create the installer
iscc SANSA_Installer.iss

echo.
echo Installer created in the 'Output' folder!
echo.
pause