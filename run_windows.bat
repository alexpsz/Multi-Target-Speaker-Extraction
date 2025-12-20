@echo off
setlocal enabledelayedexpansion

REM Switch to script directory | 切换到脚本所在目录
cd /d "%~dp0"

REM Set UTF-8 codepage (ignore errors if not supported)
chcp 65001 >nul 2>&1

echo ============================================================
echo Multi-Target Speaker Extraction (MTSE)
echo 多目标说话人提取工具
echo ============================================================
echo.

:menu
echo Please select an option:
echo [1] Run system
echo [2] View help
echo [3] Exit
echo.

set "choice="
set /p choice="Enter option (1-3): "

if "%choice%"=="1" goto run
if "%choice%"=="2" goto help
if "%choice%"=="3" goto end

echo Invalid option, please try again
echo.
goto menu

:run
echo.
echo Running Multi-Target Speaker Extraction...
echo.
python run.py
echo.
echo ============================================================
echo Process completed. Press any key to return to menu...
pause >nul
goto menu

:help
echo.
echo ============================================================
echo User Guide
echo ============================================================
echo.
echo 1. Prepare reference audio:
echo    - Create speaker folders in enrollment_audio/
echo    - Each folder contains clean audio samples of that speaker
echo    Example:
echo      enrollment_audio/SpeakerA/
echo      enrollment_audio/SpeakerB/
echo.
echo 2. Run the system:
echo    - Select [1] to run
echo    - Automatically processes all audio in input_audio/
echo.
echo 3. View results:
echo    - Output in output/
echo    - Each speaker has a separate folder
echo    - Filename format: similarity_originalname_segment.wav
echo.
echo 4. Adjust parameters:
echo    - Edit config.yaml
echo.
echo ============================================================
echo.
pause
goto menu

:end
echo Thank you for using!
endlocal
exit /b 0

