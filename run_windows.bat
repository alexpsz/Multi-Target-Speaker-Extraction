@echo off
chcp 65001 >nul
echo ============================================================
echo Multi-Target Speaker Extraction (MTSE)
echo 多目标说话人提取工具
echo ============================================================
echo.

:menu
echo Please select an option | 请选择操作:
echo [1] Run system | 运行系统
echo [2] View help | 查看帮助
echo [3] Exit | 退出
echo.

set /p choice="Enter option (1-3) | 请输入选项 (1-3): "

if "%choice%"=="1" goto run
if "%choice%"=="2" goto help
if "%choice%"=="3" goto end

echo Invalid option, please try again | 无效选项，请重新选择
echo.
goto menu

:run
echo.
echo Running Multi-Target Speaker Extraction...
echo 正在运行多目标说话人提取...
echo.
python run.py
echo.
pause
goto menu

:help
echo.
echo ============================================================
echo User Guide | 使用说明
echo ============================================================
echo.
echo 1. Prepare reference audio | 准备参考音频:
echo    - Create speaker folders in enrollment_audio/
echo    - 在 enrollment_audio/ 下创建说话人文件夹
echo    - Each folder contains clean audio samples of that speaker
echo    - 每个文件夹包含该说话人的纯净音频样本
echo    Example | 例如:
echo      enrollment_audio/SpeakerA/
echo      enrollment_audio/SpeakerB/
echo.
echo 2. Run the system | 运行系统:
echo    - Select [1] to run | 选择 [1] 运行系统
echo    - Automatically processes all audio in input_audio/
echo    - 自动处理 input_audio/ 中的所有音频
echo.
echo 3. View results | 查看结果:
echo    - Output in output/
echo    - 输出在 output/
echo    - Each speaker has a separate folder
echo    - 每个说话人有独立的文件夹
echo    - Filename format: similarity_originalname_segment.wav
echo    - 文件名格式: 相似度_原文件名_片段信息.wav
echo    - Sort by filename (descending) to view by similarity
echo    - 按文件名降序排列即可查看相似度从高到低
echo.
echo 4. Adjust parameters | 调整参数:
echo    - Edit config.yaml
echo    - 编辑 config.yaml
echo.
echo ============================================================
echo.
pause
goto menu

:end
echo Thank you for using! | 感谢使用！
exit

