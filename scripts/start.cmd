@echo off
REM 启动远程 Qwen3-ASR 会议记录工具
setlocal

cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] 未检测到虚拟环境，正在创建 .venv ...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo [ERROR] 创建虚拟环境失败，请确认已安装 Python 3.10+
        pause
        exit /b 1
    )
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\pip.exe install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] 依赖安装失败
        pause
        exit /b 1
    )
)

echo [INFO] 启动应用...
.venv\Scripts\python.exe src\app.py
endlocal
