@echo off
cd /d C:\Users\intraw_sewon\Desktop\workspace\intraw-real-ESRGAN
C:\Users\intraw_sewon\miniconda3\envs\realesrgan\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8002
pause
