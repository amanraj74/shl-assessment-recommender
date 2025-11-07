@echo off
echo 🚀 Starting SHL Assessment Recommender...

echo.
echo ✅ Activating virtual environment...
call venv\Scripts\activate

echo.
echo ✅ Starting backend API...
start cmd /k "cd backend && python app.py"

timeout /t 3

echo.
echo ✅ Starting frontend server...
start cmd /k "cd frontend && python -m http.server 8000"

echo.
echo ✅ All services started!
echo    - API: http://localhost:5000
echo    - UI:  http://localhost:8000
echo.
pause
