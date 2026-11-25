@echo off
REM ===== Chay WEBSITE du doan (Tuan 11-12). Can chay muc 5 truoc de co output\best_model.ser (can JDK 17 tro len) =====
chcp 65001 >nul
cd /d "%~dp0"
if not exist bin mkdir bin
echo Dang bien dich...
REM Liet ke tung thu muc package (javac tren Windows khong hieu dau * o ten thu muc)
javac --release 17 -encoding UTF-8 -d bin -cp "lib/*" src\main\java\nhom7\diabetes\*.java ^
    src\main\java\nhom7\diabetes\collection\*.java ^
    src\main\java\nhom7\diabetes\config\*.java ^
    src\main\java\nhom7\diabetes\data\*.java ^
    src\main\java\nhom7\diabetes\db\*.java ^
    src\main\java\nhom7\diabetes\evaluation\*.java ^
    src\main\java\nhom7\diabetes\model\*.java ^
    src\main\java\nhom7\diabetes\preprocessing\*.java ^
    src\main\java\nhom7\diabetes\report\*.java ^
    src\main\java\nhom7\diabetes\tuning\*.java ^
    src\main\java\nhom7\diabetes\web\*.java
if errorlevel 1 (
    echo.
    echo Bien dich that bai. Xem thong bao loi o tren.
    echo Neu bao 'javac' is not recognized: can cai JDK 17+ va them vao PATH.
    pause
    exit /b 1
)
start "" http://localhost:8080
java -Dstdout.encoding=UTF-8 -cp "bin;lib/*" nhom7.diabetes.Main web
pause
