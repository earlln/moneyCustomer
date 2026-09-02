@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ================================================================
echo   금전 대고객 작업 분류 - 설치 자체점검
echo ================================================================
echo.
echo 이 PC 에 파이썬이나 Java 를 설치하지 않아도 프로그램이 정상
echo 동작하는지 확인합니다. 예시 데이터 12건을 분류해 봅니다.
echo.

if not exist "_internal\" (
  echo [실패] _internal 폴더가 없습니다.
  echo        압축을 풀 때 폴더 구조가 유지되지 않았습니다.
  echo        zip 파일을 통째로 다시 풀어주세요.
  echo.
  pause
  exit /b 1
)

if not exist "batch_predict.exe" (
  echo [실패] batch_predict.exe 를 찾을 수 없습니다.
  echo        이 배치 파일은 exe 와 같은 폴더에 있어야 합니다.
  echo.
  pause
  exit /b 1
)

batch_predict.exe -i sample_input.csv -o 자체점검_결과.csv --show --no-pause
if errorlevel 1 goto failed

echo.
echo ================================================================
echo   [성공] 프로그램이 정상 동작합니다.
echo   결과 파일: 자체점검_결과.csv
echo.
echo   이제 input CSV 를 준비한 뒤 batch_predict.exe 를 실행하세요.
echo   자세한 사용법은 사용설명서.txt 를 참고하세요.
echo ================================================================
echo.
pause
exit /b 0

:failed
echo.
echo ================================================================
echo   [실패] 실행 중 오류가 발생했습니다.
echo   위에 출력된 오류 메시지를 확인해 주세요.
echo ================================================================
echo.
pause
exit /b 1
