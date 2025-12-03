@echo off
echo Building SANSA Geomagnetic Data Analysis Tool...
echo.

cd /d "%~dp0"

:: Install required packages
echo Installing required packages...
pip install pyinstaller pillow

:: Convert JPG to ICO for the application icon
echo Converting application icon to ICO format...
python -c "from PIL import Image; img = Image.open('dYlY3ZRD_400x400.jpg'); img.save('app_icon.ico', format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])"

:: Create the executable
echo Creating standalone executable...
python -m PyInstaller --onefile --windowed --name "SANSA_Geomagnetic_Tool" ^
--add-data "baselineValues.json;." ^
--add-data "SANSA (004) (2).png;." ^
--add-data "dYlY3ZRD_400x400.jpg;." ^
--hidden-import=PyQt5 ^
--hidden-import=PyQt5.QtWidgets ^
--hidden-import=PyQt5.QtCore ^
--hidden-import=PyQt5.QtGui ^
--hidden-import=matplotlib.backends.backend_qt5agg ^
--hidden-import=matplotlib.backends.backend_agg ^
--hidden-import=matplotlib.backends.backend_tkagg ^
--hidden-import=pandas._libs.tslibs.timedeltas ^
--hidden-import=pandas._libs.tslibs.np_datetime ^
--hidden-import=pandas._libs.tslibs.period ^
--hidden-import=scipy._lib.messagestream ^
--hidden-import=pytz ^
--hidden-import=six ^
--hidden-import=dateutil ^
--hidden-import=dateutil.parser ^
--hidden-import=connectFTP ^
--hidden-import=plotSingleOnGUI_FTP ^
--hidden-import=plotMultipleOnGUI_FTP ^
--collect-all matplotlib ^
--collect-all PyQt5 ^
--icon="app_icon.ico" ^
application.py

if exist "dist\SANSA_Geomagnetic_Tool.exe" (
    echo.
    echo SUCCESS: Build complete!
    echo The executable is in the 'dist' folder: SANSA_Geomagnetic_Tool.exe
    echo.
    echo You can now run create_installer.bat to create the installer.
) else (
    echo.
    echo ERROR: Build failed! Check the console for errors.
)

echo.
pause