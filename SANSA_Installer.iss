[Setup]
AppName=SANSA Geomagnetic Data Quality Check Tool
AppVersion=1.0
AppPublisher=SANSA Space Science
AppPublisherURL=https://www.sansa.org.za
AppCopyright=Copyright (C) 2025 SANSA Space Science
DefaultDirName={autopf}\SANSA Geomagnetic Tool
DefaultGroupName=SANSA Geomagnetic Tool
OutputDir=Output
OutputBaseFilename=SANSA_Geomagnetic_Tool_Setup
SetupIconFile=app_icon.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
DisableWelcomePage=no
DisableDirPage=no
DisableProgramGroupPage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\SANSA_Geomagnetic_Tool.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "baselineValues.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "SANSA (004) (2).png"; DestDir: "{app}"; Flags: ignoreversion
Source: "dYlY3ZRD_400x400.jpg"; DestDir: "{app}"; Flags: ignoreversion
Source: "app_icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\SANSA Geomagnetic Tool"; Filename: "{app}\SANSA_Geomagnetic_Tool.exe"; IconFilename: "{app}\SANSA_Geomagnetic_Tool.exe"
Name: "{autodesktop}\SANSA Geomagnetic Tool"; Filename: "{app}\SANSA_Geomagnetic_Tool.exe"; Tasks: desktopicon; IconFilename: "{app}\SANSA_Geomagnetic_Tool.exe"

[Run]
Filename: "{app}\SANSA_Geomagnetic_Tool.exe"; Description: "{cm:LaunchProgram,SANSA Geomagnetic Tool}"; Flags: nowait postinstall skipifsilent