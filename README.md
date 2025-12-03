# SANSA Magnetic Data Analysis Tool

A professional Python application for analyzing and visualizing geomagnetic data from SANSA (South African National Space Agency) magnetometer stations.

## Overview

This tool provides a graphical interface for:
- Single-station magnetic field component analysis (H, D, Z, F)
- Multi-station data comparison across different instruments
- Data quality validation and cleaning
- SQD/CTU data format conversion
- Direct FTP access to SANSA magnetic field data archives

## Features

### Single Station Analysis
Detailed plots of magnetic field components:
- **H** - Horizontal intensity (nT)
- **D** - Declination (arc-minutes)
- **Z** - Vertical intensity (nT)
- **F** - Total field intensity (nT)

### Multi-Station Comparison
Compare magnetic field data from multiple stations and instruments simultaneously to identify regional variations and data quality issues.

### Data Quality Control
Automatic detection and handling of:
- Invalid data markers (sentinel values: 88888, 99999)
- Data gaps and discontinuities
- Statistical outliers

### Supported Instruments
| Code | Instrument Type |
|------|----------------|
| OVH | Overhauser magnetometer (scalar F measurements) |
| DTU | DTU-Space fluxgate magnetometer |
| FGM1/FGM2 | Fluxgate magnetometers |
| L251 | LEMI-025 magnetometer |
| SQD | SQUID magnetometer |
| CTU | CTU magnetometer |

### Supported Stations
- **HER** - Hermanus
- **HBK** - Hartebeesthoek
- **KMH** - Keetmanshoop
- **TSU** - Tsumeb
- **SNA** - SANAE (Antarctica)

## System Requirements

- Windows 10 or Windows 11
- Python 3.9 or higher
- 4 GB RAM minimum (8 GB recommended)
- 500 MB free disk space
- Internet connection for FTP data access

## Installation

### From Source

1. Clone or download the repository:
   ```bash
   git clone https://github.com/SakhileNM/SANSA-Geomagnetic-Data-QC-Tool
   cd "Geomagnetic Data App"
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python application.py
   ```

### From Installer

1. Run `SANSA_Geomagnetic_Tool_Setup.exe`
2. Follow the installation wizard
3. Launch from Start Menu or Desktop shortcut

To learn how to **create the installer**, please refer to the Work Instruction on M-Files: https://sansa.cloudvault.m-files.com/vnext/link/BA84D00E-9167-4344-AE90-844715069F12/show?object=F741FD59-F78A-4708-9A54-B0D56C39BE08&objid=0-22816&version=latest

To access an **official version**, follow the link: https://sansa365-my.sharepoint.com/:u:/g/personal/smkhize_sansa_org_za/IQDvufiS3Lj2Sopw1krIQUkYAe2bBxCaPyk53Mr0tHbVfK0?e=oNNlw3

## Usage

1. **Configure your analysis** in the left panel:
   - Select station(s) from the dropdown
   - Choose instrument type(s)
   - Set the date range for analysis
   - Select single or multi-station mode

2. **Generate plots** by clicking "Plot Comparison" or "Plot Single Station"

3. **View results** in the right panel with interactive matplotlib plots

4. **Export** using "Save Plot" to save images in various formats

## Project Structure

```
Geomagnetic Data App/
├── application.py                     # Main GUI application (PyQt5)
├── connectFTP.py                      # FTP connection handler for SANSA servers
├── plot_convert_SingleOnGUI_FTP.py    # Single station with conversion
├── plot_convert_MultipleOnGUI_FTP.py  # Multi-station with conversion
├── baselineValues.json                # Baseline calibration values
├── requirements.txt                   # Python dependencies
├── app_icon.ico                       # SANSA logo as the application icon
├── build_sansa_tool.bat               # Build the tool as an executable application
├── create_installer.bat               # Extract and create the installer file
├── dYlY3ZRD_400x400.jpg               # SANSA logo
├── SANSA (004) (2).png                # SANSA logo (in applictaion)
├── SANSA_Geomagnetic_Tool.spec        # Executable specifications
├── SANSA_Installer.iss                # Inno Setup Script
└── README.md                          # This file
```

## Data Sources

The application connects to SANSA's FTP server to retrieve magnetic field data:
- **Server**: `storage4.her.sansa.org.za`
- **Path**: `/SANSA_INSTRUMENTS/`
- **Access**: Anonymous FTP

Data is organized by instrument type, station version, and date in the directory structure:
```
/SANSA_INSTRUMENTS/{INSTRUMENT}/{STATION_VERSION}/raw/{YEAR}/{MONTH}/
```

## Output Formats

- **Plots**: PNG (via matplotlib)
- **Data**: IAGA-2002 standard format for processed magnetic field data

## Configuration

Baseline values and calibration parameters are stored in `baselineValues.json`. These values are used for data quality checks and instrument-specific adjustments.

## Dependencies

- **PyQt5** - GUI framework
- **matplotlib** - Plotting and visualization
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **ftplib** - FTP connectivity (standard library)

See `requirements.txt` for complete dependency list with versions.

## Troubleshooting

### FTP Connection Issues
- Verify internet connectivity
- Check if SANSA FTP server is accessible
- The application will automatically attempt to reconnect on connection loss

### Missing Data
- Some date ranges may not have data for all instruments
- OVH instruments have version-specific date ranges
- Check the SANSA data archive for data availability

### Performance
- Large date ranges may take time to process
- Multi-station comparisons require more memory
- Consider processing in smaller date chunks for extensive analyses

## License

This software is developed for SANSA Space Science research purposes.

## Contact

SANSA Space Science
Hermanus, South Africa
https://www.sansa.org.za





