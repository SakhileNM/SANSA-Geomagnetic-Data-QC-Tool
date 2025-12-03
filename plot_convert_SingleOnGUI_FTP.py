"""Single-station magnetic data plotting and SQD/CTU format conversion module.

This module provides functionality for:
- Processing and plotting single-station geomagnetic data from FTP sources
- Converting SQD (SQUID magnetometer) and CTU data to IAGA-2002 format
- Comparing SQD/CTU data against HERDTU1 reference data
- Supporting multiple output formats (HDZF, DHZF, XYZF)

Supported instruments:
    - FGM1/FGM2: Fluxgate magnetometers (minute resolution)
    - DTU1: DTU-Space fluxgate magnetometer (second resolution)
    - L251: LEMI-025 magnetometer (second resolution)
    - SQD: SQUID magnetometer with rotation matrix correction
    - CTU: CTU magnetometer

Data formats:
    - HDZF: Horizontal, Declination, Vertical, Total Field
    - DHZF: Declination, Horizontal, Vertical, Total Field
    - XYZF: X, Y, Z geographic components, Total Field

Sentinel values treated as invalid data:
    - 88888: F component sentinel
    - 99999, 99999.99: General invalid data markers

Version: 2025/09/10
"""

import argparse
import json
import math
import os
import re
import sys
import tempfile
import traceback
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from connectFTP import get_ftp_handler

# Matplotlib configuration
mpl.rcParams['font.size'] = 10
mpl.rcParams['lines.linewidth'] = 0.6

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

# =============================================================================
# CONSTANTS
# =============================================================================

# Station identification for IAGA-2002 output headers
STATION_NAME = 'Hermanus Magnetic Observatory'
IAGA_CODE = 'HER'

# Output subdirectory names
SQD_SUBDIR = 'SQD'
CTU_SUBDIR = 'CTU'

# Sentinel values for invalid data in IAGA-2002 format
SEC_SENTINEL = 99999.00  # Second-resolution data sentinel
F_SENTINEL = 88888.00    # Total field (F) sentinel

# Rotation matrix for SQD instrument frame to geographic XYZ frame
# This matrix transforms SQUID measurements to geographic coordinates
R_SQUID = np.array([
    [0.6749, -0.3457, -0.0378],
    [0.5199,  0.8583, -0.0554],
    [0.0204, -0.0130,  0.8234]
], dtype=float)

# =============================================================================
# TIME CONVERSION FUNCTIONS
# =============================================================================


def mapTimeValues(time: int) -> int:
    """Map encoded time values to standard seconds since midnight.

    Converts time values encoded in HHMMSS or MMSS format to total seconds.

    Args:
        time: Encoded time value. Format depends on magnitude:
            - >= 10000: HHMMSS format (e.g., 143025 = 14:30:25)
            - >= 100: MMSS format (e.g., 3025 = 30:25)
            - < 100: Already in seconds

    Returns:
        Total seconds since midnight.

    Examples:
        >>> mapTimeValues(143025)  # 14:30:25
        52225
        >>> mapTimeValues(3025)    # 30:25
        1825
        >>> mapTimeValues(45)      # 45 seconds
        45
    """
    if time >= 10000:
        hours = time // 10000
        minutes = (time % 10000) // 100
        seconds = time % 100
        return (hours * 3600) + (minutes * 60) + seconds
    elif time >= 100:
        minutes = time // 100
        seconds = time % 100
        return (minutes * 60) + seconds
    return time


def fixTimeData(data: pd.DataFrame) -> None:
    """Apply time value mapping to a DataFrame's Time column.

    Converts encoded time values in the 'Time' column to seconds since midnight
    using mapTimeValues(). Modifies the DataFrame in place.

    Args:
        data: DataFrame with a 'Time' column containing encoded time values.
    """
    data['Time'] = data['Time'].apply(mapTimeValues)

# =============================================================================
# DATA CLEANING AND SENTINEL MASKING FUNCTIONS
# =============================================================================


def _mask_sentinels_and_extremes(
    s: pd.Series,
    sentinels: tuple = (88888, 99999, 99999.99),
    extreme_threshold: float = 1e6,
    sentinel_floor: float = 90000.0
) -> pd.Series:
    """Replace sentinel codes and extreme values with NaN.

    Performs vectorized masking of known sentinel values used to indicate
    invalid or missing data in geomagnetic data files.

    Args:
        s: Input pandas Series to clean.
        sentinels: Tuple of specific sentinel values to mask.
        extreme_threshold: Values with absolute value above this are masked.
        sentinel_floor: Values with absolute value >= this are masked.

    Returns:
        Cleaned Series with sentinel/extreme values replaced by NaN.
    """
    if s is None or len(s) == 0:
        return s
    out = s.copy().astype(float)
    mask = np.zeros(len(out), dtype=bool)
    for code in sentinels:
        mask |= (out == code)
    mask |= (out.abs() >= float(sentinel_floor))
    mask |= (out.abs() > extreme_threshold)
    out[mask] = np.nan
    return out


def remove_fat_outliers(
    series: pd.Series,
    window_size: int = 51,
    threshold_factor: float = 5
) -> pd.Series:
    """Remove clusters of outliers using rolling median and MAD.

    Identifies and removes 'fat' outliers - short groups of consecutive
    points that deviate significantly from the local trend but are not
    extreme enough to be obvious spikes.

    Args:
        series: Input data series to clean.
        window_size: Size of rolling window for median calculation.
        threshold_factor: Multiple of MAD to use as outlier threshold.

    Returns:
        Cleaned series with outlier groups replaced by NaN.
    """
    if series is None or len(series) == 0:
        return series
    s = series.copy()
    trend = s.rolling(window=window_size, center=True, min_periods=window_size//2).median()
    deviation = (s - trend).abs()
    mad = deviation.rolling(window=window_size, center=True, min_periods=window_size//2).median()
    threshold = threshold_factor * mad
    fat_mask = (deviation > threshold) & (deviation < 1000)
    fat_groups = fat_mask.astype(int).diff().ne(0).cumsum()
    for group_id, group_data in fat_mask.groupby(fat_groups):
        if group_data.any() and group_data.sum() <= 20:
            s[group_data.index] = np.nan
    removed = fat_mask.sum()
    if removed > 0:
        print(f'  Removed {removed} "fat" outlier points', flush=True)
    return s


def d_to_arcmin(D_nT: np.ndarray, H_nT: np.ndarray) -> np.ndarray:
    """Convert declination from nT to arc-minutes.

    Performs the standard conversion of declination component from nT
    to arc-minutes using the formula: D_arcmin = D_nT * 3438.0 / H_nT

    Args:
        D_nT: Declination values in nT.
        H_nT: Horizontal component values in nT (used as denominator).

    Returns:
        Declination values converted to arc-minutes.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mask = (~np.isnan(D_nT)) & (~np.isnan(H_nT)) & (np.abs(H_nT) > 0.1)
        res = np.where(valid_mask, (D_nT * 3438.0) / H_nT, np.nan)
    return res

def clean_sqd_data_after_conversion(sqd_df: pd.DataFrame) -> pd.DataFrame:
    """Apply aggressive cleaning to SQD data after format conversion.

    Removes sentinel values and fat outliers from the H_raw and Z_raw
    components of converted SQD data.

    Args:
        sqd_df: DataFrame with H_raw and Z_raw columns from SQD conversion.

    Returns:
        Cleaned DataFrame with outliers replaced by NaN.
    """
    print('Aggressively cleaning SQD data to remove "fat" outliers.', flush=True)
    for comp in ['H_raw', 'Z_raw']:
        if comp in sqd_df.columns:
            original_count = sqd_df[comp].notna().sum()
            sqd_df[comp] = _mask_sentinels_and_extremes(sqd_df[comp])
            sqd_df[comp] = remove_fat_outliers(sqd_df[comp], window_size=101, threshold_factor=3)
            cleaned_count = sqd_df[comp].notna().sum()
            print(f'  {comp}: {original_count} -> {cleaned_count} valid points', flush=True)
    return sqd_df


def clean_ctu_data_after_conversion(ctu_df: pd.DataFrame) -> pd.DataFrame:
    """Clean CTU data after format conversion.

    Applies sentinel masking to H, D, and Z components of converted CTU data.

    Args:
        ctu_df: DataFrame with H, D, Z columns from CTU conversion.

    Returns:
        Cleaned DataFrame with sentinel values replaced by NaN.
    """
    print('Cleaning CTU data.', flush=True)
    for comp in ['H', 'D', 'Z']:
        if comp in ctu_df.columns:
            original_count = ctu_df[comp].notna().sum()
            print(f'Before cleaning {comp}: min={ctu_df[comp].min():.2f}, max={ctu_df[comp].max():.2f}', flush=True)
            ctu_df[comp] = _mask_sentinels_and_extremes(ctu_df[comp])
            cleaned_count = ctu_df[comp].notna().sum()
            print(f'  {comp}: {original_count} -> {cleaned_count} valid points', flush=True)
            if cleaned_count < original_count:
                print(f'  REMOVED {original_count - cleaned_count} points from {comp}!', flush=True)
    return ctu_df

# =============================================================================
# IAGA-2002 FILE OUTPUT FUNCTIONS
# =============================================================================


def writeHeader(
    fh,
    station_name: str,
    code: str,
    reported: str = 'XYZF',
    sensor: str = 'HZ',
    lat: str = '-34.42410',
    lon: str = '19.2230',
    elev: str = '14',
    instrument_type: str = 'SQD'
) -> None:
    """Write IAGA-2002 format header to file.

    Generates a compliant IAGA-2002 header with station metadata and
    instrument-specific comment lines.

    Args:
        fh: File handle open for writing.
        station_name: Full station name (e.g., 'Hermanus Magnetic Observatory').
        code: IAGA station code (e.g., 'HER').
        reported: Data component order (e.g., 'XYZF', 'HDZF', 'DHZF').
        sensor: Sensor orientation code.
        lat: Geodetic latitude as string.
        lon: Geodetic longitude as string.
        elev: Elevation in meters as string.
        instrument_type: 'SQD' or 'CTU' to determine comment format.
    """
    # Determine the comment based on the instrument type and format
    if instrument_type.upper() == 'CTU':
        comment_line = "#Calculated by user supplied filter."
    else:  # SQD
        if reported == 'XYZF':
            comment_line = "#A rotational matrix based on X, Y and Z HERDTU1 data is applied."
            elev='26'
        elif reported in ['HDZF', 'DHZF']:
            comment_line = "#Scaling factors were applied to H (0.725) and Z (0.800)."
        else:
            comment_line = "#"
    
    header = (
        "Format                 IAGA-2002                                     |\n"
        "Source of Data         SANSA Space Science                           |\n"
        "Station Name           {station:<46}|\n"
        "IAGA CODE              {code:<46}|\n"
        "Geodetic Latitude      {lat:<46}|\n"
        "Geodetic Longitude     {lon:<46}|\n"
        "Elevation              {elev:<46}|\n"
        "Reported               {reported:<46}|\n"
        "Sensor Orientation     {sensor:<46}|\n"
        "Digital Sampling                                                     |\n"
        "Data Interval Type     Filtered 1-second                             |\n"
        "Data Type              Variation                                     |\n"
        "#                                                                    |\n"
        "{comment:<69}|\n"
        "#                                                                    |\n"
    ).format(
        station=station_name, 
        code=code, 
        lat=lat, 
        lon=lon, 
        elev=elev, 
        reported=reported, 
        sensor=sensor,
        comment=comment_line
    )
    fh.write(header)

# =============================================================================
# FTP DATA DOWNLOAD FUNCTIONS
# =============================================================================


def download_from_ftp(instrument: str, date_str: str) -> Optional[str]:
    """Download raw SQD/CTU file from FTP server.

    Retrieves the appropriate data file for the specified instrument and date
    from the SANSA FTP server.

    Args:
        instrument: Instrument type ('SQD' or 'CTU').
        date_str: Date string in 'YYYY-MM-DD' format.

    Returns:
        Local path to downloaded file, or None if download failed.
    """
    print(f'[FTP] Downloading {instrument} file for {date_str}.', flush=True)
    ftp = get_ftp_handler()
    try:
        instrument_folder = "SQD"
        ext = "squid" if instrument.upper() == 'SQD' else "ctumag"
        filename = f"{date_str}.{ext}"
        temp_dir = tempfile.gettempdir()
        temp_local = os.path.join(temp_dir, filename)
        ok_path = ftp.download_file(f"{instrument_folder}/{filename}", temp_local)
        if ok_path:
            print(f"[FTP] Downloaded: {ok_path}", flush=True)
            return ok_path
        else:
            print(f"[FTP] Failed to fetch {instrument_folder}/{filename}", flush=True)
            return None
    except Exception as e:
        print(f"[FTP] Error fetching {instrument}: {e}", flush=True)
        return None

def load_herdtu1_data(date_str: str) -> Optional[pd.DataFrame]:
    """Load HERDTU1 reference data for a given date from FTP.

    Downloads and processes DTU1 magnetometer data from Hermanus station,
    which is used as reference data for SQD comparisons.

    Args:
        date_str: Date string in 'YYYY-MM-DD' format.

    Returns:
        DataFrame with HERDTU1 data including dateTime column,
        or None if data unavailable.
    """
    ftp = get_ftp_handler()
    try:
        # Convert date string to datetime
        date_dt = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Construct remote path for HERDTU1
        remote_path = ftp.construct_remote_path("HER", "DTU1", date_dt)

        if ftp.file_exists(remote_path):
            local_path = ftp.download_file(remote_path)
            if local_path:
                # Load DTU1 data
                df = pd.read_csv(local_path, sep=r"\s+", skiprows=15, engine="python")
                
                # DEBUG: Check what columns we have
                print(f"[DEBUG] HERDTU1 columns: {df.columns.tolist()}")
                print(f"[DEBUG] HERDTU1 shape: {df.shape}")
                if not df.empty:
                    print(f"[DEBUG] First few rows:\n{df.head(3)}")
                
                df["dateTime"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
                
                # MASK SENTINELS IN DTU DATA
                print(f"[DEBUG] Masking sentinels in HERDTU1 data")
                
                # Identify the magnetic field columns (HERD, HERH, HERZ)
                for col in df.columns:
                    if 'HERD' in col or 'HERH' in col or 'HERZ' in col:
                        original_count = df[col].notna().sum()
                        df[col] = _mask_sentinels_and_extremes(df[col])
                        cleaned_count = df[col].notna().sum()
                        print(f"[DEBUG] {col}: {original_count} -> {cleaned_count} after sentinel masking")
                '''        
                # APPLY BASELINES TO GET ABSOLUTE VALUES
                print(f"[DEBUG] Applying baseline values to get absolute components")
                H0, D0, Z0 = load_herdtu1_baseline()
                print(f"[DEBUG] Baseline values - H0: {H0}, D0: {D0}, Z0: {Z0}")
                
                # Apply baselines to get absolute values
                # Note: D0 is in arcmin, same as the data
                for col in df.columns:
                    if 'HERH' in col:
                        df[col] = df[col] + H0
                        print(f"[DEBUG] Applied H baseline: {H0} nT")
                    elif 'HERD' in col:
                        df[col] = df[col] + D0  
                        print(f"[DEBUG] Applied D baseline: {D0} arcmin")
                    elif 'HERZ' in col:
                        df[col] = df[col] + Z0
                        print(f"[DEBUG] Applied Z baseline: {Z0} nT")
                '''
                # Clean up temporary file
                if os.path.exists(local_path):
                    os.remove(local_path)
                
                return df
        else:
            print(f"[DEBUG] HERDTU1 file not found: {remote_path}")
    except Exception as e:
        print(f"Error loading HERDTU1 data for {date_str}: {e}")
    
    return None

def correct_sqd_orientation(
    sqd_h: np.ndarray,
    sqd_z: np.ndarray,
    date_str: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply rotation matrix to transform SQD data to geographic coordinates.

    Uses the R_SQUID rotation matrix to convert SQUID instrument frame
    measurements to geographic XYZ frame.

    Args:
        sqd_h: SQD H component values (treated as X in instrument frame).
        sqd_z: SQD Z component values.
        date_str: Date string for logging purposes.

    Returns:
        Tuple of (X_geo, Y_geo, Z_geo) arrays in geographic coordinates.
    """
    try:
        print(f"[DEBUG] Applying rotation matrix to SQD data for {date_str}")
        print(f"[DEBUG] Input H range: {np.nanmin(sqd_h):.2f} to {np.nanmax(sqd_h):.2f}")
        print(f"[DEBUG] Input Z range: {np.nanmin(sqd_z):.2f} to {np.nanmax(sqd_z):.2f}")
        
        # The rotation matrix transforms from SQUID instrument frame to geographic XYZ
        # We have H and Z from SQD, but we're missing the third component
        # Let's assume the SQD measures in a frame that's roughly aligned but needs correction
        
        # For now, let's just apply the matrix to see what happens
        # Stack the vectors for matrix multiplication: shape (3, n)
        # We'll use H as X_instrument, 0 as Y_instrument, Z as Z_instrument
        X_instr = sqd_h
        Y_instr = np.zeros_like(sqd_h)  # Missing component
        Z_instr = sqd_z
        
        original_instr = np.vstack([X_instr, Y_instr, Z_instr])
        
        # Apply rotation: geographic_XYZ = R * instrument_XYZ
        rotated_xyz = np.dot(R_SQUID, original_instr)
        
        # Extract rotated components in geographic frame
        X_geo = rotated_xyz[0, :]
        Y_geo = rotated_xyz[1, :] 
        Z_geo = rotated_xyz[2, :]
        
        print(f"[DEBUG] Rotated XYZ - X: {np.nanmean(X_geo):.2f}, Y: {np.nanmean(Y_geo):.2f}, Z: {np.nanmean(Z_geo):.2f}")
        print(f"[DEBUG] Output variations - X: {np.nanmin(X_geo):.2f} to {np.nanmax(X_geo):.2f}")
        print(f"[DEBUG] Output variations - Y: {np.nanmin(Y_geo):.2f} to {np.nanmax(Y_geo):.2f}")
        print(f"[DEBUG] Output variations - Z: {np.nanmin(Z_geo):.2f} to {np.nanmax(Z_geo):.2f}")
        
        return X_geo, Y_geo, Z_geo  # Fixed the typo here
        
    except Exception as e:
        print(f"Error in SQD rotation matrix correction: {e}")
        import traceback
        traceback.print_exc()
        return sqd_h, np.zeros_like(sqd_h), sqd_z

# =============================================================================
# SQD/CTU PROCESSING FUNCTIONS
# =============================================================================


def process_sqd_with_formats(
    filepath: str,
    out_dir: str,
    output_format: str = 'HDZF',
    compare_dtu: bool = True,
    save_file: bool = True,
    plot: bool = False
) -> Tuple[Optional[pd.DataFrame], Optional[plt.Figure]]:
    """Process SQD data file and convert to specified output format.

    Reads raw SQD data, aggregates to 1-second resolution, applies
    rotation matrix for XYZF format or uses HERDTU1 D component for
    HDZF/DHZF formats, and optionally saves to IAGA-2002 format.

    Args:
        filepath: Path to raw SQD data file.
        out_dir: Output directory for converted files.
        output_format: Target format ('HDZF', 'DHZF', or 'XYZF').
        compare_dtu: Whether to load HERDTU1 data for D component.
        save_file: Whether to save the converted data to disk.
        plot: Whether to generate comparison plots.

    Returns:
        Tuple of (DataFrame with converted data, matplotlib Figure or None).
    """
    try:
        # Read raw SQD data with error handling for inconsistent formatting
        print(f"[SQD] Reading file: {filepath}")
        try:
            # First try reading with standard 3 columns
            raw = pd.read_csv(filepath, header=None, sep=r'\s+', names=['Time', 'H', 'Z'])
        except pd.errors.ParserError as e:
            print(f"[SQD] Standard reading failed: {e}, trying flexible reading...")
            # If that fails, read with flexible column handling
            raw = pd.read_csv(filepath, header=None, sep=r'\s+', engine='python')
            # Keep only first 3 columns (Time, H, Z)
            if raw.shape[1] >= 3:
                raw = raw.iloc[:, :3]
                raw.columns = ['Time', 'H', 'Z']
            else:
                print(f"[SQD] File has insufficient columns: {raw.shape[1]}")
                return None, None
        
        print(f"[SQD] Successfully read {len(raw)} rows with {raw.shape[1]} columns")
        
    except Exception as e:
        print(f"[SQD] Failed to read {filepath}: {e}")
        return None, None

    fixTimeData(raw)
    raw[['H','Z']] = raw[['H','Z']].apply(pd.to_numeric, errors='coerce')
    raw = raw.dropna(subset=['Time'])

    dateStr = os.path.basename(filepath).split('.')[0]
    try:
        date = datetime.strptime(dateStr, '%Y-%m-%d')
    except Exception as e:
        print(f"[SQD] Bad filename date for {filepath}: {e}")
        return None, None
    doy = date.timetuple().tm_yday

    # Load HERDTU1 data for D component and comparison
    herdtu1_df = None
    if compare_dtu or output_format in ['HDZF', 'DHZF']:
        herdtu1_df = load_herdtu1_data(dateStr)
        if herdtu1_df is None:
            print(f"[WARNING] No HERDTU1 data found for {dateStr}")
            if output_format in ['HDZF', 'DHZF']:
                print(f"[ERROR] Cannot create {output_format} without HERDTU1 D component")
                return None, None

    # Aggregate SQD to 1-second
    raw['second'] = raw['Time'].astype(int)
    
    print(f"[SQD] Raw data: {len(raw)} rows, time range: {raw['Time'].min()} to {raw['Time'].max()}")
    
    grouped = raw.groupby('second').agg({
        'H': ['count', 'mean'],
        'Z': 'mean'
    }).reset_index()
    grouped.columns = ['second', 'count', 'H_mean', 'Z_mean']
    
    print(f"[SQD] After grouping: {len(grouped)} seconds with data")

    # Create full day DataFrame
    allSeconds = pd.DataFrame({'second': np.arange(86400)})
    final_merged = pd.merge(allSeconds, grouped, on='second', how='left')
    
    # Get SQD components
    sqd_h = final_merged['H_mean'].values
    sqd_z = final_merged['Z_mean'].values
    
    # Process based on output format
    if output_format == 'XYZF':
        # Apply rotation matrix to get XYZ
        print(f"[SQD] Applying rotation matrix for XYZ format")
        X_geo, Y_geo, Z_geo = correct_sqd_orientation(sqd_h, sqd_z, dateStr)
        
        # Ensure all arrays have exactly 86400 elements
        outdf = pd.DataFrame({
            'DATE': [dateStr] * 86400,
            'TIME': pd.to_datetime(np.arange(86400), unit='s').strftime('%H:%M:%S.000'),
            'DOY': [doy] * 86400,
            'X': X_geo,
            'Y': Y_geo,
            'Z': Z_geo,
            'F': [F_SENTINEL] * 86400
        })
        
    else:  # HDZF or DHZF
        # Get D component from HERDTU1
        if herdtu1_df is not None:
            # FIXED: Proper time calculation for HERDTU1
            print(f"[SQD] Aligning HERDTU1 D component for {output_format} format")
            
            # Convert TIME to seconds using string parsing (more reliable)
            def time_to_seconds(time_str):
                try:
                    # Handle formats like "00:00:00.000"
                    parts = time_str.split(':')
                    if len(parts) >= 3:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds = float(parts[2])
                        return int(hours * 3600 + minutes * 60 + seconds)
                    return 0
                except Exception as e:
                    print(f"Warning: Could not parse time '{time_str}': {e}")
                    return 0
            
            herdtu1_df['second'] = herdtu1_df['TIME'].apply(time_to_seconds)
            
            # Debug: Check the time conversion
            print(f"[DEBUG] HERDTU1 time range: {herdtu1_df['second'].min()} to {herdtu1_df['second'].max()} seconds")
            print(f"[DEBUG] HERDTU1 sample times: {herdtu1_df[['TIME', 'second']].head(3).to_dict('records')}")
            
            # Check what D column actually exists in HERDTU1 data
            d_column = None
            for col in herdtu1_df.columns:
                if 'D' in col.upper() and col.upper() not in ['DATE', 'TIME', 'DOY']:
                    d_column = col
                    break

            if d_column:
                print(f"[DEBUG] Using D column: {d_column}")
                
                # Ensure we get exactly 86400 D values
                # Create a mapping from second to D value, then map to allSeconds
                d_mapping = herdtu1_df.set_index('second')[d_column].to_dict()
                
                # Map the D values to all 86400 seconds, using NaN for missing seconds
                sqd_d = allSeconds['second'].map(d_mapping).values
                
                print(f"[SQD] Using HERDTU1 D component for {output_format} format")
                print(f"[DEBUG] D component stats: min={np.nanmin(sqd_d):.2f}, max={np.nanmax(sqd_d):.2f}, valid={np.sum(~np.isnan(sqd_d))} points")
            else:
                print(f"[WARNING] No D column found in HERDTU1 data. Available columns: {herdtu1_df.columns.tolist()}")
                sqd_d = np.full(86400, np.nan)
            
            print(f"[SQD] Using HERDTU1 D component for {output_format} format")
            print(f"[DEBUG] D component stats: min={np.nanmin(sqd_d):.2f}, max={np.nanmax(sqd_d):.2f}, valid={np.sum(~np.isnan(sqd_d))} points")
        else:
            sqd_d = np.full(len(allSeconds), np.nan)
            print(f"[SQD] No D component available for {output_format}")
        
        # FIXED: Ensure all arrays have exactly 86400 elements
        if output_format == 'HDZF':
            outdf = pd.DataFrame({
                'DATE': [dateStr] * 86400,
                'TIME': pd.to_datetime(np.arange(86400), unit='s').strftime('%H:%M:%S.000'),
                'DOY': [doy] * 86400,
                'H': sqd_h,
                'D': sqd_d,
                'Z': sqd_z,
                'F': [F_SENTINEL] * 86400
            })
        else:  # DHZF
            outdf = pd.DataFrame({
                'DATE': [dateStr] * 86400,
                'TIME': pd.to_datetime(np.arange(86400), unit='s').strftime('%H:%M:%S.000'),
                'DOY': [doy] * 86400,
                'D': sqd_d,
                'H': sqd_h,
                'Z': sqd_z,
                'F': [F_SENTINEL] * 86400
            })
    
    # Clean the data
    outdf = clean_sqd_data_after_conversion(outdf)
    
    fig = None
    if save_file:
        # Create format-specific subdirectory
        format_dir = output_format.replace('F', '')  # Remove 'F' from HDZF/DHZF/XYZF
        target_dir = os.path.join(out_dir, SQD_SUBDIR, format_dir)
        os.makedirs(target_dir, exist_ok=True)
        
        # Use consistent filename regardless of format
        outpath = os.path.join(target_dir, f"sqd{dateStr.replace('-', '')}vsec.sec")
        
        with open(outpath, 'w') as fh:
            if output_format == 'XYZF':
                writeHeader(fh, STATION_NAME, IAGA_CODE, reported='XYZF', sensor='HZ', instrument_type='SQD')
                headerFormat = "{:<10} {:<10} {:>5} {:>8} {:>9} {:>9} {:>9}"
                fh.write(headerFormat.format('DATE', 'TIME', 'DOY', "HERX", "HERY", "HERZ", "HERF") + '   |\n')
                rowFormat = "{:<10} {:<11} {:<5} {:>10.2f} {:>9.2f} {:>9.2f} {:>9.2f}\n"
                for idx, r in outdf.iterrows():
                    x_val = SEC_SENTINEL if np.isnan(r['X']) else r['X']
                    y_val = SEC_SENTINEL
                    z_val = SEC_SENTINEL if np.isnan(r['Z']) else r['Z']
                    fh.write(rowFormat.format(
                        r['DATE'], r['TIME'], int(r['DOY']),
                        x_val, y_val, z_val, F_SENTINEL
                    ))
            else:
                writeHeader(fh, STATION_NAME, IAGA_CODE, reported=output_format, sensor='HZ', instrument_type='SQD')
                if output_format == 'HDZF':
                    headerFormat = "{:<10} {:<10} {:>5} {:>8} {:>9} {:>9} {:>9}"
                    fh.write(headerFormat.format('DATE', 'TIME', 'DOY', 'HERH', 'HERD', 'HERZ', 'HERF') + '   |\n')
                    rowFormat = "{:<10} {:<11} {:<5} {:>10.2f} {:>9.2f} {:>9.2f} {:>9.2f}\n"
                    for idx, r in outdf.iterrows():
                        h_val = SEC_SENTINEL if np.isnan(r['H']) else r['H']
                        d_val = SEC_SENTINEL
                        z_val = SEC_SENTINEL if np.isnan(r['Z']) else r['Z']
                        fh.write(rowFormat.format(
                            r['DATE'], r['TIME'], int(r['DOY']),
                            h_val, d_val, z_val, F_SENTINEL
                        ))
                else:  # DHZF
                    headerFormat = "{:<10} {:<10} {:>5} {:>8} {:>9} {:>9} {:>9}"
                    fh.write(headerFormat.format('DATE', 'TIME', 'DOY', 'HERD', 'HERH', 'HERZ', 'HERF') + '   |\n')
                    rowFormat = "{:<10} {:<11} {:<5} {:>10.2f} {:>9.2f} {:>9.2f} {:>9.2f}\n"
                    for idx, r in outdf.iterrows():
                        d_val = SEC_SENTINEL
                        h_val = SEC_SENTINEL if np.isnan(r['H']) else r['H']
                        z_val = SEC_SENTINEL if np.isnan(r['Z']) else r['Z']
                        fh.write(rowFormat.format(
                            r['DATE'], r['TIME'], int(r['DOY']),
                            d_val, h_val, z_val, F_SENTINEL
                        ))
        
        print(f"[SQD] Saved {output_format} format: {outpath}")

    return outdf, fig  # Return only one figure

def plot_sqd_only(
    sqd_data_list: List[pd.DataFrame],
    date_range_str: str
) -> Optional[plt.Figure]:
    """Plot SQD data without DTU1 comparison.

    Creates a multi-panel plot showing SQD magnetic field component
    variations over the specified date range.

    Args:
        sqd_data_list: List of DataFrames containing SQD data for each day.
        date_range_str: Date range string for plot title formatting.

    Returns:
        Matplotlib Figure object, or None if no data to plot.
    """
    if not sqd_data_list:
        print("No SQD data to plot")
        return None
    
    # Combine all SQD data
    combined_sqd = pd.concat(sqd_data_list, ignore_index=True)
    
    # Create datetime index
    combined_sqd['dateTime'] = pd.to_datetime(combined_sqd['DATE'] + ' ' + combined_sqd['TIME'])
    
    # Set the exact time range for the entire period
    start_time = combined_sqd['dateTime'].min()
    end_time = combined_sqd['dateTime'].max()
    
    print(f"[SQD_SINGLE] Combined SQD data: {len(combined_sqd)} records from {start_time} to {end_time}")
    
    # Determine which components we have
    has_x = 'X' in combined_sqd.columns and combined_sqd['X'].notna().any()
    has_y = 'Y' in combined_sqd.columns and combined_sqd['Y'].notna().any() 
    has_z = 'Z' in combined_sqd.columns and combined_sqd['Z'].notna().any()
    has_h = 'H' in combined_sqd.columns and combined_sqd['H'].notna().any()
    has_d = 'D' in combined_sqd.columns and combined_sqd['D'].notna().any()
    
    # Create appropriate number of subplots
    components = []
    if has_x: components.append('X')
    if has_y: components.append('Y') 
    if has_z: components.append('Z')
    if has_h: components.append('H')
    if has_d: components.append('D')
    
    n_plots = len(components)
    if n_plots == 0:
        print("[SQD_SINGLE] No data to plot")
        return None
    
    fig, axs = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots), sharex=True)
    if n_plots == 1:
        axs = [axs]
    
    # Plot each component
    for i, comp in enumerate(components):
        # Remove means for variations
        comp_data = combined_sqd[comp] - combined_sqd[comp].mean()
        
        valid_mask = comp_data.notna()
        if valid_mask.any():
            axs[i].plot(combined_sqd.loc[valid_mask, 'dateTime'], comp_data[valid_mask], 
                       'b-', linewidth=1, label=f'SQD {comp}')
            axs[i].set_ylabel(f'{comp} Variations (nT)')
            axs[i].legend(loc='upper right')
            axs[i].grid(True, alpha=0.3)
            axs[i].minorticks_on()
            axs[i].set_xlim(start_time, end_time)
    
    # Format x-axis for multi-day range
    ax = axs[-1]
    span_days = (end_time - start_time).total_seconds() / 86400.0
    
    if span_days <= 1:
        # Format for single day
        span_secs = (end_time - start_time).total_seconds()
        max_ticks = 12
        raw_interval = span_secs / max_ticks
        
        nice_steps = [60, 300, 600, 900, 1200, 1800, 3600, 7200, 10800, 14400]
        interval = next(step for step in nice_steps if step >= raw_interval)
        
        if interval % 3600 == 0:
            hours = interval // 3600
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=hours))
        else:
            minutes = interval // 60
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=minutes))
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0)
            lbl.set_ha('center')
            lbl.set_va('center')

        axs[1].set_xlabel('Time (UTC)')
        ax.tick_params(axis='x', which='major', pad=12)

        start_dt_rounded = start_time.round(f'{interval}s')
        end_dt_rounded = end_time.round(f'{interval}s')
        ax.set_xlim(start_dt_rounded, end_dt_rounded)

        total_hours = (end_dt_rounded - start_dt_rounded).total_seconds() / 3600
        
        if total_hours.is_integer():
            hours_display = int(total_hours)
        else:
            hours_display = total_hours
        
        caption = f"{hours_display} hours (UTC) for {start_time.strftime('%Y/%m/%d')}"
        
    elif span_days <= 4:
        # Format for multi-day (up to 5 days)
        major_times = []
        current_date = start_time.normalize()
        end_date_normalized = end_time.normalize()
        
        while current_date <= end_date_normalized:
            # Add midnight
            if current_date >= start_time and current_date <= end_time:
                major_times.append(current_date)
            # Add 6am, noon, 6pm
            for hour in [6, 12, 18]:
                candidate = current_date + pd.Timedelta(hours=hour)
                if candidate >= start_time and candidate <= end_time:
                    major_times.append(candidate)
            current_date += pd.Timedelta(days=1)
        
        axs[-1].set_xlabel('Time (UTC)')
        ax.set_xticks(major_times)
        
        def fmt(x, pos):
            dt = mdates.num2date(x)
            if dt.hour == 0 and dt.minute == 0:
                return dt.strftime('%H:%M\n%d')
            else:
                return dt.strftime('%H:%M')
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))
        
        n_days = int(math.ceil(span_days))
        first_date = start_time.strftime('%Y/%m/%d')
        last_date = end_time.strftime('%Y/%m/%d')
        if first_date == last_date:
            caption = f"{n_days} Day on {first_date}"
        else:
            caption = f"{n_days} Days from {first_date} to {last_date}"
    else:
        # Format for longer ranges
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        
        n_days = int(math.ceil(span_days))
        first_date = start_time.strftime('%Y/%m/%d')
        last_date = end_time.strftime('%Y/%m/%d')
        if first_date == last_date:
            caption = f"{n_days} Day on {first_date}"
        else:
            caption = f"{n_days} Days from {first_date} to {last_date}"

    '''
    # Add caption
    ax.annotate(
        caption,
        xy=(0.5, -0.5),
        xycoords='axes fraction',
        ha='center',
        va='top',
        fontsize=12
    )
    '''
    
    # Use figure coordinates instead of axes coordinates to ensure visibility
    fig.text(
        0.5, 0.02,  # x=center, y=2% from bottom of figure
        caption,
        ha='center',
        va='bottom',
        fontsize=12
    )
    
    plt.suptitle(f"SQD Data - {date_range_str.replace('_to_', ' to ')}", fontsize=12, y=0.95)
    plt.subplots_adjust(hspace=0.3, bottom=0.15)
    
    return fig

def plot_sqd_range(
    sqd_data_list: List[pd.DataFrame],
    herdtu1_data_list: List[pd.DataFrame],
    date_range_str: str,
    output_format: str = 'HDZF'
) -> Optional[plt.Figure]:
    """Plot SQD vs HERDTU1 comparison for entire date range.

    Creates a combined comparison plot showing SQD data overlaid with
    HERDTU1 reference data, with correlation statistics.

    Args:
        sqd_data_list: List of DataFrames containing SQD data for each day.
        herdtu1_data_list: List of DataFrames containing HERDTU1 reference data.
        date_range_str: Date range string for plot title formatting.
        output_format: Output format determining which components to compare.

    Returns:
        Matplotlib Figure object, or None if insufficient data.
    """
    if not sqd_data_list or not herdtu1_data_list:
        print("No SQD or HERDTU1 data to plot")
        return None
    
    # Combine all data
    combined_sqd = pd.concat(sqd_data_list, ignore_index=True)
    combined_herdtu1 = pd.concat(herdtu1_data_list, ignore_index=True)
    
    # ENSURE SENTINELS ARE MASKED IN COMBINED DATA
    print(f"[RANGE_PLOT] Masking sentinels in combined data")
    
    # Mask sentinels in combined SQD data - ONLY numeric columns
    for col in combined_sqd.columns:
        if col not in ['DATE', 'TIME', 'DOY'] and combined_sqd[col].dtype in [np.float64, np.int64]:
            original_count = combined_sqd[col].notna().sum()
            combined_sqd[col] = _mask_sentinels_and_extremes(combined_sqd[col])
            cleaned_count = combined_sqd[col].notna().sum()
            if cleaned_count < original_count:
                print(f"[RANGE_PLOT] Masked {original_count - cleaned_count} sentinels in SQD {col}")
    
    # Mask sentinels in combined DTU data - ONLY magnetic field columns
    magnetic_columns = []
    for col in combined_herdtu1.columns:
        if any(mag_col in col.upper() for mag_col in ['HERD', 'HERH', 'HERZ', 'H', 'D', 'Z']):
            magnetic_columns.append(col)
    
    for col in magnetic_columns:
        if col in combined_herdtu1.columns and combined_herdtu1[col].dtype in [np.float64, np.int64]:
            original_count = combined_herdtu1[col].notna().sum()
            combined_herdtu1[col] = _mask_sentinels_and_extremes(combined_herdtu1[col])
            cleaned_count = combined_herdtu1[col].notna().sum()
            if cleaned_count < original_count:
                print(f"[RANGE_PLOT] Masked {original_count - cleaned_count} sentinels in DTU {col}")
    
    # Rename HERDTU1 columns to match expected names
    combined_herdtu1 = combined_herdtu1.rename(columns={
        'HERD': 'D',
        'HERH': 'H', 
        'HERZ': 'Z'
    })
    
    # Create datetime indices
    combined_sqd['dateTime'] = pd.to_datetime(combined_sqd['DATE'] + ' ' + combined_sqd['TIME'])
    combined_herdtu1['dateTime'] = pd.to_datetime(combined_herdtu1['DATE'] + ' ' + combined_herdtu1['TIME'])
    
    # Set the exact time range for the entire period
    start_time = min(combined_sqd['dateTime'].min(), combined_herdtu1['dateTime'].min())
    end_time = max(combined_sqd['dateTime'].max(), combined_herdtu1['dateTime'].max())
    
    print(f"[SQD_RANGE] Combined SQD data: {len(combined_sqd)} records")
    print(f"[SQD_RANGE] Combined HERDTU1 data: {len(combined_herdtu1)} records")
    print(f"[SQD_RANGE] Time range: {start_time} to {end_time}")
    print(f"[SQD_RANGE] Plotting in {output_format} format")
    
    # Create figure with proper sizing
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # --- PLOT BASED ON SELECTED FORMAT ---
    if output_format == 'XYZF':
        # For XYZF format: plot X and Z components (already rotated)
        print(f"[SQD_RANGE] Using pre-rotated X and Z components for XYZ format")
        
        # Use the already rotated X and Z components from SQD
        sqd_x = combined_sqd['X'].values
        sqd_z = combined_sqd['Z'].values
        
        # Compute HERDTU1 X component from H and D
        D_rad = np.radians(combined_herdtu1['D'] / 60.0)
        herdtu1_x = combined_herdtu1['H'] * np.cos(D_rad)
        
        # Remove means for variations
        sqd_x_var = sqd_x - np.nanmean(sqd_x)
        sqd_z_var = sqd_z - np.nanmean(sqd_z)
        herdtu1_x_var = herdtu1_x - np.nanmean(herdtu1_x)
        herdtu1_z_var = combined_herdtu1['Z'] - np.nanmean(combined_herdtu1['Z'])
        
        # Plot X component
        ylabel1 = 'X Variations (nT)'
        comp1 = 'X'
        data1_sqd = sqd_x_var
        data1_dtu = herdtu1_x_var
        
    else:  # HDZF or DHZF format
        # For HDZF/DHZF format: plot H and Z components (NO rotation)
        print(f"[SQD_RANGE] Using H and Z components directly for HDZ format")
        
        # Get SQD H and Z directly (no rotation)
        sqd_h = combined_sqd['H'].values if 'H' in combined_sqd.columns else combined_sqd['H_raw'].values
        sqd_z = combined_sqd['Z'].values if 'Z' in combined_sqd.columns else combined_sqd['Z_raw'].values
        
        # Remove means for variations
        sqd_h_var = sqd_h - np.nanmean(sqd_h)
        sqd_z_var = sqd_z - np.nanmean(sqd_z)
        herdtu1_h_var = combined_herdtu1['H'] - np.nanmean(combined_herdtu1['H'])
        herdtu1_z_var = combined_herdtu1['Z'] - np.nanmean(combined_herdtu1['Z'])
        
        # --- ADD SCALING FACTORS HERE ---
        # Reduce SQD amplitude to match DTU1 better
        h_scale_factor = 0.725  
        z_scale_factor = 0.800  

        # Apply scaling to SQD variations only
        sqd_h_var = sqd_h_var * h_scale_factor
        sqd_z_var = sqd_z_var * z_scale_factor
        
        # Plot H component
        ylabel1 = 'H Variations (nT)'
        comp1 = 'H'
        data1_sqd = sqd_h_var
        data1_dtu = herdtu1_h_var
    
    # Common plotting logic for both formats
    # Plot 1: SQD Component 1 vs HERDTU1 Component 1
    sqd_data1 = pd.DataFrame({
        'time': combined_sqd['dateTime'],
        f'sqd_{comp1.lower()}': data1_sqd
    }).set_index('time')
    
    dtu_data1 = pd.DataFrame({
        'time': combined_herdtu1['dateTime'],
        f'dtu_{comp1.lower()}': data1_dtu
    }).set_index('time')
    
    combined_comp1 = pd.merge(sqd_data1, dtu_data1, left_index=True, right_index=True, how='inner')
    
    if len(combined_comp1) > 0:
        axs[0].plot(combined_comp1.index, combined_comp1[f'sqd_{comp1.lower()}'], 'b-', label=f'SQD {comp1}', linewidth=1)
        axs[0].plot(combined_comp1.index, combined_comp1[f'dtu_{comp1.lower()}'], 'r-', label=f'DTU1 {comp1}', linewidth=1, alpha=0.7)
        axs[0].set_ylabel(ylabel1)
        axs[0].legend(loc='upper right')
        axs[0].grid(True, alpha=0.3)
        axs[0].minorticks_on()
        axs[0].set_xlim(start_time, end_time)
        
        # Calculate correlation
        valid_mask = combined_comp1[[f'sqd_{comp1.lower()}', f'dtu_{comp1.lower()}']].notna().all(axis=1)
        if valid_mask.sum() > 0:
            corr_comp1 = np.corrcoef(combined_comp1.loc[valid_mask, f'sqd_{comp1.lower()}'], 
                                   combined_comp1.loc[valid_mask, f'dtu_{comp1.lower()}'])[0,1]
            axs[0].text(0.02, 0.98, f'Correlation: {corr_comp1:.4f}', 
                       transform=axs[0].transAxes, fontsize=11,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: SQD Z vs HERDTU1 Z (common for both formats)
    sqd_Z_data = pd.DataFrame({
        'time': combined_sqd['dateTime'],
        'sqd_z': sqd_z_var
    }).set_index('time')
    
    dtu_Z_data = pd.DataFrame({
        'time': combined_herdtu1['dateTime'],
        'dtu_z': herdtu1_z_var
    }).set_index('time')
    
    combined_z = pd.merge(sqd_Z_data, dtu_Z_data, left_index=True, right_index=True, how='inner')
    
    if len(combined_z) > 0:
        axs[1].plot(combined_z.index, combined_z['sqd_z'], 'b-', label='SQD Z', linewidth=1)
        axs[1].plot(combined_z.index, combined_z['dtu_z'], 'r-', label='DTU1 Z', linewidth=1, alpha=0.7)
        axs[1].set_ylabel('Z Variations (nT)')
        axs[1].set_xlabel('Time (UTC)')
        axs[1].legend(loc='upper right')
        axs[1].grid(True, alpha=0.3)
        axs[1].minorticks_on()
        axs[1].set_xlim(start_time, end_time)
        
        # Calculate correlation
        valid_mask = combined_z[['sqd_z', 'dtu_z']].notna().all(axis=1)
        if valid_mask.sum() > 0:
            corr_z = np.corrcoef(combined_z.loc[valid_mask, 'sqd_z'], 
                               combined_z.loc[valid_mask, 'dtu_z'])[0,1]
            axs[1].text(0.02, 0.98, f'Correlation: {corr_z:.4f}', 
                       transform=axs[1].transAxes, fontsize=11,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Format x-axis for multi-day range
    ax = axs[-1]
    span_days = (end_time - start_time).total_seconds() / 86400.0
    
    if span_days <= 1:
        # Format for single day
        span_secs = (end_time - start_time).total_seconds()
        max_ticks = 12
        raw_interval = span_secs / max_ticks
        
        nice_steps = [60, 300, 600, 900, 1200, 1800, 3600, 7200, 10800, 14400]
        interval = next(step for step in nice_steps if step >= raw_interval)
        
        if interval % 3600 == 0:
            hours = interval // 3600
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=hours))
        else:
            minutes = interval // 60
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=minutes))
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0)
            lbl.set_ha('center')
            lbl.set_va('center')

        axs[1].set_xlabel('Time (UTC)')
        ax.tick_params(axis='x', which='major', pad=12)

        start_dt_rounded = start_time.round(f'{interval}s')
        end_dt_rounded = end_time.round(f'{interval}s')
        ax.set_xlim(start_dt_rounded, end_dt_rounded)

        total_hours = (end_dt_rounded - start_dt_rounded).total_seconds() / 3600
        
        if total_hours.is_integer():
            hours_display = int(total_hours)
        else:
            hours_display = total_hours
        
        caption = f"{hours_display} hours (UTC) for {start_time.strftime('%Y/%m/%d')}"
        
    elif span_days <= 4:
        # Format for multi-day (up to 5 days)
        major_times = []
        current_date = start_time.normalize()
        end_date_normalized = end_time.normalize()
        
        while current_date <= end_date_normalized:
            # Add midnight
            if current_date >= start_time and current_date <= end_time:
                major_times.append(current_date)
            # Add 6am, noon, 6pm
            for hour in [6, 12, 18]:
                candidate = current_date + pd.Timedelta(hours=hour)
                if candidate >= start_time and candidate <= end_time:
                    major_times.append(candidate)
            current_date += pd.Timedelta(days=1)
        
        axs[-1].set_xlabel('Time (UTC)')
        ax.set_xticks(major_times)
        
        def fmt(x, pos):
            dt = mdates.num2date(x)
            if dt.hour == 0 and dt.minute == 0:
                return dt.strftime('%H:%M\n%d')
            else:
                return dt.strftime('%H:%M')
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))
        
        n_days = int(math.ceil(span_days))
        first_date = start_time.strftime('%Y/%m/%d')
        last_date = end_time.strftime('%Y/%m/%d')
        if first_date == last_date:
            caption = f"{n_days} Day on {first_date}"
        else:
            caption = f"{n_days} Days from {first_date} to {last_date}"
    else:
        # Format for longer ranges
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        
        n_days = int(math.ceil(span_days))
        first_date = start_time.strftime('%Y/%m/%d')
        last_date = end_time.strftime('%Y/%m/%d')
        if first_date == last_date:
            caption = f"{n_days} Day on {first_date}"
        else:
            caption = f"{n_days} Days from {first_date} to {last_date}"

    '''
    # Add caption
    ax.annotate(
        caption,
        xy=(0.5, -0.5),
        xycoords='axes fraction',
        ha='center',
        va='top',
        fontsize=12
    )
    '''
    
    # Use figure coordinates instead of axes coordinates to ensure visibility
    fig.text(
        0.5, 0.02,  # x=center, y=2% from bottom of figure
        caption,
        ha='center',
        va='bottom',
        fontsize=12
    )
        
    # Set main title
    if "to" in date_range_str:
        title_date = date_range_str.replace("_to_", " to ")
    else:
        title_date = date_range_str
        
    plt.suptitle(f"SQD vs DTU1", fontsize=12, y=0.95)
    plt.subplots_adjust(hspace=0.3, bottom=0.12)
    
    print(f"[SQD_RANGE] {comp1.lower()} correlation: {corr_comp1:.4f}")
    print(f"[SQD_RANGE] Z correlation: {corr_z:.4f}")
    
    return fig
######################################################
######################################################
######################################################
######################################################
######################################################
######################################################




# --- conversion functions (SQD/CTU) ---

def process_ctu_file(
    filepath: str,
    out_dir: str,
    output_format: str = 'HDZF',
    save_file: bool = True,
    plot: bool = False,
    date_str: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[plt.Figure]]:
    """Process CTU magnetometer data file and convert to IAGA-2002 format.

    Reads raw CTU data, converts D component to arc-minutes, aggregates
    to 1-second resolution, and optionally saves in IAGA-2002 format.

    Args:
        filepath: Path to raw CTU data file.
        out_dir: Output directory for converted files.
        output_format: Target format ('HDZF' or 'DHZF').
        save_file: Whether to save the converted data to disk.
        plot: Whether to generate plots (not implemented).
        date_str: Optional date string override; if None, extracted from filename.

    Returns:
        Tuple of (DataFrame with converted data, Figure or None).
    """
    try:
        data = pd.read_csv(filepath, header=None, sep=r'\s+', names=['Time', 'H', 'D', 'Z'])
    except Exception as e:
        print(f"[CTU] Failed to read {filepath}: {e}", flush=True)
        return None, None

    # Use the same time conversion
    fixTimeData(data)
    data[['H','D','Z']] = data[['H','D','Z']].apply(pd.to_numeric, errors='coerce')

    if date_str is not None:
        dateStr = date_str
    else:
        dateStr = os.path.basename(filepath).split('.')[0]

    try:
        date = datetime.strptime(dateStr, '%Y-%m-%d')
    except Exception as e:
        print(f"[CTU] Bad filename date: {e}", flush=True)
        return None, None
        
    doy = date.timetuple().tm_yday

    # SIMPLE AGGREGATION (like dataprocessv2.py)
    data['second'] = data['Time'].astype(int)
    grouped = data.groupby('second').agg({
        'H': ['count', 'mean'],
        'D': 'mean',
        'Z': 'mean'
    }).reset_index()
    grouped.columns = ['second', 'count', 'H_mean', 'D_mean', 'Z_mean']

    # Convert D (nT) -> arcmin
    Hvals = grouped['H_mean'].to_numpy()
    Dvals_nT = grouped['D_mean'].to_numpy()
    D_arcmin = d_to_arcmin(Dvals_nT, Hvals)

    # Fill missing seconds
    allSeconds = pd.DataFrame({'second': np.arange(86400)})
    final_merged = pd.merge(allSeconds, grouped, on='second', how='left')
    
    # Apply D conversion to valid rows
    final_merged['D_arcmin'] = SEC_SENTINEL  # Default to sentinel
    valid_mask = final_merged['second'].isin(grouped['second'])
    if valid_mask.any():
        second_to_d = dict(zip(grouped['second'], D_arcmin))
        final_merged.loc[valid_mask, 'D_arcmin'] = final_merged.loc[valid_mask, 'second'].map(second_to_d)

    # Fill missing values
    final_merged['H_mean'] = final_merged['H_mean'].fillna(SEC_SENTINEL)
    final_merged['Z_mean'] = final_merged['Z_mean'].fillna(SEC_SENTINEL)

    # Create output
    final_merged['TIME'] = pd.to_datetime(final_merged['second'], unit='s').dt.strftime('%H:%M:%S.000')
    final_merged['DATE'] = dateStr
    final_merged['DOY'] = doy

    # Create output DataFrame based on format
    if output_format == 'HDZF':
        outdf = pd.DataFrame({
            'DATE': final_merged['DATE'],
            'TIME': final_merged['TIME'], 
            'DOY': final_merged['DOY'],
            'H': final_merged['H_mean'],
            'D': final_merged['D_arcmin'],
            'Z': final_merged['Z_mean'],
            'F': F_SENTINEL
        })
    else:  # DHZF
        outdf = pd.DataFrame({
            'DATE': final_merged['DATE'],
            'TIME': final_merged['TIME'], 
            'DOY': final_merged['DOY'],
            'D': final_merged['D_arcmin'],
            'H': final_merged['H_mean'],
            'Z': final_merged['Z_mean'],
            'F': F_SENTINEL
        })

    # NO CLEANING - preserve all aggregated data
    
    if save_file:
        # Create format-specific subdirectory
        format_dir = output_format.replace('F', '')  # Remove 'F' from HDZF/DHZF
        target_dir = os.path.join(out_dir, CTU_SUBDIR, format_dir)
        os.makedirs(target_dir, exist_ok=True)
        
        outpath = os.path.join(target_dir, f"ctu{dateStr.replace('-', '')}vsec.sec")
        
        with open(outpath, 'w') as fh:
            writeHeader(fh, STATION_NAME, IAGA_CODE, reported=output_format, sensor='HDZ', instrument_type='CTU')
            
            if output_format == 'HDZF':
                headerFormat = "{:<10} {:<10} {:>5} {:>8} {:>9} {:>9} {:>9}"
                fh.write(headerFormat.format('DATE', 'TIME', 'DOY', 'HERH', 'HERD', 'HERZ', 'HERF') + '   |\n')
                rowFormat = "{:<10} {:<11} {:<5} {:>10.2f} {:>9.2f} {:>9.2f} {:>9.2f}\n"
                for idx, r in outdf.iterrows():
                    fh.write(rowFormat.format(r['DATE'], r['TIME'], int(r['DOY']), r['H'], r['D'], r['Z'], F_SENTINEL))
            else:  # DHZF
                headerFormat = "{:<10} {:<10} {:>5} {:>8} {:>9} {:>9} {:>9}"
                fh.write(headerFormat.format('DATE', 'TIME', 'DOY', 'HERD', 'HERH', 'HERZ', 'HERF') + '   |\n')
                rowFormat = "{:<10} {:<11} {:<5} {:>10.2f} {:>9.2f} {:>9.2f} {:>9.2f}\n"
                for idx, r in outdf.iterrows():
                    fh.write(rowFormat.format(r['DATE'], r['TIME'], int(r['DOY']), r['D'], r['H'], r['Z'], F_SENTINEL))
                    
        print(f"[CTU] Saved {output_format} format: {outpath}", flush=True)

    return outdf, None


# ---------- NEW: Multi-day plotting functions ----------
def plot_ctu_range(ctu_data_list, date_range_str):
    """
    Plot CTU data for entire date range with MEANS REMOVED (H, D, Z variations)
    This creates a SINGLE combined plot for all days
    """
    if not ctu_data_list:
        print("No CTU data to plot")
        return None
    
    # Combine all CTU data
    combined_ctu = pd.concat(ctu_data_list, ignore_index=True)
    
    # Create datetime index
    combined_ctu['dateTime'] = pd.to_datetime(combined_ctu['DATE'] + ' ' + combined_ctu['TIME'])
    
    # Set the exact time range for the entire period
    start_time = combined_ctu['dateTime'].min()
    end_time = combined_ctu['dateTime'].max()
    
    print(f"[CTU_RANGE] Combined data: {len(combined_ctu)} records from {start_time} to {end_time}")
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # --- REMOVE MEANS FOR VARIATIONS ---
    h_mean = np.nanmean(combined_ctu['H'])
    d_mean = np.nanmean(combined_ctu['D'])
    z_mean = np.nanmean(combined_ctu['Z'])
    
    print(f"[CTU_RANGE] H mean: {h_mean:.2f} nT, removing mean for variations")
    print(f"[CTU_RANGE] D mean: {d_mean:.2f} arcmin, removing mean for variations")
    print(f"[CTU_RANGE] Z mean: {z_mean:.2f} nT, removing mean for variations")
    
    # Remove means to get variations
    h_var = combined_ctu['H'] - h_mean
    d_var = combined_ctu['D'] - d_mean
    z_var = combined_ctu['Z'] - z_mean
    
    print(f"[CTU_RANGE] H variations: {h_var.min():.2f} to {h_var.max():.2f} nT")
    print(f"[CTU_RANGE] D variations: {d_var.min():.2f} to {d_var.max():.2f} arcmin")
    print(f"[CTU_RANGE] Z variations: {z_var.min():.2f} to {z_var.max():.2f} nT")
    
    # Plot H component variations
    valid_h = h_var.notna()
    if valid_h.any():
        axs[0].plot(combined_ctu.loc[valid_h, 'dateTime'], h_var[valid_h], 'b-', linewidth=1, label='H variations')
        axs[0].set_ylabel('H Variations (nT)')
        axs[0].legend(loc='upper right')
        axs[0].grid(True, alpha=0.3)
        axs[0].minorticks_on()
        axs[0].set_xlim(start_time, end_time)
    
    # Plot D component variations (arcmin)
    valid_d = d_var.notna()
    if valid_d.any():
        axs[1].plot(combined_ctu.loc[valid_d, 'dateTime'], d_var[valid_d], 'g-', linewidth=1, label='D variations')
        axs[1].set_ylabel('D (arcmin)')
        axs[1].legend(loc='upper right')
        axs[1].grid(True, alpha=0.3)
        axs[1].minorticks_on()
        axs[1].set_xlim(start_time, end_time)
    
    # Plot Z component variations
    valid_z = z_var.notna()
    if valid_z.any():
        axs[2].plot(combined_ctu.loc[valid_z, 'dateTime'], z_var[valid_z], 'r-', linewidth=1, label='Z variations')
        axs[2].set_ylabel('Z Variations (nT)')
        #axs[2].set_xlabel('Time (UTC)')
        axs[2].legend(loc='upper right')
        axs[2].grid(True, alpha=0.3)
        axs[2].minorticks_on()
        axs[2].set_xlim(start_time, end_time)
    
    # Format x-axis for multi-day range (copied from plotSingleOnGUI_FTP.py)
    ax = axs[-1]
    span_days = (end_time - start_time).total_seconds() / 86400.0
    
    if span_days <= 1:
        # Format for single day
        span_secs = (end_time - start_time).total_seconds()
        max_ticks = 12
        raw_interval = span_secs / max_ticks
        
        nice_steps = [60, 300, 600, 900, 1200, 1800, 3600, 7200, 10800, 14400]
        interval = next(step for step in nice_steps if step >= raw_interval)
        
        if interval % 3600 == 0:
            hours = interval // 3600
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=hours))
        else:
            minutes = interval // 60
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=minutes))
        
        axs[2].set_xlabel('Time (UTC)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0)
            lbl.set_ha('center')
            lbl.set_va('center')

        ax.tick_params(axis='x', which='major', pad=12)

        start_dt_rounded = start_time.round(f'{interval}s')
        end_dt_rounded = end_time.round(f'{interval}s')
        ax.set_xlim(start_dt_rounded, end_dt_rounded)

        total_hours = (end_dt_rounded - start_dt_rounded).total_seconds() / 3600
        
        if total_hours.is_integer():
            hours_display = int(total_hours)
        else:
            hours_display = total_hours
        
        caption = f"{hours_display} hours (UTC) for {start_time.strftime('%Y/%m/%d')}"
        
    elif span_days <= 4:
        # Format for multi-day (up to 5 days)
        major_times = []
        current_date = start_time.normalize()
        end_date_normalized = end_time.normalize()
        
        while current_date <= end_date_normalized:
            # Add midnight
            if current_date >= start_time and current_date <= end_time:
                major_times.append(current_date)
            # Add 6am, noon, 6pm
            for hour in [6, 12, 18]:
                candidate = current_date + pd.Timedelta(hours=hour)
                if candidate >= start_time and candidate <= end_time:
                    major_times.append(candidate)
            current_date += pd.Timedelta(days=1)
        
        axs[2].set_xlabel('Time (UTC)')
        ax.set_xticks(major_times)
        
        def fmt(x, pos):
            dt = mdates.num2date(x)
            if dt.hour == 0 and dt.minute == 0:
                return dt.strftime('%H:%M\n%d')
            else:
                return dt.strftime('%H:%M')
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))
        
        n_days = int(math.ceil(span_days))
        first_date = start_time.strftime('%Y/%m/%d')
        last_date = end_time.strftime('%Y/%m/%d')
        if first_date == last_date:
            caption = f"{n_days} Day on {first_date}"
        else:
            caption = f"{n_days} Days from {first_date} to {last_date}"
    else:
        # Format for longer ranges
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        
        n_days = int(math.ceil(span_days))
        first_date = start_time.strftime('%Y/%m/%d')
        last_date = end_time.strftime('%Y/%m/%d')
        if first_date == last_date:
            caption = f"{n_days} Day on {first_date}"
        else:
            caption = f"{n_days} Days from {first_date} to {last_date}"

    # Add caption
    ax.annotate(
        caption,
        xy=(0.5, -0.5),
        xycoords='axes fraction',
        ha='center',
        va='top',
        fontsize=12
    )
    
    # Set main title
    if "to" in date_range_str:
        title_date = date_range_str.replace("_to_", " to ")
    else:
        title_date = date_range_str
        
    plt.suptitle(f"CTU Magnetometer", fontsize=14, y=0.95)
    plt.subplots_adjust(hspace=0.3, bottom=0.15)
    
    print(f"[CTU_RANGE] Combined plot created with {valid_h.sum()} H, {valid_d.sum()} D, {valid_z.sum()} Z variation points")
    
    return fig

# ---------- END ADD ----------


################################################################################################################################



_INSTR_ALIAS = {
    'L251': 'L025'
}

# Load baseline data once at module level
with open("baselineValues.json", "r") as _f:
    _BASELINE_DATA = json.load(_f)

def loadBaselineValues(station: str, instrument: str) -> tuple[float,float,float]:
    inst_key = instrument.upper()
    inst_key = _INSTR_ALIAS.get(inst_key, inst_key)
    try:
        entry = _BASELINE_DATA[station.upper()][inst_key]
    except KeyError:
        raise KeyError(f"Missing baseline for {station}/{inst_key}")
    return float(entry["H0"]), float(entry["D0"]), float(entry["Z0"])

def debug_ftp_paths(station, instrument, date):
    """Debug function to check FTP paths"""
    ftp_handler = get_ftp_handler()
    remote_path = ftp_handler.construct_remote_path(station, instrument, date)
    exists = ftp_handler.file_exists(remote_path)
    
    print(f"DEBUG: Station: {station}, Instrument: {instrument}, Date: {date}")
    print(f"DEBUG: Remote path: {remote_path}")
    print(f"DEBUG: File exists: {exists}")
    
    # Try to list parent directory to see what's available
    if not exists:
        try:
            parent_dir = '/'.join(remote_path.split('/')[:-1])
            print(f"DEBUG: Parent directory: {parent_dir}")
            contents = ftp_handler.list_directory(parent_dir)
            print(f"DEBUG: Contents of parent directory: {contents}")
        except Exception as e:
            print(f"DEBUG: Could not list parent directory: {e}")
    
    return remote_path, exists

def _mask_sentinels_and_extremes(s: pd.Series,
                                 sentinels=(88888, 99999, 99999.99),
                                 extreme_threshold=1e6,
                                 sentinel_floor=90000.0) -> pd.Series:
    """Fast sentinel masking using vectorized operations"""
    if s is None or len(s) == 0:
        return s
        
    out = s.copy().astype(float)
    
    # Vectorized sentinel masking
    mask = np.zeros(len(out), dtype=bool)
    for code in sentinels:
        mask |= (out == code)
    mask |= (out.abs() >= float(sentinel_floor))
    mask |= (out.abs() > extreme_threshold)
    
    out[mask] = np.nan
    return out

def clean_component_at_original_resolution(series: pd.Series) -> pd.Series:
    """Clean at original resolution before any aggregation - LESS AGGRESSIVE"""
    if series is None or len(series) == 0:
        return series

    # Step 1: Remove obvious sentinels and extremes ONLY
    s = _mask_sentinels_and_extremes(series)
    
    # Step 2: Make outlier detection MUCH less aggressive (only extreme outliers)
    if s.notna().sum() > 100:  # Only if we have substantial data
        median_val = s.median()
        mad = (s - median_val).abs().median()
        
        # Use much wider thresholds (20 MADs instead of 5)
        lower_bound = median_val - 20 * mad
        upper_bound = median_val + 20 * mad
        
        # Only remove extreme outliers
        extreme_mask = (s < lower_bound) | (s > upper_bound)
        if extreme_mask.sum() > 0:
            print(f"  Removed {extreme_mask.sum()} extreme outliers (bounds: {lower_bound:.1f} to {upper_bound:.1f})", flush=True)
            s[extreme_mask] = np.nan
    
    return s


def minute_aggregate_clean_data(df, ts_col='dateTime'):
    """Aggregate to minutes AFTER cleaning at original resolution"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[ts_col])

    d = df.copy()
    
    if ts_col not in d.columns and d.index.name == ts_col:
        d = d.reset_index()

    if ts_col not in d.columns:
        raise KeyError(f"minute_aggregate_clean_data: no '{ts_col}' column or index in dataframe")

    d[ts_col] = pd.to_datetime(d[ts_col], errors='coerce')
    d = d.set_index(ts_col)
    
    cols = [c for c in ('MAGH', 'MAGD', 'MAGZ', 'F_meas') if c in d.columns]
    
    if not cols:
        res_idx = d.resample('60s', label='left', closed='left').apply(lambda s: np.nan)
        return res_idx.reset_index()[[ts_col]]

    def fast_mean(series):
        vals = series.values
        valid_vals = vals[~np.isnan(vals)]
        return np.nan if len(valid_vals) == 0 else np.mean(valid_vals)

    # Aggregate cleaned data
    res = d[cols].resample('60s', label='left', closed='left').apply(fast_mean)
    res = res.reset_index()
    res[ts_col] = pd.to_datetime(res[ts_col], errors='coerce')
    return res

def load_fgm(
    path: str,
    station_name: Optional[str] = None,
    instrument_name: Optional[str] = None
) -> pd.DataFrame:
    """Load FGM (fluxgate magnetometer) data file.

    Handles various FGM file formats with robust parsing of header
    lines and data records.

    Args:
        path: Path to FGM data file.
        station_name: Station code override.
        instrument_name: Instrument name override.

    Returns:
        DataFrame with dateTime, MAGH, MAGD, MAGZ columns.

    Raises:
        RuntimeError: If no valid data found in file.
    """
    filename = os.path.basename(path).upper()
    inst = (
        instrument_name or
        ('FGM1' if 'FGM1' in filename else
         'FGM2' if 'FGM2' in filename else
         os.path.basename(os.path.dirname(path)).upper())
    )
    station = (
        station_name or
        ('SNA' if 'SNA' in filename else
         os.path.basename(os.path.dirname(os.path.dirname(path))).upper())
    )

    print(f'[DEBUG] FGM loader: station={station}, instrument={inst}')

    scale = 40.0 if inst == 'FGM2' else 150.0

    if station == 'SNA' and inst == 'FGM1':
        scale_H, scale_D, scale_Z = 320.0, -320.0, -320.0
        print('[DEBUG] Applied SANAE FGM1 scaling (320, -320, -320 nT/V)')
    else:
        scale_H = scale_D = scale
        scale_Z = -scale

    records = []
    cur_base = None
    cur_min = None
    error_count = 0
    max_errors_to_log = 12

    float_re = re.compile(r'[+-]?\d+\.\d+(?:[eE][+-]?\d+)?|[+-]?\d+')

    with open(path, 'r', errors='replace') as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue

            hdr = line.split()
            if len(hdr) == 3 and hdr[0].isdigit():
                try:
                    year, doy, mcount = map(int, hdr)
                    cur_base = datetime(year, 1, 1) + timedelta(days=doy - 1)
                    cur_min = int(mcount)
                except Exception:
                    if error_count < max_errors_to_log:
                        print(
                            f'Warning: Invalid header in {path} '
                            f'line {line_num}: {line}',
                            flush=True
                        )
                        error_count += 1
                continue

            if cur_base is None:
                continue

            floats = float_re.findall(line)
            if len(floats) < 3:
                if error_count < max_errors_to_log:
                    print(
                        f'Warning: Not enough numeric fields in {path} '
                        f'line {line_num}: {line}',
                        flush=True
                    )
                    error_count += 1
                continue

            try:
                h_v = float(floats[0])
                d_v = float(floats[1])
                z_v = float(floats[2])
            except Exception as e:
                if error_count < max_errors_to_log:
                    print(
                        f'Warning: Error parsing numbers in {path} '
                        f'line {line_num}: {line} - {e}',
                        flush=True
                    )
                    error_count += 1
                continue

            if any(math.isinf(x) or math.isnan(x) for x in (h_v, d_v, z_v)):
                if error_count < max_errors_to_log:
                    print(
                        f'Warning: NaN/inf in {path} line {line_num}: {line}',
                        flush=True
                    )
                    error_count += 1
                continue

            if any(abs(x) > 1e4 for x in (h_v, d_v, z_v)):
                if error_count < max_errors_to_log:
                    print(
                        f'Warning: extreme values in {path} '
                        f'line {line_num}: {line}',
                        flush=True
                    )
                    error_count += 1
                continue

            HnT = h_v * scale_H
            DnT = d_v * scale_D
            ZnT = z_v * scale_Z

            ts = cur_base + timedelta(minutes=cur_min)
            records.append({
                'dateTime': ts,
                'MAGH': HnT,
                'MAGD': DnT,
                'MAGZ': ZnT
            })
            cur_min += 1

    if not records:
        raise RuntimeError(f'No valid FGM data in {path!r}')

    if error_count >= max_errors_to_log:
        print(f'Additional errors in {path} were not displayed', flush=True)

    return pd.DataFrame.from_records(records)

def debug_ovh_paths(station: str, date: datetime) -> None:
    """Debug OVH path construction by testing multiple versions.

    Args:
        station: Station code.
        date: Date to check.
    """
    ftp_handler = get_ftp_handler()

    print(f'DEBUG OVH for {station} on {date}:')

    main_path = ftp_handler.construct_remote_path(station, 'OVH', date)
    print(f'  Main path: {main_path} - Exists: {ftp_handler.file_exists(main_path)}')

    ds = date.strftime('%Y%m%d')
    for version in [1, 2, 3, 4]:
        alt_path = (
            f'OVH/{station.upper()}OVH{version}/raw/'
            f'{date.year}/{date.month:02d}/{station.upper()}OVH{version}-{ds}'
        )
        exists = ftp_handler.file_exists(alt_path)
        print(f'  OVH{version}: {alt_path} - Exists: {exists}')
        if exists:
            try:
                size = ftp_handler.ftp.size(alt_path)
                print(f'    Size: {size} bytes')
            except Exception:
                print('    Size: Unknown')

def load_ovh_for_station(station, start_dt, end_dt, base_path):
    """Load OVH data - FTP version"""
    rows = []
    ftp_handler = get_ftp_handler()
    
    for single_date in pd.date_range(start_dt, end_dt):
        # FIXED: Remove the station_version parameter
        remote_path = ftp_handler.construct_remote_path(station, 'OVH', single_date)

        debug_ovh_paths(station, single_date)

        if ftp_handler.file_exists(remote_path):
            path_found = remote_path
        else:
            # Try alternative OVH paths
            ds = single_date.strftime("%Y%m%d")
            alternative_paths = [
                f"OVH/{station.upper()}OVH1/raw/{single_date.year}/{single_date.month:02d}/{station.upper()}OVH1-{ds}",
                f"OVH/{station.upper()}OVH2/raw/{single_date.year}/{single_date.month:02d}/{station.upper()}OVH2-{ds}",
                f"OVH/{station.upper()}OVH3/raw/{single_date.year}/{single_date.month:02d}/{station.upper()}OVH3-{ds}",
                f"OVH/{station.upper()}OVH4/raw/{single_date.year}/{single_date.month:02d}/{station.upper()}OVH4-{ds}",
            ]
            
            path_found = None
            for alt_path in alternative_paths:
                if ftp_handler.file_exists(alt_path):
                    path_found = alt_path
                    break
        
        if path_found is None:
            # Create empty DataFrame for missing days
            start_day = pd.to_datetime(f"{single_date.strftime('%Y-%m-%d')} 00:00:00")
            times = pd.date_range(start=start_day, periods=86400, freq='s')
            Fvals = np.full(86400, np.nan)
            tmp = pd.DataFrame({'dateTime': pd.to_datetime(times), 'F_meas': Fvals})
            rows.append(tmp)
            continue
        
        try:
            # Download and process the file
            local_path = ftp_handler.download_file(path_found)
            if local_path:
                try:
                    df_ovh = pd.read_csv(local_path, sep=r'\s+', header=None, engine='python')
                    if df_ovh.shape[1] < 2:
                        continue
                    
                    start_day = pd.to_datetime(f"{single_date.strftime('%Y-%m-%d')} 00:00:00")
                    times = pd.date_range(start=start_day, periods=len(df_ovh), freq='s')
                    Fvals = pd.to_numeric(df_ovh.iloc[:, 1], errors='coerce').astype(float)
                    
                    # Basic masking (SAME)
                    Fvals[(Fvals.abs() >= 90000.0) | (Fvals.abs() > 1e6)] = np.nan
                    Fvals[Fvals < 25000] = np.nan

                    # Quality flag masking
                    if df_ovh.shape[1] > 2:
                        qcol = pd.to_numeric(df_ovh.iloc[:, 2], errors='coerce')
                        Fvals[qcol != 99] = np.nan

                    # Simple cleaning
                    Fvals_series = pd.Series(Fvals)
                    rolling_med = Fvals_series.rolling(21, center=True, min_periods=7).median()
                    deviation = (Fvals_series - rolling_med).abs()
                    
                    # Remove obvious spikes (> 300 nT from median)
                    spike_mask = deviation > 300
                    Fvals_clean = Fvals_series.copy()
                    Fvals_clean[spike_mask] = np.nan
                    
                    # NO INTERPOLATION - preserve gaps
                    if len(times) != len(Fvals_clean):
                        minlen = min(len(times), len(Fvals_clean))
                        times = pd.to_datetime(times[:minlen])
                        Fvals_clean = Fvals_clean[:minlen]
                    
                    tmp = pd.DataFrame({'dateTime': pd.to_datetime(times), 'F_meas': Fvals_clean})
                    rows.append(tmp)
                    
                except Exception as e:
                    print(f"Error processing OVH data for {single_date}: {e}")
                    # Create empty DataFrame for this day if reading fails
                    start_day = pd.to_datetime(f"{single_date.strftime('%Y-%m-%d')} 00:00:00")
                    times = pd.date_range(start=start_day, periods=86400, freq='s')
                    Fvals = np.full(86400, np.nan)
                    tmp = pd.DataFrame({'dateTime': pd.to_datetime(times), 'F_meas': Fvals})
                    rows.append(tmp)
                finally:
                    if os.path.exists(local_path):
                        os.remove(local_path)
        except Exception as e:
            print(f"Error downloading OVH data for {single_date}: {e}")
            # Create empty DataFrame for this day if downloading fails
            start_day = pd.to_datetime(f"{single_date.strftime('%Y-%m-%d')} 00:00:00")
            times = pd.date_range(start=start_day, periods=86400, freq='s')
            Fvals = np.full(86400, np.nan)
            tmp = pd.DataFrame({'dateTime': pd.to_datetime(times), 'F_meas': Fvals})
            rows.append(tmp)
            continue
    
    if not rows:
        full_times = pd.date_range(start=start_dt, end=end_dt, freq='s')
        return pd.DataFrame({'dateTime': full_times, 'F_meas': np.full(len(full_times), np.nan)})
    
    out = pd.concat(rows, ignore_index=True)
    out['dateTime'] = pd.to_datetime(out['dateTime'], errors='coerce')
    
    # Filter to the exact time range requested
    out = out[(out['dateTime'] >= start_dt) & (out['dateTime'] <= end_dt)]
    
    valid_f_points = out['F_meas'].notna().sum()
    print(f"OVH F_meas summary: {len(out)} records, {valid_f_points} valid F points", flush=True)
    
    return out

def calculate_Fc_corrected(H_series, Z_series):
    """Calculate Fc from variometer H and Z components - MORE ROBUST"""
    # Use the ORIGINAL series without additional cleaning
    # (they're already cleaned at original resolution)
    valid_mask = H_series.notna() & Z_series.notna()
    
    Fc = pd.Series(np.nan, index=H_series.index)
    if valid_mask.any():
        Fc[valid_mask] = np.sqrt(H_series[valid_mask]**2 + Z_series[valid_mask]**2)
    
    return Fc

# =============================================================================
# MAIN PLOTTING FUNCTIONS
# =============================================================================


def plotSingleStation(
    stationData: pd.DataFrame,
    stationLabel: str,
    startTime: str,
    endTime: str,
    outputFile: Optional[str],
    applyBias: bool,
    station: str,
    base_path: str
) -> plt.Figure:
    """Generate multi-panel plot for single station magnetic data.

    Creates a comprehensive plot showing H, D, Z, F, and dF components
    with automatic axis formatting based on time range.

    Args:
        stationData: DataFrame with magnetic field components and dateTime.
        stationLabel: Label for plot title (e.g., 'HERFGM1').
        startTime: Start time string in 'YYYY-MM-DD HH:MM:SS' format.
        endTime: End time string in 'YYYY-MM-DD HH:MM:SS' format.
        outputFile: Unused (kept for API compatibility).
        applyBias: Whether to apply SQD bias scaling factors.
        station: Station code for OVH data loading.
        base_path: Base path for data files.

    Returns:
        Matplotlib Figure object for GUI display.
    """
    # Parse start and end times
    start_dt = pd.to_datetime(startTime)
    end_dt = pd.to_datetime(endTime)
    
    print(f"Starting plotSingleStation for {stationLabel}", flush=True)
    print(f"Station: {station}, Variometer data provided", flush=True)
    
    # Handle timestamp detection
    if 'dateTime' not in stationData.columns:
        if 'DATE' in stationData.columns and 'TIME' in stationData.columns:
            stationData['dateTime'] = pd.to_datetime(stationData['DATE'] + ' ' + stationData['TIME'])
        elif 'dateTime_str' in stationData.columns:
            stationData['dateTime'] = pd.to_datetime(stationData['dateTime_str'], format="%Y\\%m\\%d %H:%M:%S")
        elif stationData.index.name == 'dateTime':
            stationData = stationData.reset_index()
        else:
            raise KeyError("Could not find any timestamp column")
    
    stationData['dateTime'] = pd.to_datetime(stationData['dateTime'], errors='coerce')
    
    # Filter to time range
    mask = (stationData['dateTime'] >= start_dt) & (stationData['dateTime'] <= end_dt)
    stationData = stationData.loc[mask].copy()
    
    print(f"Variometer data after filtering: {len(stationData)} records", flush=True)
    
    # CLEAN VARIOMETER DATA AT ORIGINAL RESOLUTION FIRST
    # ONLY clean H, D, Z - NOT F (F comes from OVH)
    print("Cleaning variometer components at original resolution...", flush=True)
    for comp in ('MAGH', 'MAGD', 'MAGZ'):  # REMOVED MAGF - it comes from OVH
        if comp in stationData.columns:
            original_count = stationData[comp].notna().sum()
            stationData[comp] = clean_component_at_original_resolution(stationData[comp])
            cleaned_count = stationData[comp].notna().sum()
            print(f"  {comp}: {original_count} -> {cleaned_count} valid points ({original_count - cleaned_count} removed)", flush=True)
    
    # Now aggregate variometer data to minutes if needed
    points_per_day = len(stationData) / ((end_dt - start_dt).days + 1)
    print(f"Variometer data density after cleaning: {points_per_day:.0f} points per day", flush=True)
    
    if points_per_day > 1440:  # More than 1 point per minute = high resolution
        print(f"Aggregating variometer data from {len(stationData)} to minute resolution...", flush=True)
        original_size = len(stationData)
        stationData = minute_aggregate_clean_data(stationData)
        print(f"Aggregated variometer data from {original_size} to {len(stationData)} records", flush=True)
    
    # Get baseline values for variometer
    station_inst = stationLabel.replace(station, "")
    try:
        H0, D0, Z0 = loadBaselineValues(station, station_inst)
        print(f"Applied baseline to variometer: H0={H0}, D0={D0}, Z0={Z0}", flush=True)
    except Exception as e:
        print(f"Warning: Could not load baseline values for variometer: {e}", flush=True)
        H0, D0, Z0 = 0.0, 0.0, 0.0
    
    # Apply baseline corrections to VARIOMETER data
    print("Applying baseline corrections to variometer data...", flush=True)
    if 'MAGH' in stationData.columns:
        stationData['MAGH'] = np.where(stationData['MAGH'].notna(), stationData['MAGH'] + H0, np.nan)
    if 'MAGD' in stationData.columns:
        stationData['MAGD'] = np.where(stationData['MAGD'].notna(), stationData['MAGD'] + D0, np.nan)
    if 'MAGZ' in stationData.columns:
        stationData['MAGZ'] = np.where(stationData['MAGZ'].notna(), stationData['MAGZ'] + Z0, np.nan)

    # Compute MAGD in arc-min (after baseline)
    if 'MAGH' in stationData.columns and 'MAGD' in stationData.columns:
        # Use absolute values for the conversion like plot4.py
        Habs = stationData['MAGH']          # already contains H0
        Dabs = stationData['MAGD']          # already contains D0
        
        # Safe denominator: avoid divide-by-zero; use absolute H magnitude
        denom = Habs.abs().replace({0.0: np.nan})
    
    # Convert declination (nT) to arc-minutes using baseline-corrected values
    
    # --- Safe MAGD calculation: ensure denom and Dabs always exist ---
    # denom: prefer absolute MAGH; fallback to other H-like columns or a NaN series
    if 'MAGH' in stationData.columns:
        denom = stationData['MAGH'].abs()
    elif 'H' in stationData.columns:
        denom = stationData['H'].abs()
    else:
        denom = pd.Series(np.nan, index=stationData.index)

    # Dabs: prefer a local variable if present, else check DataFrame columns for D-like values
    try:
        _Dabs = Dabs  # use previously computed Dabs if it exists in scope
    except NameError:
        # Look for likely D columns in stationData; prefer MAGD, then D, then any col with 'D' (but not DATE/TIME)
        if 'MAGD' in stationData.columns:
            _Dabs = stationData['MAGD']
        elif 'D' in stationData.columns:
            _Dabs = stationData['D']
        else:
            # last resort: NaN series
            _Dabs = pd.Series(np.nan, index=stationData.index)

    # Ensure _Dabs is numeric (coerce if not)
    try:
        _Dabs = pd.to_numeric(_Dabs, errors='coerce')
    except Exception:
        _Dabs = pd.Series(np.nan, index=stationData.index)

    # Now compute MAGD safely (arc-min = D * 3438 / |H|)
    stationData['MAGD'] = np.where(denom.notna() & (~denom.isna()) & (denom != 0),
                                (_Dabs * 3438.0) / denom,
                                np.nan)
    # --- end safe MAGD calculation ---

    print("Converted D component to arc-minutes", flush=True)
    
    # Debug: check D component values
    if 'MAGD' in stationData.columns and stationData['MAGD'].notna().any():
        d_stats = stationData['MAGD'].describe()
        print(f"D component stats after conversion: min={d_stats['min']:.2f}, max={d_stats['max']:.2f}, mean={d_stats['mean']:.2f}", flush=True)

    # CALCULATE Fc FROM VARIOMETER H AND Z COMPONENTS
    print("Calculating Fc from variometer H and Z components...", flush=True)
    if 'MAGH' in stationData.columns and 'MAGZ' in stationData.columns:
        stationData['Fc'] = calculate_Fc_corrected(stationData['MAGH'], stationData['MAGZ'])
        fc_valid = stationData['Fc'].notna().sum()
        print(f"Fc calculated from variometer: {fc_valid} valid points", flush=True)
        if fc_valid > 0:
            print(f"Fc range: {stationData['Fc'].min():.1f} to {stationData['Fc'].max():.1f} nT", flush=True)
    else:
        stationData['Fc'] = np.nan
        print("Cannot calculate Fc: missing MAGH or MAGZ in variometer data", flush=True)

    # LOAD F_meas FROM OVH - THIS IS THE ONLY SOURCE FOR MEASURED F
    print("Loading F_meas from OVH instrument...", flush=True)
    ovh_data = load_ovh_for_station(station.upper(), start_dt, end_dt, base_path)
    
    if not ovh_data.empty and 'F_meas' in ovh_data.columns and ovh_data['F_meas'].notna().any():
        print(f"OVH F_meas data: {ovh_data['F_meas'].notna().sum()} valid points", flush=True)
        
        # Merge OVH F_meas data with variometer data
        stationData = stationData.merge(ovh_data[['dateTime', 'F_meas']], 
                                      on='dateTime', how='left')
        
        f_meas_valid = stationData['F_meas'].notna().sum()
        print(f"Final F_meas from OVH: {f_meas_valid} valid points", flush=True)
        if f_meas_valid > 0:
            f_meas_times = stationData[stationData['F_meas'].notna()]['dateTime']
            print(f"F_meas time range: {f_meas_times.min()} to {f_meas_times.max()}", flush=True)
            print(f"F_meas range: {stationData['F_meas'].min():.1f} to {stationData['F_meas'].max():.1f} nT", flush=True)
    else:
        stationData['F_meas'] = np.nan
        print("WARNING: No F_meas data available from OVH", flush=True)

    # Calculate dF = F_meas (from OVH) - Fc (from variometer)
    print("Calculating dF = F_meas (OVH) - Fc (variometer)...", flush=True)
    stationData['dF'] = np.nan
    valid_mask = stationData['F_meas'].notna() & stationData['Fc'].notna()
    stationData.loc[valid_mask, 'dF'] = stationData.loc[valid_mask, 'F_meas'] - stationData.loc[valid_mask, 'Fc']
    
    dF_valid = stationData['dF'].notna().sum()
    print(f"dF calculated: {dF_valid} valid points", flush=True)
    if dF_valid > 0:
        print(f"dF range: {stationData['dF'].min():.1f} to {stationData['dF'].max():.1f} nT", flush=True)

    # DEBUG: Compare F_meas (OVH) and Fc (variometer)
    if stationData['F_meas'].notna().any() and stationData['Fc'].notna().any():
        f_meas_mean = stationData['F_meas'].mean()
        fc_mean = stationData['Fc'].mean()
        print(f"F_meas (OVH) mean: {f_meas_mean:.1f}, Fc (variometer) mean: {fc_mean:.1f}, mean difference: {f_meas_mean - fc_mean:.1f} nT", flush=True)

    # Compute normalized components for plotting
    # Compute normalized components for plotting - FIXED VERSION
    print("Computing normalized components for plotting...", flush=True)
    if 'MAGH' in stationData.columns and stationData['MAGH'].notna().any():
        station_H_mean = stationData['MAGH'].mean()
        # REMOVE division by std - just subtract mean to center the data
        stationData['avgH'] = stationData['MAGH'] - station_H_mean
        print(f"H component: mean={station_H_mean:.1f}, {stationData['avgH'].notna().sum()} valid points", flush=True)
    
    if 'MAGZ' in stationData.columns and stationData['MAGZ'].notna().any():
        station_Z_mean = stationData['MAGZ'].mean()
        # REMOVE division by std - just subtract mean to center the data
        stationData['avgZ'] = stationData['MAGZ'] - station_Z_mean
        print(f"Z component: mean={station_Z_mean:.1f}, {stationData['avgZ'].notna().sum()} valid points", flush=True)
    
    if 'MAGD' in stationData.columns and stationData['MAGD'].notna().any():
        D_mean = stationData['MAGD'].mean()
        # REMOVE division by std - just subtract mean to center the data
        stationData['avgD'] = stationData['MAGD'] - D_mean
        print(f"D component: mean={D_mean:.1f}, {stationData['avgD'].notna().sum()} valid points", flush=True)
    else:
        stationData['avgD'] = np.nan

    # Apply bias if needed
    if applyBias:
        if 'avgH' in stationData.columns:
            stationData['avgH'] *= 0.7
        if 'avgZ' in stationData.columns:
            stationData['avgZ'] *= 0.8
        print("Applied bias scaling", flush=True)

    # Determine number of subplots
    has_D = 'avgD' in stationData.columns and stationData['avgD'].notna().any()
    NoPlts = 5 if has_D else 4

    print(f"Creating plot with {NoPlts} subplots...", flush=True)
    print(f"Final data counts - H: {stationData['avgH'].notna().sum()}, Z: {stationData['avgZ'].notna().sum()}, F_meas (OVH): {stationData['F_meas'].notna().sum()}, Fc (variometer): {stationData['Fc'].notna().sum()}, dF: {stationData['dF'].notna().sum()}", flush=True)

    # Create plot
    fig, axs = plt.subplots(
        NoPlts, 1, sharex=True,
        figsize=(10, 12),
        gridspec_kw={'hspace': 0.08},
        dpi=100
    )

    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=2, label=stationLabel),
        plt.Line2D([0], [0], color='green', linewidth=2, label='F (OVH measured)'),
        plt.Line2D([0], [0], color='orange', linewidth=2, label='Fc (variometer calculated)', linestyle='--')
    ]
    
    # Plot variometer components (H, D, Z)
    if 'avgH' in stationData.columns and stationData['avgH'].notna().any():
        axs[0].plot(stationData['dateTime'], stationData['avgH'], color='blue')
        # Create a single legend at the top of the first subplot
        axs[0].legend(handles=legend_elements, loc='upper center', 
                      bbox_to_anchor=(0.5, 1.3), ncol=5, frameon=False,
                      columnspacing=1.0, handlelength=2.5, handletextpad=0.5)
        axs[0].grid(True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        axs[0].grid(True)
        axs[0].minorticks_on()
        axs[0].set_ylabel('H (nT)')
        # Add padding to y-axis to prevent cut-off appearance
        y_data = stationData['avgH'].dropna()
        if len(y_data) > 0:
            y_range = y_data.max() - y_data.min()
            axs[0].set_ylim(y_data.min() - 0.1*y_range, y_data.max() + 0.1*y_range)
    
    if has_D and 'avgD' in stationData.columns and stationData['avgD'].notna().any():
        axs[1].plot(stationData['dateTime'], stationData['avgD'], color='blue')
        axs[1].set_ylabel('D (min)')
        # Add padding to y-axis to prevent cut-off appearance
        y_data = stationData['avgD'].dropna()
        if len(y_data) > 0:
            y_range = y_data.max() - y_data.min()
            axs[1].set_ylim(y_data.min() - 0.1*y_range, y_data.max() + 0.1*y_range)
        axs[1].minorticks_on()
        axs[1].grid(True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        axs[1].grid(True)
        z_index = 2
    else:
        z_index = 1
    
    if 'avgZ' in stationData.columns and stationData['avgZ'].notna().any():
        axs[z_index].plot(stationData['dateTime'], stationData['avgZ'], color='blue')
        axs[z_index].grid(True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        axs[z_index].minorticks_on()
        axs[z_index].grid(True)
        axs[z_index].set_ylabel('Z (nT)')
        # Add padding to y-axis to prevent cut-off appearance
        y_data = stationData['avgZ'].dropna()
        if len(y_data) > 0:
            y_range = y_data.max() - y_data.min()
            axs[z_index].set_ylim(y_data.min() - 0.1*y_range, y_data.max() + 0.1*y_range)
    
    f_index = z_index + 1
    
    # Plot F components - F_meas from OVH vs Fc from variometer
    has_f_meas = 'F_meas' in stationData.columns and stationData['F_meas'].notna().any()
    has_fc = 'Fc' in stationData.columns and stationData['Fc'].notna().any()
    
    if has_f_meas or has_fc:
        if has_f_meas:
            F_meas_normalized = stationData['F_meas'] - stationData['F_meas'].mean()
            axs[f_index].plot(stationData['dateTime'], F_meas_normalized, color='green', linewidth=1)
            print(f"Plotted F_meas (OVH) with {stationData['F_meas'].notna().sum()} points", flush=True)
        
        if has_fc:
            Fc_normalized = stationData['Fc'] - stationData['Fc'].mean()
            axs[f_index].plot(stationData['dateTime'], Fc_normalized, color='orange', linestyle='--', linewidth=1)
            print(f"Plotted Fc (variometer) with {stationData['Fc'].notna().sum()} points", flush=True)
        
        axs[f_index].set_ylabel('F (nT)')
        axs[f_index].minorticks_on()
        axs[f_index].grid(True)
        axs[f_index].grid(True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    else:
        print("WARNING: No F data to plot", flush=True)
    
    df_index = f_index + 1
    if 'dF' in stationData.columns and stationData['dF'].notna().any():
        dF_normalized = stationData['dF'] - stationData['dF'].mean()
        axs[df_index].plot(stationData['dateTime'], dF_normalized, color='blue', linewidth=1)
        axs[df_index].set_ylabel('dF (nT)')
        axs[df_index].minorticks_on()
        axs[df_index].grid(True)
        axs[df_index].grid(True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        print(f"Plotted dF with {stationData['dF'].notna().sum()} points", flush=True)
    else:
        print("WARNING: No dF data to plot", flush=True)
    
    # Set exact time range for x-axis
    for ax in axs:
        ax.set_xlim(start_dt, end_dt)
    
    # Format x-axis based on time range
    span_days = (end_dt - start_dt).total_seconds() / 86400.0
    ax = axs[-1]
    
    if span_days <= 1:
        # Format for single day
        span_secs = (end_dt - start_dt).total_seconds()
        max_ticks = 12
        raw_interval = span_secs / max_ticks
        
        nice_steps = [60, 300, 600, 900, 1200, 1800, 3600, 7200, 10800, 14400]
        interval = next(step for step in nice_steps if step >= raw_interval)
        
        if interval % 3600 == 0:
            hours = interval // 3600
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=hours))
        else:
            minutes = interval // 60
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=minutes))
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0)
            lbl.set_ha('center')
            lbl.set_va('center')

        ax.tick_params(axis='x', which='major', pad=12)

        start_dt_rounded = start_dt.round(f'{interval}s')
        end_dt_rounded = end_dt.round(f'{interval}s')
        ax.set_xlim(start_dt_rounded, end_dt_rounded)

        # Calculate hours correctly
        total_hours = (end_dt_rounded - start_dt_rounded).total_seconds() / 3600
        
        # Remove .0 if it's a whole number
        if total_hours.is_integer():
            hours_display = int(total_hours)
        else:
            hours_display = total_hours
        
        caption = f"{hours_display} hours (UTC) for {start_dt.strftime('%Y/%m/%d')}"
        
    elif span_days <= 4:
        # Format for multi-day (up to 5 days) - FIXED VERSION
        major_times = []
        current_date = start_dt.normalize()
        end_date_normalized = end_dt.normalize()
        
        while current_date <= end_date_normalized:
            # Add midnight
            if current_date >= start_dt and current_date <= end_dt:
                major_times.append(current_date)
            # Add 6am, noon, 6pm
            for hour in [6, 12, 18]:
                candidate = current_date + pd.Timedelta(hours=hour)
                if candidate >= start_dt and candidate <= end_dt:
                    major_times.append(candidate)
            current_date += pd.Timedelta(days=1)
        
        ax.set_xticks(major_times)
        
        def fmt(x, pos):
            dt = mdates.num2date(x)
            if dt.hour == 0 and dt.minute == 0:
                return dt.strftime('%H:%M\n%d')
            else:
                return dt.strftime('%H:%M')
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))
        
        n_days = int(math.ceil(span_days))
        first_date = start_dt.strftime('%Y/%m/%d')
        last_date = end_dt.strftime('%Y/%m/%d')
        if first_date == last_date:
            caption = f"{n_days} Day on {first_date}"
        else:
            caption = f"{n_days} Days from {first_date} to {last_date}"
    else:
        # Format for longer ranges
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        
        n_days = int(math.ceil(span_days))
        first_date = start_dt.strftime('%Y/%m/%d')
        last_date = end_dt.strftime('%Y/%m/%d')
        if first_date == last_date:
            caption = f"{n_days} Day on {first_date}"
        else:
            caption = f"{n_days} Days from {first_date} to {last_date}"

    # Add caption
    ax.annotate(
        caption,
        xy=(0.5, -0.5),
        xycoords='axes fraction',
        ha='center',
        va='top',
        fontsize=12
    )
    
    # Set title and adjust layout
    plt.suptitle(f"{stationLabel}", fontsize=12, y=0.95)
    plt.subplots_adjust(hspace=0.3, bottom=0.15)
    
    # ALWAYS return figure for GUI display - never auto-save
    print("Plot created successfully for GUI display", flush=True)
    return fig

def save_plot(
    fig: plt.Figure,
    station: str,
    instrument: str,
    start_date: str,
    end_date: str
) -> None:
    """Placeholder for plot saving - GUI handles file dialogs.

    This function is not used in GUI mode as the application provides
    its own save functionality through the interface.

    Args:
        fig: Matplotlib Figure to save.
        station: Station code.
        instrument: Instrument code.
        start_date: Start date string.
        end_date: End date string.
    """
    print('Warning: save_plot called but plots are displayed in GUI', flush=True)
    return None


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================


def process_single_station(
    base_path: Optional[str] = None,
    station: Optional[str] = None,
    instrument: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    out_base: Optional[str] = None,
    mode: str = 'plot',
    instruments: Optional[List[str]] = None,
    gui: bool = False,
    output_format: Optional[str] = None,
    compare_dtu: bool = True
):
    """Main entry point for single-station data processing and plotting.

    Supports multiple instruments (FGM, DTU, L251, SQD, CTU) with options
    for data conversion, plotting, or both. Handles FTP data retrieval
    and format conversion to IAGA-2002.

    Args:
        base_path: Base directory for data files (or output directory).
        station: Station code (e.g., 'HER', 'HBK').
        instrument: Instrument code (e.g., 'FGM1', 'DTU1', 'SQD').
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        start_time: Start time in 'HH:MM:SS' format (default '00:00:00').
        end_time: End time in 'HH:MM:SS' format (default '23:59:59').
        out_base: Alternative output directory path.
        mode: Operation mode - 'plot', 'convert', or 'both'.
        instruments: List of instruments (alternative to single instrument).
        gui: Whether running in GUI mode (affects return type).
        output_format: Output format for SQD/CTU ('HDZF', 'DHZF', 'XYZF').
        compare_dtu: Whether to compare SQD data with DTU1 reference.

    Returns:
        In GUI mode: Matplotlib Figure object.
        In CLI mode: List of (instrument, date, (data, figure)) tuples.
        For convert-only mode: None on success.

    Raises:
        ValueError: If required parameters are missing.
        RuntimeError: If no data found for specified parameters.
    """
    try:
        # --- Map alternate keyword names into the vars used throughout this function ---
        # If caller passed out_base (convertData style) map it to base_path
        if out_base is not None and base_path is None:
            base_path = out_base
            
        if base_path is None and out_base is None:
            # Use the default output directory
            base_path = os.path.expanduser(r"~/Documents/SANSA_Processed_Data")
            print(f"[DEBUG] Using default output directory: {base_path}")

        # If caller passed instruments (list) and instrument not explicitly given, pick the first instrument
        if instruments is not None and instrument is None:
            if isinstance(instruments, (list, tuple)) and len(instruments) > 0:
                instrument = instruments[0]
            else:
                instrument = instruments

        # Provide safe defaults if missing (tries not to break callers that omitted args)
        if station is None:
            station = 'HER'   # sensible default for your app; change if needed
        if start_date is None or end_date is None:
            raise ValueError("process_single_station requires start_date and end_date")
        if start_time is None:
            start_time = '00:00:00'
        if end_time is None:
            end_time = '23:59:59'

        # Determine if this is a SQD/CTU instrument (needs conversion) or other instrument (plot only)
        is_sqd = instrument is not None and instrument.upper().startswith("SQD")
        is_ctu = instrument is not None and instrument.upper().startswith("CTU")
        is_conversion_instrument = is_sqd or is_ctu
        
        # For non-SQD/CTU instruments, force mode to 'plot' since they don't need conversion
        if not is_conversion_instrument and mode in ['convert', 'both']:
            print(f"[INFO] {instrument} doesn't need conversion, switching to 'plot' mode")
            mode = 'plot'

        # Save/convert behavior: decide whether to write converted files to disk
        save_files = False
        plot_files = False
        if mode is not None:
            mode_lower = str(mode).lower()
            save_files = mode_lower in ('convert', 'both')
            plot_files = mode_lower in ('plot', 'both')
            print(f"DEBUG: mode='{mode}', mode_lower='{mode_lower}', save_files={save_files}, plot_files={plot_files}", flush=True)


        # Convert input dates to datetime objects
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        station_list = []
        last_sqd_fig = None            # store SQD plot
        sqd_processed = False          # flag indicating an SQD pipeline ran
        ftp_handler = get_ftp_handler()
        
        sqd_data_list = []
        herdtu1_data_list = []
        ctu_data_list = []
        
        # Use FTP file discovery logic
        for single_date in pd.date_range(start_dt, end_dt):
            ds = single_date.strftime("%Y%m%d")
            df = None  # Initialize df to None for each date
            
            if instrument.upper().startswith("FGM"):
                remote_path = ftp_handler.construct_remote_path(station, instrument, single_date)
                
                if ftp_handler.file_exists(remote_path):
                    local_path = ftp_handler.download_file(remote_path)
                    if local_path:
                        try:
                            df = load_fgm(local_path, station_name=station, instrument_name=instrument)
                            if df is not None and not df.empty:
                                required_cols = ['dateTime', 'MAGH', 'MAGD', 'MAGZ']
                                if not all(col in df.columns for col in required_cols):
                                    print(f"Warning: FGM file {remote_path} missing required columns", flush=True)
                                    df = None
                        except Exception as e:
                            print(f"Error loading FGM file {remote_path}: {e}", flush=True)
                            df = None
                        finally:
                            if os.path.exists(local_path):
                                os.remove(local_path)
                else:
                    df = None
                        
            elif instrument.upper().startswith("DTU"):
                remote_path = ftp_handler.construct_remote_path(station, instrument, single_date)
                
                if ftp_handler.file_exists(remote_path):
                    local_path = ftp_handler.download_file(remote_path)
                    if local_path:
                        try:
                            df = pd.read_csv(local_path, sep=r"\s+", skiprows=15, engine="python")
                            df["dateTime"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
                        except Exception as e:
                            print(f"Error reading DTU file {remote_path}: {e}", flush=True)
                            df = None
                        finally:
                            if os.path.exists(local_path):
                                os.remove(local_path)
                else:
                    df = None
            
            elif instrument.upper().startswith("L25"):
                remote_path = ftp_handler.construct_remote_path(station, instrument, single_date)

                if ftp_handler.file_exists(remote_path):
                    local_path = ftp_handler.download_file(remote_path)
                    if local_path:
                        try:
                            df_tmp = pd.read_csv(local_path, sep=r",\s*", header=None, engine="python")
                            
                            if df_tmp.shape[1] >= 4:
                                df_tmp.columns = ["dateTime_str", "MAGH", "MAGD", "MAGZ"]
                                df_tmp["dateTime"] = pd.to_datetime(
                                    df_tmp["dateTime_str"],
                                    format="%Y\\%m\\%d %H:%M:%S"
                                )
                                df = df_tmp
                            else:
                                df = None
                        except Exception as e:
                            print(f"Error reading L251 file {remote_path}: {e}", flush=True)
                            df = None
                        finally:
                            if os.path.exists(local_path):
                                os.remove(local_path)
                else:
                    df = None
                    
            #
            # --- START: SQD / CTU handling---
            #
            elif instrument.upper().startswith("SQD"):
                # Use convertData's download helper so the downloaded filename is YYYY-MM-DD.squid
                date_str = single_date.strftime('%Y-%m-%d')
                local_path = None
                try:
                    local_path = download_from_ftp(instrument, date_str)
                    if local_path and os.path.exists(local_path):
                        try:
                            # DEBUG: Check what output directory we're using
                            output_dir = base_path or out_base or os.getcwd()
                            print(f"[DEBUG] SQD output directory: {output_dir}")
                            print(f"[DEBUG] SQD save_file flag: {save_files}")
                            
                            # Delegate conversion + plotting to convertData's SQD pipeline.
                            outdf, _ = process_sqd_with_formats(
                                local_path,
                                output_dir,
                                output_format=(output_format if output_format else 'HDZF'),
                                compare_dtu=compare_dtu,
                                save_file=save_files,
                                plot=False # Don't plot individual days
                            )

                            if outdf is not None:
                                # Normalise timestamp
                                if 'DATE' in outdf.columns and 'TIME' in outdf.columns:
                                    outdf['dateTime'] = pd.to_datetime(outdf['DATE'].astype(str) + ' ' + outdf['TIME'].astype(str), errors='coerce')
                                elif 'dateTime' in outdf.columns:
                                    outdf['dateTime'] = pd.to_datetime(outdf['dateTime'], errors='coerce')

                                # Robustly map possible SQD column names to MAGH/MAGD/MAGZ
                                h_col = next((c for c in ('H', 'H_raw', 'X')), None)
                                d_col = next((c for c in ('D', 'D_raw', 'Y')), None)
                                z_col = next((c for c in ('Z', 'Z_raw')), None)

                                tmp = pd.DataFrame({'dateTime': outdf['dateTime']})
                                tmp['MAGH'] = outdf[h_col] if (h_col and h_col in outdf.columns) else np.nan
                                tmp['MAGD'] = outdf[d_col] if (d_col and d_col in outdf.columns) else np.nan
                                tmp['MAGZ'] = outdf[z_col] if (z_col and z_col in outdf.columns) else np.nan

                                df = tmp.copy()
                                sqd_data_list.append(outdf)
                                if compare_dtu:
                                    herdtu1_df = load_herdtu1_data(date_str)
                                    if herdtu1_df is not None:
                                        herdtu1_data_list.append(herdtu1_df)
                                
                                sqd_processed = True
                        except Exception as e:
                            print(f"Error processing SQD file {date_str}: {e}", flush=True)
                            df = None
                    else:
                        df = None
                finally:
                    # cleanup the downloaded file created by download_from_ftp
                    try:
                        if local_path and os.path.exists(local_path):
                            os.remove(local_path)
                    except Exception:
                        pass

            elif instrument.upper().startswith("CTU"):
                # Use the same download helper as SQD for consistency
                date_str = single_date.strftime('%Y-%m-%d')
                local_path = None
                try:
                    local_path = download_from_ftp(instrument, date_str)
                    if local_path and os.path.exists(local_path):
                        try:
                            # process_ctu_file returns (outdf, fig) where outdf has DATE, TIME, DOY, D (arcmin), H, Z, F
                            outdf, fig = process_ctu_file(
                                local_path, 
                                base_path, 
                                output_format=output_format if output_format else 'HDZF', 
                                save_file=save_files, 
                                plot=plot_files, 
                                date_str=single_date.strftime('%Y-%m-%d')
                            )

                            if outdf is not None:
                                outdf['dateTime'] = pd.to_datetime(outdf['DATE'] + ' ' + outdf['TIME'])
                                # Map CTU columns to expected MAG* names
                                tmp = outdf[['dateTime', 'H', 'D', 'Z']].rename(columns={'H': 'MAGH', 'D': 'MAGD', 'Z': 'MAGZ'})
                                # Ensure presence of columns
                                for c in ('MAGH', 'MAGD', 'MAGZ'):
                                    if c not in tmp.columns:
                                        tmp[c] = np.nan
                                df = tmp.copy()
                                
                                # ADD: Collect data for combined plotting
                                station_list.append(df)
                                
                        except Exception as e:
                            print(f"Error processing CTU file {remote_path}: {e}", flush=True)
                            df = None
                    else:
                        df = None
                finally:
                    # cleanup the downloaded file created by download_from_ftp
                    try:
                        if local_path and os.path.exists(local_path):
                            os.remove(local_path)
                    except Exception:
                        pass

            #
            # --- END: SQD / CTU handling ---
            #

            else:
                # Handle unknown instrument types
                df = None
                print(f"Warning: Unknown instrument type {instrument}", flush=True)
            
            # If there is no data for that day, skip
            if df is None:
                continue
            
            # For non-SQD instruments, add to station_list
            if not instrument.upper().startswith("SQD"):
                station_list.append(df)
            
        # --- SQD-specific plot creation ---
        if is_sqd and plot_files and sqd_data_list:
            date_range_str = f"{start_date}_to_{end_date}"
            
            if compare_dtu and herdtu1_data_list and len(herdtu1_data_list) > 0:
                if output_format == 'XYZF':
                    last_sqd_fig = plot_sqd_range(sqd_data_list, herdtu1_data_list, date_range_str, output_format='XYZF')
                else:
                    last_sqd_fig = plot_sqd_range(sqd_data_list, herdtu1_data_list, date_range_str, output_format='HDZF')
            else:
                last_sqd_fig = plot_sqd_only(sqd_data_list, date_range_str)
            
            # Return the SQD figure directly - don't go through the normal plotting flow
            if gui:
                return last_sqd_fig
            else:
                date_str = start_date if isinstance(start_date, str) else pd.to_datetime(start_date).strftime('%Y-%m-%d')
                # For SQD, we don't have stationData in the same way
                return [(instrument, date_str, (None, last_sqd_fig))]

        # For conversion-only mode, return success
        if mode == 'convert' and save_files and is_conversion_instrument:
            print(f"[SUCCESS] Conversion completed for {instrument} from {start_date} to {end_date}")
            return None
        
        # For other instruments or when no SQD data, continue with normal processing
        if len(station_list) == 0 and not (is_sqd and sqd_data_list):
            raise RuntimeError(f"No {instrument} data found for {station} between {start_date} and {end_date}.")
        
        # For non-SQD instruments, create the normal plot
        if not is_sqd and len(station_list) > 0:
            stationData = pd.concat(station_list, ignore_index=True)
            
            # Rename columns to standard names
            rename_map = {}
            for col in stationData.columns:
                if col.upper().endswith('H') and not col.startswith('MAG'):
                    rename_map[col] = 'MAGH'
                elif col.upper().endswith('D') and not col.startswith('MAG'):
                    rename_map[col] = 'MAGD'
                elif col.upper().endswith('Z') and not col.startswith('MAG'):
                    rename_map[col] = 'MAGZ'
            
            if rename_map:
                stationData = stationData.rename(columns=rename_map)
                print(f"Renamed columns: {rename_map}", flush=True)
            
            # Build the legend-label
            stationLabel = f"{station.upper()}{instrument}"
            
            # Compute "applyBias" - only for SQD
            applyBias = is_sqd

            # Only reindex FGM data (minute resolution)
            if instrument.upper().startswith("FGM"):
                start_ts = pd.to_datetime(f"{start_date} {start_time}")
                end_ts = pd.to_datetime(f"{end_date} {end_time}")
                target_idx = pd.date_range(start=start_ts, end=end_ts, freq='1min')
                
                if 'dateTime' in stationData.columns:
                    stationData = stationData.set_index('dateTime')
                    stationData = stationData.reindex(target_idx)
                    stationData = stationData.reset_index().rename(columns={'index': 'dateTime'})
                    print(f"Reindexed FGM data to {len(stationData)} minute records", flush=True)

            # Debug info
            print(f"\n=== DEBUG before plotSingleStation ===", flush=True)
            print(f"StationData type: {type(stationData)}", flush=True)
            print(f"StationData shape: {stationData.shape}", flush=True)
            print(f"StationData columns: {stationData.columns.tolist()}", flush=True)
            if not stationData.empty:
                print(f"First few rows:", flush=True)
                print(stationData.head(3), flush=True)
            
            # Ensure we have the required timestamp column
            if 'dateTime' not in stationData.columns:
                timestamp_cols = [col for col in stationData.columns if 'time' in col.lower() or 'date' in col.lower()]
                print(f"Potential timestamp columns: {timestamp_cols}", flush=True)
                
                if 'DATE' in stationData.columns and 'TIME' in stationData.columns:
                    print("Creating dateTime from DATE and TIME columns", flush=True)
                    stationData['dateTime'] = pd.to_datetime(stationData['DATE'] + ' ' + stationData['TIME'])
                elif 'dateTime_str' in stationData.columns:
                    print("Creating dateTime from dateTime_str column", flush=True)
                    stationData['dateTime'] = pd.to_datetime(
                        stationData['dateTime_str'],
                        format="%Y\\%m\\%d %H:%M:%S"
                    )

            # Create plot for non-SQD instruments
            fig = plotSingleStation(
                stationData,
                stationLabel,
                f"{start_date} {start_time}",
                f"{end_date} {end_time}",
                None,  # outputFile is always None now
                applyBias,
                station.upper(),
                base_path
            )
            
            # Return value compatibility
            if gui:
                return fig
            else:
                date_str = start_date if isinstance(start_date, str) else pd.to_datetime(start_date).strftime('%Y-%m-%d')
                return [(instrument, date_str, (stationData, fig))]
        
    except Exception:
        traceback.print_exc()
        raise
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and plot single station time series data.')
    parser.add_argument('base_path', help='Base directory containing data files')
    parser.add_argument('station1', help='First station code (e.g., HER)')
    parser.add_argument('instrument1', help='First station instrument folder (e.g., OVH or L251)')
    parser.add_argument('station2', help='Second station code (ignored for single station mode)')
    parser.add_argument('instrument2', help='Second station instrument folder (ignored for single station mode)')
    parser.add_argument('start_date', help='Start date in YYYY-MM-DD format')
    parser.add_argument('end_date', help='End date in YYYY-MM-DD format')
    parser.add_argument('start_time', help='Start time (HH:MM or HH:MM:SS format)')
    parser.add_argument('end_time', help='End time (HH:MM or HH:MM:SS format)')
    # ADDED: Optional argument for GUI mode
    parser.add_argument('--gui', action='store_true', help='Run in GUI mode (returns figure instead of saving)')
    
    args = parser.parse_args()
    
    # Add seconds if not present
    if len(args.start_time.split(':')) == 2:
        args.start_time = args.start_time + ':00'
    if len(args.end_time.split(':')) == 2:
        args.end_time = args.end_time + ':00'
    
    # MODIFIED: Pass gui_mode parameter
    process_single_station(
        args.base_path,
        args.station1,
        args.instrument1,
        args.start_date,
        args.end_date,
        args.start_time,
        args.end_time
    )
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and convert SQD/CTU data')
    parser.add_argument('start_date', help='Start date in YYYY-MM-DD format')
    parser.add_argument('end_date', help='End date in YYYY-MM-DD format')
    parser.add_argument('--mode', type=str, default='both', choices=['convert', 'plot', 'both'], 
                       help='Operation mode: convert, plot, or both')
    parser.add_argument('--outbase', type=str, 
                       default=os.path.expanduser(r"~/Documents/SANSA_Processed_Data"),
                       help='Output base directory')
    parser.add_argument('--instruments', nargs='+', required=True,
                       help='Instruments to process (SQD, CTU, or both)')
    
    args = parser.parse_args()
    
    # Call the processing function
    try:
        results = process_single_station(
            start_date=args.start_date,
            end_date=args.end_date,
            out_base=args.outbase,
            mode=args.mode,
            instruments=args.instruments
        )
        print("Conversion completed successfully")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        traceback.print_exc()
        sys.exit(1)