"""Multi-station geomagnetic data plotting and conversion module.

This module provides functionality for plotting and converting geomagnetic field data
from multiple stations and instruments, specifically designed for SANSA (South African
National Space Agency) operations. It handles various data formats including SQD, CTU,
FGM1/FGM2, DTU, OVH (Overhauser), and L251.

Key components:
    - H (horizontal), D (declination), Z (vertical), F (total field)
    - Sentinel values: 88888, 99999, 99999.99
    - Data conversion between formats (SQD/CTU)
    - Multi-station comparative plotting
    - FTP data retrieval integration

Version: 2025/09/10
"""

import argparse
import glob
import json
import math
import os
import pathlib
import re
import sys
import textwrap
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from connectFTP import get_ftp_handler

mpl.rcParams['font.size'] = 10
mpl.rcParams['lines.linewidth'] = 0.6
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))


# ============================================================================
# CONSTANTS
# ============================================================================

# Station information constants
STATION_NAME = 'Hermanus Magnetic Observatory'
IAGA_CODE = 'HER'
CTU_SUBDIR = 'CTU'

# Sentinel values used to mark missing or invalid data
SEC_SENTINEL = 99999.00
F_SENTINEL = 88888.00


# ============================================================================
# TIME CONVERSION FUNCTIONS
# ============================================================================

def mapTimeValues(time: int) -> int:
    """Map encoded time values to total seconds.

    Converts compact time representations to standard seconds format.
    Handles formats: HHMMSS (>=10000), MMSS (>=100), or raw seconds (<100).

    Args:
        time: Encoded time value (e.g., 10530 for 1h 5m 30s, 130 for 1m 30s).

    Returns:
        Total seconds as integer.

    Examples:
        >>> mapTimeValues(10530)  # 1h 5m 30s
        3930
        >>> mapTimeValues(130)    # 1m 30s
        90
        >>> mapTimeValues(45)     # 45s
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
    """Apply time value mapping to dataframe in-place.

    Args:
        data: DataFrame containing a 'Time' column to be corrected.
    """
    data['Time'] = data['Time'].apply(mapTimeValues)


# ============================================================================
# DATA CLEANING AND SENTINEL MASKING FUNCTIONS
# ============================================================================

def _mask_sentinels_and_extremes(
    s: pd.Series,
    sentinels: tuple = (88888, 99999, 99999.99),
    extreme_threshold: float = 1e6,
    sentinel_floor: float = 90000.0
) -> pd.Series:
    """Replace sentinel codes and extreme values with NaN.

    Vectorized masking of sentinel values and absurdly large measurements
    commonly found in geomagnetic data files.

    Args:
        s: Pandas series containing numeric data to be cleaned.
        sentinels: Tuple of sentinel codes to mask (default: 88888, 99999, 99999.99).
        extreme_threshold: Absolute value threshold for extreme values (default: 1e6).
        sentinel_floor: Minimum absolute value to treat as sentinel (default: 90000.0).

    Returns:
        Cleaned series with sentinels and extremes replaced by NaN.
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
    threshold_factor: int = 5
) -> pd.Series:
    """Remove clusters of outliers using rolling median deviation.

    Identifies and removes 'fat' outliers (groups of deviant points) using
    a rolling median and MAD (Median Absolute Deviation) approach. Short
    groups (<=20 samples) of extreme deviations are set to NaN.

    Args:
        series: Pandas series containing data to despike.
        window_size: Size of rolling window for median calculation (default: 51).
        threshold_factor: Multiplier for MAD threshold (default: 5).

    Returns:
        Series with fat outliers replaced by NaN.
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
        print(f"  Removed {removed} 'fat' outlier points", flush=True)
    return s

def d_to_arcmin(D_nT: np.ndarray, H_nT: np.ndarray) -> np.ndarray:
    """Convert declination from nanoTesla to arc minutes.

    Vectorized conversion using the standard formula: D(arcmin) = D(nT) * 3438 / H(nT).
    Invalid values (NaN or H near zero) result in NaN output.

    Args:
        D_nT: Declination values in nanoTesla.
        H_nT: Horizontal component values in nanoTesla.

    Returns:
        Declination values in arc minutes.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mask = (~np.isnan(D_nT)) & (~np.isnan(H_nT)) & (np.abs(H_nT) > 0.1)
        # REMOVE np.abs() - use H_nT directly
        res = np.where(valid_mask, (D_nT * 3438.0) / H_nT, np.nan)
    return res

def clean_ctu_data_after_conversion(ctu_df: pd.DataFrame) -> pd.DataFrame:
    """Clean CTU data after format conversion.

    Applies sentinel masking and extreme value filtering to H, D, and Z components.
    Reports cleaning statistics for each component.

    Args:
        ctu_df: DataFrame containing CTU data with H, D, Z columns.

    Returns:
        Cleaned DataFrame with sentinels replaced by NaN.
    """
    print("Cleaning CTU data.", flush=True)
    for comp in ['H', 'D', 'Z']:
        if comp in ctu_df.columns:
            original_count = ctu_df[comp].notna().sum()
            print(f"Before cleaning {comp}: min={ctu_df[comp].min():.2f}, max={ctu_df[comp].max():.2f}", flush=True)
            ctu_df[comp] = _mask_sentinels_and_extremes(ctu_df[comp])
            cleaned_count = ctu_df[comp].notna().sum()
            print(f"  {comp}: {original_count} -> {cleaned_count} valid points", flush=True)
            if cleaned_count < original_count:
                print(f"  REMOVED {original_count - cleaned_count} points from {comp}!", flush=True)
    return ctu_df



# ============================================================================
# CTU FILE PROCESSING AND CONVERSION
# ============================================================================

def process_ctu_file(
    filepath: str,
    out_dir: str,
    save_file: bool = True,
    plot: bool = False,
    date_str: str = None
) -> tuple:
    """Process and convert CTU format geomagnetic data files.

    Reads raw CTU data, aggregates to seconds resolution, converts D from nT to
    arc minutes, and fills missing seconds with sentinel values. Based on
    dataprocessv2.py processing pipeline.

    Args:
        filepath: Path to input CTU file.
        out_dir: Output directory for converted files.
        save_file: Whether to save converted output (default: True).
        plot: Whether to generate diagnostic plots (default: False).
        date_str: Date string override in 'YYYY-MM-DD' format (default: None).

    Returns:
        Tuple of (DataFrame with converted data, None for plot placeholder).
        Returns (None, None) on processing errors.
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
    
    # --- ADD CTU SCALING ---
    D_SCALE_CTU = 2.96  # Adjust as needed
    print(f"[CTU] Applying D scaling: {D_SCALE_CTU}", flush=True)
    D_arcmin = D_arcmin * D_SCALE_CTU
    # --- END CTU SCALING ---

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

    outdf = pd.DataFrame({
        'DATE': final_merged['DATE'],
        'TIME': final_merged['TIME'], 
        'DOY': final_merged['DOY'],
        'H': final_merged['H_mean'],
        'D': final_merged['D_arcmin'],
        'Z': final_merged['Z_mean'],
        'F': F_SENTINEL
    })

    return outdf, None


# ============================================================================
# DESPIKING AND OUTLIER REMOVAL FUNCTIONS
# ============================================================================

def _mask_sentinels_and_extremes(
    s: pd.Series,
    sentinels: tuple = (88888, 99999, 99999.99),
    extreme_threshold: float = 1e6,
    sentinel_floor: float = 90000.0
) -> pd.Series:
    """Replace sentinel codes and absurdly large values with NaN.

    Alternative implementation with explicit replacement logic. Handles both
    integer and float sentinel variants.

    Args:
        s: Pandas series containing numeric data to be cleaned.
        sentinels: Tuple of sentinel codes to mask.
        extreme_threshold: Absolute value threshold for extreme values.
        sentinel_floor: Minimum absolute value to treat as sentinel.

    Returns:
        Cleaned series with dtype float64.
    """
    if s is None:
        return s
    out = s.copy().astype(float)
    # replace explicit sentinel codes (int or float)
    for code in sentinels:
        try:
            out.replace(code, np.nan, inplace=True)
        except Exception:
            pass
    # also treat any value >= sentinel_floor as sentinel/NaN (covers 99999.99, 99999.0, etc.)
    out[(out.abs() >= float(sentinel_floor))] = np.nan
    # treat anything absurdly large as NaN as well
    out[(out.abs() > extreme_threshold)] = np.nan
    return out

def despike_series(
    x: pd.Series,
    window: int = 11,
    n_mad: float = 3.0,
    interpolate: bool = True,
    interpolate_limit: int = 3,
    min_abs_threshold: float = 0.1
) -> pd.Series:
    """Remove spikes using rolling median and MAD (Median Absolute Deviation).

    Implements a robust despiking algorithm based on rolling median and MAD.
    Points exceeding n_mad * MAD from the median are marked as spikes.

    Args:
        x: Pandas series containing data to despike.
        window: Rolling window size for median calculation (default: 11).
        n_mad: Number of MAD units for spike threshold (default: 3.0).
        interpolate: Whether to fill small gaps after despiking (default: True).
        interpolate_limit: Max consecutive points to interpolate (default: 3).
        min_abs_threshold: Minimum absolute threshold to prevent tiny thresholds (default: 0.1).

    Returns:
        Despiked series with spikes replaced by NaN or interpolated values.
    """
    if x is None or len(x) == 0:
        return x

    s = _mask_sentinels_and_extremes(x)
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()

    # threshold: either n_mad * mad or a small absolute floor to avoid tiny thresholds
    threshold = (n_mad * mad).clip(lower=min_abs_threshold)

    spike_mask = (s - med).abs() > threshold
    # do not mark values that are already NaN
    spike_mask &= s.notna()

    y = s.copy()
    if spike_mask.any():
        print(f"MAD-despike filter: clamped {int(spike_mask.sum())} points -> NaN", flush=True)
        y[spike_mask] = np.nan

    if interpolate:
        # only interpolate short gaps (protect long sentinel sequences)
        return y.interpolate(limit=interpolate_limit, limit_direction='both')
    else:
        return y


# ============================================================================
# TIME AGGREGATION FUNCTIONS
# ============================================================================

def minute_aggregate_exclude_nans(
    df: pd.DataFrame,
    ts_col: str = 'dateTime'
) -> pd.DataFrame:
    """Aggregate seconds-resolution data to minute-resolution.

    Resamples data to 60-second intervals using mean aggregation, excluding NaN
    values. Minutes with all NaN values remain NaN.

    Args:
        df: DataFrame containing timestamp column and numeric MAG* columns.
        ts_col: Name of the datetime column (default: 'dateTime').

    Returns:
        DataFrame with one row per minute (left-closed intervals), columns
        aggregated by mean excluding NaNs.

    Raises:
        KeyError: If ts_col is not found in dataframe columns or index.
    """
    if df is None:
        return pd.DataFrame(columns=[ts_col])

    d = df.copy()

    # ensure datetime col (if index contains dateTime, promote it)
    if ts_col not in d.columns and d.index.name == ts_col:
        d = d.reset_index()

    if ts_col not in d.columns:
        raise KeyError(f"minute_aggregate_exclude_nans: no '{ts_col}' column or index in dataframe")

    # coerce to datetime
    d[ts_col] = pd.to_datetime(d[ts_col], errors='coerce')

    # set index and choose candidate columns to aggregate
    d = d.set_index(ts_col)
    cols = [c for c in ('MAGH', 'MAGD', 'MAGZ', 'MAGF', 'F_meas') if c in d.columns]

    if not cols:
        # nothing to aggregate, return index-only frame for compatibility
        res_idx = d.resample('60s', label='left', closed='left').apply(lambda s: np.nan)
        res_idx = res_idx.reset_index()[[ts_col]]
        return res_idx

    def mean_dropna(series):
        non_na = series.dropna()
        return np.nan if non_na.empty else non_na.mean()

    # resample at 60s and apply mean_dropna on chosen columns
    res = d[cols].resample('60s', label='left', closed='left').apply(mean_dropna)

    # reset index to get dateTime as column again
    res = res.reset_index()
    res[ts_col] = pd.to_datetime(res[ts_col], errors='coerce')
    return res

def window_majority_despike(
    s: pd.Series,
    window: int = 11,
    majority: int = 8,
    threshold: float = 20.0,
    interpolate_limit: int = 2
) -> pd.Series:
    """Remove spikes using window majority consensus approach.

    Implements a sliding window algorithm where a point is marked as a spike if
    fewer than 'majority' points in its window are within 'threshold' of the
    window median.

    Args:
        s: Pandas series containing data to despike.
        window: Size of sliding window (default: 11).
        majority: Minimum number of inlier points required (default: 8).
        threshold: Maximum deviation from median for inliers in nT (default: 20.0).
        interpolate_limit: Max consecutive points to interpolate (default: 2).

    Returns:
        Despiked series with identified spikes replaced by interpolated values.
    """
    if s is None or len(s) == 0:
        return s

    y = s.copy().astype(float)
    y = _mask_sentinels_and_extremes(y)  # guard sentinels

    n = len(y)
    half = window // 2
    mask = np.zeros(n, dtype=bool)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + window)
        window_vals = y.iloc[lo:hi]
        # ignore NaNs in window when computing median
        med = np.nanmedian(window_vals) if window_vals.notna().any() else np.nan
        if np.isnan(med):
            continue
        inlier = (window_vals - med).abs() <= threshold
        # if too few inliers AND current point deviates from med beyond threshold => mark
        if inlier.sum() < majority and abs(y.iloc[i] - med) > threshold:
            mask[i] = True

    if mask.any():
        print(f"Window-majority despike: clamped {mask.sum()} points -> NaN", flush=True)
        y.iloc[mask] = np.nan

    # limit interpolation across short gaps only
    return y.interpolate(limit=interpolate_limit, limit_direction='both')

def aggregate_seconds_to_minute(
    df: pd.DataFrame,
    ts_col: str = 'dateTime',
    despike_cols: tuple = ('MAGH', 'MAGD', 'MAGZ', 'MAGF'),
    despike_window: int = 11,
    despike_n_mad: float = 3.0
) -> pd.DataFrame:
    """Aggregate seconds data to minutes with conservative despiking.

    Multi-stage processing pipeline:
        1. Mask sentinel/extreme values
        2. Remove derivative spikes (no interpolation)
        3. Apply MAD despiking (no interpolation to avoid inventing values)
        4. Apply window-majority guard despiking
        5. Resample to minutes using mean (excluding NaNs)

    Args:
        df: DataFrame containing seconds-resolution data.
        ts_col: Name of datetime column (default: 'dateTime').
        despike_cols: Columns to despike (default: MAGH, MAGD, MAGZ, MAGF).
        despike_window: Window size for MAD despiking (default: 11).
        despike_n_mad: MAD threshold multiplier (default: 3.0).

    Returns:
        DataFrame resampled to minute resolution with despiked data.

    Raises:
        KeyError: If ts_col not found in dataframe.
    """
    if ts_col not in df.columns:
        raise KeyError(f"aggregate_seconds_to_minute: no {ts_col} in dataframe")

    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col])

    # early masking of sentinel codes on all relevant columns
    for c in despike_cols:
        if c in d.columns:
            d[c] = _mask_sentinels_and_extremes(d[c])

    # apply non-interpolating despike + derivative clamp to mark bad seconds NaN
    for c in despike_cols:
        if c in d.columns:
            # 1) derivative clamp (do not interpolate here)
            vv = d[c].copy()
            vv = remove_derivative_spikes(vv, spike_diff=50.0, interpolate_limit=0)
            # 2) MAD clamp without interpolation
            vv = despike_series(vv, window=despike_window, n_mad=despike_n_mad, interpolate=False)
            # 3) final short-window majority guard
            vv = window_majority_despike(vv, window=9, majority=max(6, (9*2)//3), threshold=20.0, interpolate_limit=0)
            d[c] = vv

    # resample to minutes, excluding NaNs (mean of available seconds)
    d = d.set_index(ts_col)
    cols = [c for c in despike_cols if c in d.columns]
    def mean_dropna(s):
        non_na = s.dropna()
        return np.nan if non_na.empty else non_na.mean()
    res = d[cols].resample('60s', label='left', closed='left').apply(mean_dropna)
    res = res.reset_index()
    res['dateTime'] = pd.to_datetime(res['dateTime'], errors='coerce')
    return res

def remove_derivative_spikes(
    series: pd.Series,
    spike_diff: float = 50.0,
    interpolate_limit: int = 2
) -> pd.Series:
    """Remove abrupt spikes using first-order derivative analysis.

    Detects spikes by examining forward and backward first differences. Points
    with large jumps in both directions (or extremely large in one direction)
    are marked as spikes.

    Args:
        series: Pandas series containing data to despike.
        spike_diff: Threshold for spike detection in nT (default: 50.0).
        interpolate_limit: Max consecutive points to fill after despiking (default: 2).

    Returns:
        Series with derivative spikes removed and gaps filled with rolling mean.
    """
    if series is None or len(series) == 0:
        return series
        
    y = series.copy().astype(float)
    dy = y.diff().abs()
    fwd = dy
    bwd = dy.shift(-1)
    
    # detect strong or asymmetric spikes
    spike_mask = (fwd > spike_diff) & (bwd > spike_diff)
    spike_mask |= (fwd > (spike_diff * 1.5)) | (bwd > (spike_diff * 1.5))
    
    y[spike_mask] = np.nan
    
    # second-pass catch for any surviving isolated large derivative
    dy2 = y.diff().abs()
    y[dy2 > (spike_diff * 1.2)] = np.nan
    
    # safer fill using rolling mean rather than direct interpolate
    y = y.fillna(y.rolling(3, min_periods=1, center=True).mean())
    return y


# ============================================================================
# OVH (OVERHAUSER MAGNETOMETER) DATA LOADING
# ============================================================================

def debug_ovh_paths(station: str, date: datetime) -> None:
    """Debug OVH file path construction and availability.

    Tests main and alternative OVH paths for a given station and date,
    reporting file existence and sizes.

    Args:
        station: Station code (e.g., 'HER', 'HBK').
        date: Date to check for OVH data.
    """
    ftp_handler = get_ftp_handler()
    
    print(f"DEBUG OVH for {station} on {date}:")
    
    # Try the main constructed path
    main_path = ftp_handler.construct_remote_path(station, 'OVH', date)
    print(f"  Main path: {main_path} - Exists: {ftp_handler.file_exists(main_path)}")
    
    # Try alternative paths
    ds = date.strftime("%Y%m%d")
    for version in [1, 2, 3, 4]:
        alt_path = f"OVH/{station.upper()}OVH{version}/raw/{date.year}/{date.month:02d}/{station.upper()}OVH{version}-{ds}"
        exists = ftp_handler.file_exists(alt_path)
        print(f"  OVH{version}: {alt_path} - Exists: {exists}")
        if exists:
            try:
                size = ftp_handler.ftp.size(alt_path)
                print(f"    Size: {size} bytes")
            except:
                print(f"    Size: Unknown")


def load_ovh_for_range(
    station: str,
    start_dt: datetime,
    end_dt: datetime,
    base_path: str
) -> pd.DataFrame:
    """Load OVH (Overhauser magnetometer) data for a date range via FTP.

    Downloads OVH data files for each day in the range, applies quality filtering,
    and performs basic spike removal. Uses per-day caching to avoid redundant
    downloads. Empty DataFrames with NaN values are created for missing days.

    Args:
        station: Station code (e.g., 'HER', 'HBK').
        start_dt: Start datetime for data range.
        end_dt: End datetime for data range.
        base_path: Base directory path (used for local caching).

    Returns:
        DataFrame with 'dateTime' and 'MAGF' columns at seconds resolution.
        Missing or invalid data points are set to NaN.
    """
    rows = []
    ftp_handler = get_ftp_handler()
    
    # Cache for already loaded days
    loaded_days = {}
    
    for single_date in pd.date_range(start_dt, end_dt):
        date_key = single_date.strftime('%Y%m%d')
        cache_key = f"{station}_{date_key}"
        
        # Check if we already have this day loaded
        if cache_key in loaded_days:
            rows.append(loaded_days[cache_key])
            continue
            
        # FIXED: Remove the station_version parameter
        remote_path = ftp_handler.construct_remote_path(station, 'OVH', single_date)

        # COMMENT OUT or reduce debug prints
        # debug_ovh_paths(station, single_date)

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
            tmp = pd.DataFrame({'dateTime': pd.to_datetime(times), 'MAGF': Fvals})
            loaded_days[cache_key] = tmp
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
                    
                    tmp = pd.DataFrame({'dateTime': pd.to_datetime(times), 'MAGF': Fvals_clean})
                    loaded_days[cache_key] = tmp
                    rows.append(tmp)
                    
                except Exception as e:
                    print(f"Error processing OVH data for {single_date}: {e}")
                    # Create empty DataFrame for this day if reading fails
                    start_day = pd.to_datetime(f"{single_date.strftime('%Y-%m-%d')} 00:00:00")
                    times = pd.date_range(start=start_day, periods=86400, freq='s')
                    Fvals = np.full(86400, np.nan)
                    tmp = pd.DataFrame({'dateTime': pd.to_datetime(times), 'MAGF': Fvals})
                    loaded_days[cache_key] = tmp
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
            tmp = pd.DataFrame({'dateTime': pd.to_datetime(times), 'MAGF': Fvals})
            loaded_days[cache_key] = tmp
            rows.append(tmp)
            continue
    
    if not rows:
        full_times = pd.date_range(start=start_dt, end=end_dt, freq='s')
        return pd.DataFrame({'dateTime': full_times, 'MAGF': np.full(len(full_times), np.nan)})
    
    out = pd.concat(rows, ignore_index=True)
    out['dateTime'] = pd.to_datetime(out['dateTime'], errors='coerce')
    return out
def clean_component(series: pd.Series, aggressive: bool = False) -> pd.Series:
    """High-level component cleaning pipeline for plotting.

    Applies comprehensive cleaning using multiple despiking methods:
        1. Sentinel masking
        2. Derivative spike removal
        3. MAD-based despiking
        4. Window-majority guard despiking

    Args:
        series: Pandas series containing magnetic field component data.
        aggressive: Whether to use more aggressive interpolation limits (default: False).

    Returns:
        Cleaned series with outliers removed and short gaps optionally interpolated.
    """
    if series is None or len(series) == 0:
        return series

    s = _mask_sentinels_and_extremes(series)

    # Enhanced derivative removal
    s = remove_derivative_spikes(s, spike_diff=80.0, 
                                interpolate_limit=(4 if aggressive else 2))

    # MAD despike with adjusted parameters
    s = despike_series(s, window=13, n_mad=3.5,
                       interpolate=True, interpolate_limit=(4 if aggressive else 2))

    # Final guard with tighter thresholds
    s = window_majority_despike(s, window=9, majority=6, threshold=15.0,
                                interpolate_limit=(2 if aggressive else 1))
    return s


# ============================================================================
# BASELINE AND FIELD CALCULATION FUNCTIONS
# ============================================================================

_INSTR_ALIAS = {'L251': 'L025'}


def loadBaselineValues(
    station: str,
    instrument: str,
    file_path: str = 'baselineValues.json'
) -> tuple:
    """Load baseline correction values from JSON configuration file.

    Reads station and instrument-specific baseline values (H0, D0, Z0) from
    a JSON file. These values are used to correct raw measurements to absolute
    field values.

    Args:
        station: Station code (e.g., 'HER', 'HBK').
        instrument: Instrument name (e.g., 'FGM1', 'DTU', 'L251').
        file_path: Path to baseline JSON file (default: 'baselineValues.json').

    Returns:
        Tuple of (H0, D0, Z0) baseline values as floats.

    Raises:
        RuntimeError: If baseline file cannot be opened.
        KeyError: If station/instrument combination not found.
        ValueError: If baseline entry is malformed.
    """
    inst_key = instrument.upper()
    inst_key = _INSTR_ALIAS.get(inst_key, inst_key)
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Cannot open baseline file {file_path}: {e}")
    
    try:
        entry = data[station.upper()][inst_key]
        H0 = float(entry['H0'])
        D0 = float(entry['D0'])
        Z0 = float(entry['Z0'])
    except KeyError:
        raise KeyError(f"Baseline for station={station!r}, instr={instrument!r} not found in {file_path}")
    except Exception as e:
        raise ValueError(f"Malformed baseline entry for {station}/{inst_key}: {e}")
    
    print(f"Applying baseline -> {station}/{inst_key}: H0={H0}, D0={D0}, Z0={Z0}", flush=True)
    return H0, D0, Z0

def debug_ftp_paths(station: str, instrument: str, date: datetime) -> tuple:
    """Debug FTP path construction and file availability.

    Tests constructed FTP paths and attempts to list parent directory contents
    if file is not found.

    Args:
        station: Station code (e.g., 'HER', 'HBK').
        instrument: Instrument name (e.g., 'FGM1', 'DTU').
        date: Date to check for data files.

    Returns:
        Tuple of (remote_path, file_exists_boolean).
    """
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

def apply_baseline_and_compute_F(
    df: pd.DataFrame,
    station: str,
    instrument: str
) -> pd.DataFrame:
    """Apply baseline corrections and compute total field values.

    Applies station/instrument-specific baseline corrections to H, D, Z components,
    converts D to arc minutes, and calculates F (total field) and dF (difference
    between measured and calculated F).

    Args:
        df: DataFrame containing raw magnetic field data.
        station: Station code for baseline lookup.
        instrument: Instrument name for baseline lookup.

    Returns:
        DataFrame with baseline-corrected components and computed F values.
    """
    if df is None or df.empty:
        return df
    
    for c in ('MAGH', 'MAGD', 'MAGZ', 'MAGF', 'F_meas'):
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = _mask_sentinels_and_extremes(df[c])
    
    try:
        H0, D0, Z0 = loadBaselineValues(station, instrument)
    except Exception:
        H0, D0, Z0 = 0.0, 0.0, 0.0
    
    df['MAGH'] = np.where(df['MAGH'].notna(), df['MAGH'] + H0, np.nan)
    df['MAGD'] = np.where(df['MAGD'].notna(), df['MAGD'] + D0, np.nan)
    df['MAGZ'] = np.where(df['MAGZ'].notna(), df['MAGZ'] + Z0, np.nan)

    denom = df['MAGH'].abs().replace({0.0: np.nan})
    df['MAGD'] = np.where(denom.notna(),
                          df['MAGD'] * 3438.0 / denom,
                          np.nan)
    df['F_calc'] = np.sqrt(df['MAGH'].pow(2) + df['MAGZ'].pow(2))
    df['Fc']     = df['F_calc']

    df['MAGF']   = pd.to_numeric(df['MAGF'], errors='coerce')
    df['F_meas'] = df['F_meas'].combine_first(df['MAGF'])
    df['dF']     = np.where(df['F_meas'].notna() & df['Fc'].notna(),
                             df['F_meas'] - df['Fc'],
                             np.nan)
    return df



# ============================================================================
# FGM (FLUXGATE MAGNETOMETER) DATA LOADING
# ============================================================================

def load_fgm(
    path: str,
    station_name: str = None,
    instrument_name: str = None
) -> pd.DataFrame:
    """Load FGM1/FGM2 fluxgate magnetometer data files.

    Robust parser that handles various FGM file formats including concatenated
    values and extra trailing numbers. Extracts the first three floats from each
    line as H, D, Z components and applies station/instrument-specific scaling.

    FGM file format:
        - Header lines: YYYY DOY HHMM (timestamp)
        - Data lines: H D Z [optional extra fields]

    Special handling:
        - SANAE (SNA) FGM1: Uses (320, -320, -320) nT/V scaling
        - FGM2: Default 40 nT/V scaling
        - FGM1: Default 150 nT/V scaling

    Args:
        path: Path to FGM data file.
        station_name: Station code override (default: extracted from filename/path).
        instrument_name: Instrument name override (default: extracted from filename/path).

    Returns:
        DataFrame with 'dateTime', 'MAGH', 'MAGD', 'MAGZ' columns at minute resolution.

    Raises:
        RuntimeError: If no valid FGM data found in file.
    """
    # Extract base names from path for fallback
    filename = os.path.basename(path).upper()
    inst = (instrument_name or
            ("FGM1" if "FGM1" in filename else
             "FGM2" if "FGM2" in filename else
             os.path.basename(os.path.dirname(path)).upper()))
    station = (station_name or
               ("SNA" if "SNA" in filename else
                os.path.basename(os.path.dirname(os.path.dirname(path))).upper()))

    print(f"[DEBUG] FGM loader: station={station}, instrument={inst}")

    # Default scaling
    scale = 40.0 if inst == 'FGM2' else 150.0

    # SANAE (SNA) FGM1 special conversion factors (Volts → nT)
    if station == "SNA" and inst == "FGM1":
        scale_H, scale_D, scale_Z = 320.0, -320.0, -320.0
        print("[DEBUG] Applied SANAE FGM1 scaling (320, -320, -320 nT/V)")
    else:
        # Preserve existing logic for all others
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

            # header lines e.g. "2025 204 0000"
            hdr = line.split()
            if len(hdr) == 3 and hdr[0].isdigit():
                try:
                    year, doy, mcount = map(int, hdr)
                    cur_base = datetime(year, 1, 1) + timedelta(days=doy - 1)
                    cur_min = int(mcount)
                except Exception:
                    if error_count < max_errors_to_log:
                        print(f"Warning: Invalid header in {path} line {line_num}: {line}", flush=True)
                        error_count += 1
                continue

            # skip until we have a base timestamp
            if cur_base is None:
                continue

            # find floats anywhere in the line
            floats = float_re.findall(line)
            if len(floats) < 3:
                if error_count < max_errors_to_log:
                    print(f"Warning: Not enough numeric fields in {path} line {line_num}: {line}", flush=True)
                    error_count += 1
                continue

            try:
                # take first three numeric tokens as H, D, Z
                h_v = float(floats[0])
                d_v = float(floats[1])
                z_v = float(floats[2])
            except Exception as e:
                if error_count < max_errors_to_log:
                    print(f"Warning: Error parsing numbers in {path} line {line_num}: {line} - {e}", flush=True)
                    error_count += 1
                continue

            # sanity checks
            if any(math.isinf(x) or math.isnan(x) for x in (h_v, d_v, z_v)):
                if error_count < max_errors_to_log:
                    print(f"Warning: NaN/inf in {path} line {line_num}: {line}", flush=True)
                    error_count += 1
                continue
            if any(abs(x) > 1e4 for x in (h_v, d_v, z_v)):  # extremely big -> probably parse error
                if error_count < max_errors_to_log:
                    print(f"Warning: extreme values in {path} line {line_num}: {line}", flush=True)
                    error_count += 1
                continue

            # Apply station/instrument-specific scaling
            HnT = h_v * scale_H
            DnT = d_v * scale_D
            ZnT = z_v * scale_Z

            ts = cur_base + timedelta(minutes=cur_min)
            records.append({'dateTime': ts, 'MAGH': HnT, 'MAGD': DnT, 'MAGZ': ZnT})
            cur_min += 1

    if not records:
        raise RuntimeError(f"No valid FGM data in {path!r}")

    if error_count >= max_errors_to_log:
        print(f"Additional errors in {path} were not displayed", flush=True)

    return pd.DataFrame.from_records(records)



# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def plotData(
    firstData: pd.DataFrame,
    firstStation: str,
    secondData: pd.DataFrame,
    secondStation: str,
    firstLabel: str,
    secondLabel: str,
    startTime: str,
    endTime: str,
    outputFile: str,
    applyBias: bool,
    first_Fm: np.ndarray,
    second_Fm: np.ndarray = None
) -> plt.Figure:
    """Generate multi-panel comparison plot for two stations/instruments.

    Creates a comprehensive comparison plot with up to 6 panels:
        1. H component (horizontal)
        2. D component (declination) - if available
        3. Z component (vertical)
        4. F component (total field) - from OVH data
        5. dF (difference between measured and calculated F)
        6. Component differences (H, D, Z)

    Data processing includes:
        - Time range filtering and reindexing
        - Cleaning and despiking
        - Baseline correction
        - OVH F-field integration with per-station caching
        - Gap identification and visualization

    Args:
        firstData: DataFrame with first station data.
        firstStation: First station code.
        secondData: DataFrame with second station data.
        secondStation: Second station code.
        firstLabel: Label for first dataset (for legend).
        secondLabel: Label for second dataset (for legend).
        startTime: Start time string (format: 'YYYY-MM-DD HH:MM').
        endTime: End time string (format: 'YYYY-MM-DD HH:MM').
        outputFile: Output file path (unused in GUI mode).
        applyBias: Whether to apply SQD-specific bias corrections.
        first_Fm: OVH F measurements for first station.
        second_Fm: OVH F measurements for second station (default: None).

    Returns:
        Matplotlib figure object for GUI display.

    Raises:
        KeyError: If required data columns are missing.
        ValueError: If no data available in specified time range.
    """
    
    # Ensure datetime columns exist
    for df in [firstData, secondData]:
        if 'dateTime' not in df.columns:
            if {'DATE', 'TIME'}.issubset(df.columns):
                df['dateTime'] = pd.to_datetime(df['DATE'].str.strip() + ' ' + df['TIME'].str.strip())
            else:
                raise KeyError("Data must contain 'dateTime' or both 'DATE' and 'TIME' columns")
    
    # Parse time range
    start_dt = pd.to_datetime(startTime)
    end_dt = pd.to_datetime(endTime)
    
    # Filter data to time range
    firstData = firstData[(firstData['dateTime'] >= start_dt) & (firstData['dateTime'] <= end_dt)].copy()
    secondData = secondData[(secondData['dateTime'] >= start_dt) & (secondData['dateTime'] <= end_dt)].copy()
    
    if firstData.empty or secondData.empty:
        raise ValueError("No data available in the specified time range")
    
    # Determine data cadence
    raw_diffs = firstData['dateTime'].diff().dt.total_seconds().dropna()
    cadence_secs = raw_diffs.min() if len(raw_diffs) else 1
    cadence = pd.Timedelta(seconds=cadence_secs)
    
    if cadence_secs != cadence_secs or cadence <= pd.Timedelta(0):
        cadence = pd.Timedelta(seconds=1)
    
    # Clean component data
    for df in [firstData, secondData]:
        if cadence_secs <= 1:
            for comp in ('MAGH', 'MAGZ', 'MAGD', 'MAGF', 'F_meas'):
                if comp in df:
                    # mask large sentinels first (defensive)
                    df[comp] = _mask_sentinels_and_extremes(df[comp])
                    # run conservative cleaning (not aggressive here)
                    df[comp] = clean_component(df[comp])
                        
    # Ensure numeric columns
    for df in [firstData, secondData]:
        for c in ('MAGH', 'MAGZ', 'MAGD', 'MAGF', 'F_meas', 'F_calc'):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Sync F_meas and MAGF
        if 'MAGF' in df.columns and 'F_meas' in df.columns:
            if df['MAGF'].notna().any():
                df['F_meas'] = df['MAGF'].copy()
            elif df['F_meas'].notna().any():
                df['MAGF'] = df['F_meas'].copy()
        
        # Calculate F_calc only if both H and Z are available
        if 'MAGH' in df.columns and 'MAGZ' in df.columns:
            Hn = pd.to_numeric(df['MAGH'], errors='coerce')
            Zn = pd.to_numeric(df['MAGZ'], errors='coerce')
            # Only calculate where both H and Z are available
            df['F_calc'] = np.where(Hn.notna() & Zn.notna(), np.sqrt(Hn**2 + Zn**2), np.nan)
            df['Fc'] = df['F_calc']
        else:
            # If H or Z is missing, set F_calc to NaN
            df['F_calc'] = np.nan
            df['Fc'] = np.nan
        
        # Calculate dF ONLY when both F_meas and F_calc have valid data
        meas = pd.to_numeric(df.get('F_meas', pd.Series([np.nan] * len(df))), errors='coerce')
        calc = pd.to_numeric(df.get('F_calc', pd.Series([np.nan] * len(df))), errors='coerce')
        
        # Set dF to NaN everywhere by default
        df['dF'] = np.nan
        
        # Only calculate dF where BOTH F_meas and F_calc have valid values
        valid_mask = meas.notna() & calc.notna()
        df.loc[valid_mask, 'dF'] = meas[valid_mask] - calc[valid_mask]
    
    # Create full timeline
    full_idx = pd.date_range(start=start_dt, end=end_dt, freq=cadence)
    n = len(full_idx)
    
    # Identify data gaps
    diffs = pd.Series(full_idx).diff().dt.total_seconds()
    nominal_interval = diffs[diffs > 0].median()
    gap_threshold = nominal_interval * 2
    gaps = diffs[diffs > gap_threshold]
    gap_intervals = [(full_idx[i-1], full_idx[i]) for i in gaps.index]
    
    # Reindex data to full timeline
    firstData = firstData.loc[~firstData['dateTime'].duplicated()].copy()
    secondData = secondData.loc[~secondData['dateTime'].duplicated()].copy()
    
    firstData = firstData.set_index('dateTime').reindex(full_idx).rename_axis('dateTime').reset_index()
    secondData = secondData.set_index('dateTime').reindex(full_idx).rename_axis('dateTime').reset_index()
    
    # ----------------------------
    # OVH: build per-station minute-series (cache per station, reuse for multiple instruments)
    # - Prefer the lowest-numbered OVH version per-date (connectFTP.get_ovh_version already implements that)
    # - Do the expensive load_ovh_for_range() work once per station
    # - Never np.interp/stretch short arrays; pad with NaNs for missing tail
    # ----------------------------
    def _build_ovh_for_station(station_name):
        """Return a numpy array length n (minute samples aligned to full_idx) for the station,
        or an array of NaNs if nothing found."""
        station_key = station_name.upper()
        try:
            ovh_df = load_ovh_for_range(station_key, start_dt, end_dt, SCRIPT_DIR)
            if not ovh_df.empty and 'MAGF' in ovh_df.columns:
                ovh_min = minute_aggregate_exclude_nans(ovh_df[['dateTime', 'MAGF']], ts_col='dateTime')
                ovh_min['MAGF'] = pd.to_numeric(ovh_min['MAGF'], errors='coerce')
                ovh_min = ovh_min.set_index('dateTime').reindex(full_idx).rename_axis('dateTime').reset_index()
                return ovh_min['MAGF'].astype(float).values
        except Exception as e:
            print(f"[DEBUG OVH] load_ovh_for_range failed for {station_key}: {e}", flush=True)
        return None

    def _use_passed_array_safely(passed_arr):
        """Pad shorter arrays with NaN; truncate longer arrays to n."""
        if passed_arr is None:
            return np.full(n, np.nan)
        arr = np.asarray(passed_arr, dtype=float)
        if arr.size == 0:
            return np.full(n, np.nan)
        if arr.size == n:
            return arr
        if arr.size < n:
            pad = np.full(n - arr.size, np.nan, dtype=float)
            return np.concatenate([arr, pad])
        # longer -> truncate (prefer earliest samples)
        return arr[:n]

    # Cache per-station OVH arrays so two instruments from same station reuse same series
    _ovh_station_cache = {}

    # First station
    if firstStation.upper() not in _ovh_station_cache:
        built = _build_ovh_for_station(firstStation)
        if built is None:
            # fallback to passed-in array (if any)
            built = _use_passed_array_safely(first_Fm)
        _ovh_station_cache[firstStation.upper()] = np.asarray(built, dtype=float)

    # Second station: reuse cache if same station, otherwise build
    if secondStation.upper() not in _ovh_station_cache:
        built2 = _build_ovh_for_station(secondStation)
        if built2 is None:
            built2 = _use_passed_array_safely(second_Fm)
        _ovh_station_cache[secondStation.upper()] = np.asarray(built2, dtype=float)
    else:
        # reuse (no extra file checks/logging)
        pass

    # Assign the arrays
    first_Fm_full = _ovh_station_cache[firstStation.upper()]
    second_Fm_full = _ovh_station_cache[secondStation.upper()]

    # Safety: ensure shapes are length n
    if first_Fm_full.shape[0] != n:
        first_Fm_full = np.resize(first_Fm_full, n).astype(float)
    if second_Fm_full.shape[0] != n:
        second_Fm_full = np.resize(second_Fm_full, n).astype(float)

    print(f"[DEBUG OVH CACHE] station={firstStation.upper()} non-nan={np.isfinite(first_Fm_full).sum()}, "
        f"station2={secondStation.upper()} non-nan={np.isfinite(second_Fm_full).sum()}", flush=True)

    # ---------------------------------------------------------
    # Safe fill into DataFrames (fill only NaNs so we don't overwrite instrument values)
    # ---------------------------------------------------------
    # Prepare numeric columns
    firstData['MAGF'] = pd.to_numeric(firstData.get('MAGF', pd.Series(np.nan, index=firstData.index)), errors='coerce')
    firstData['F_meas'] = pd.to_numeric(firstData.get('F_meas', pd.Series(np.nan, index=firstData.index)), errors='coerce')
    secondData['MAGF'] = pd.to_numeric(secondData.get('MAGF', pd.Series(np.nan, index=secondData.index)), errors='coerce')
    secondData['F_meas'] = pd.to_numeric(secondData.get('F_meas', pd.Series(np.nan, index=secondData.index)), errors='coerce')

    mask1 = firstData['MAGF'].isna()
    if mask1.any():
        firstData.loc[mask1, 'MAGF'] = pd.Series(first_Fm_full, index=firstData.index)[mask1]
        firstData.loc[mask1, 'F_meas'] = pd.Series(first_Fm_full, index=firstData.index)[mask1]

    mask2 = secondData['MAGF'].isna()
    if mask2.any():
        secondData.loc[mask2, 'MAGF'] = pd.Series(second_Fm_full, index=secondData.index)[mask2]
        secondData.loc[mask2, 'F_meas'] = pd.Series(second_Fm_full, index=secondData.index)[mask2]

    
    # Ensure required columns exist
    for df in [firstData, secondData]:
        if 'dateTime' in df.columns:
            df['dateTime'] = pd.to_datetime(df['dateTime'], errors='coerce')
        else:
            df['dateTime'] = pd.to_datetime([])
        
        if 'F_meas' in df.columns and 'MAGF' not in df.columns:
            df['MAGF'] = pd.to_numeric(df['F_meas'], errors='coerce')
        if 'F_calc' in df.columns and 'Fc' not in df.columns:
            df['Fc'] = pd.to_numeric(df['F_calc'], errors='coerce')
        
        # Ensure dF is calculated properly
        if 'dF' not in df.columns:
            meas = pd.to_numeric(df.get('F_meas', pd.Series([np.nan] * len(df))), errors='coerce')
            calc = pd.to_numeric(df.get('Fc', pd.Series([np.nan] * len(df))), errors='coerce')
            df['dF'] = np.where(meas.notna() & calc.notna(), meas - calc, np.nan)
        
        for col in ('MAGH', 'MAGZ', 'MAGF', 'Fc', 'dF'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Replace sentinel/very-large values (catch float sentinels like 99999.99)
    for df in [firstData, secondData]:
        for comp in ('MAGH', 'MAGZ', 'MAGD', 'MAGF', 'F_meas'):
            if comp in df.columns:
                # anything >= 90000 treated as sentinel -> NaN
                df[comp] = df[comp].where(df[comp].abs() < 90000.0, np.nan)

    
    # Calculate means and normalize
    first_H_mean = firstData['MAGH'].mean()
    first_Z_mean = firstData['MAGZ'].mean()
    second_H_mean = secondData['MAGH'].mean()
    second_Z_mean = secondData['MAGZ'].mean()
    
    firstData['avgH'] = firstData['MAGH'] - first_H_mean
    secondData['avgH'] = secondData['MAGH'] - second_H_mean
    firstData['avgZ'] = firstData['MAGZ'] - first_Z_mean
    secondData['avgZ'] = secondData['MAGZ'] - second_Z_mean
    
    if 'MAGD' in firstData.columns and 'MAGD' in secondData.columns:
        first_D_mean = firstData['MAGD'].mean()
        second_D_mean = secondData['MAGD'].mean()
        firstData['avgD'] = firstData['MAGD'] - first_D_mean
        secondData['avgD'] = secondData['MAGD'] - second_D_mean
    
    # Apply bias if needed
    if applyBias:
        firstData['avgH'] *= 0.7
        firstData['avgZ'] *= 0.8
    
    # Calculate differences
    differences = pd.DataFrame()
    differences['dateTime'] = firstData['dateTime']
    differences['first_second_avgHDiff'] = firstData['avgH'] - secondData['avgH']
    differences['first_second_avgZDiff'] = firstData['avgZ'] - secondData['avgZ']
    
    if 'avgD' in firstData.columns and 'avgD' in secondData.columns:
        differences['first_second_avgDDiff'] = firstData['avgD'] - secondData['avgD']
    
    # Determine number of subplots
    has_D = 'avgD' in firstData.columns and 'avgD' in secondData.columns
    numSubplots = 6 if has_D else 5
    
    # Create figure and subplots
    fig, axs = plt.subplots(
        numSubplots, 1, sharex=True,
        figsize=(10, 12),
        gridspec_kw={'hspace': 0.08},
        dpi=100
    )
    
    # Plot data gaps
    for start_gap, end_gap in gap_intervals:
        for ax in axs:
            ax.axvspan(start_gap, end_gap, color='gray', alpha=0.3, zorder=0)

    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=2, label=firstLabel),
        plt.Line2D([0], [0], color='red', linewidth=2, label=secondLabel),
        plt.Line2D([0], [0], color='green', linewidth=2, label='Diff H'),
        plt.Line2D([0], [0], color='orange', linewidth=2, label='Diff D'),
        plt.Line2D([0], [0], color='purple', linewidth=2, label='Diff Z')
    ]
    
    # Plot H component
    axs[0].plot(firstData['dateTime'], firstData['avgH'], color='blue')
    axs[0].plot(secondData['dateTime'], secondData['avgH'], color='red')
    # Create a single legend at the top of the first subplot
    axs[0].legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 1.3), ncol=5, frameon=False,
                  columnspacing=1.0, handlelength=2.5, handletextpad=0.5)
    axs[0].grid(True)
    axs[0].minorticks_on()
    axs[0].grid(True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axs[0].set_ylabel('H (nT)')
    
    # Plot D component if available
    if has_D:
        axs[1].plot(firstData['dateTime'], firstData['avgD'], color='blue')
        axs[1].plot(secondData['dateTime'], secondData['avgD'], color='red')
        axs[1].set_ylabel('D (min)')
        axs[1].minorticks_on()
        axs[1].grid(True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        axs[1].grid(True)
    
    # Plot Z component
    z_plot_idx = 2 if has_D else 1
    axs[z_plot_idx].plot(firstData['dateTime'], firstData['avgZ'], color='blue')
    axs[z_plot_idx].plot(secondData['dateTime'], secondData['avgZ'], color='red')
    axs[z_plot_idx].grid(True)
    axs[z_plot_idx].minorticks_on()
    axs[z_plot_idx].grid(True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axs[z_plot_idx].set_ylabel('Z (nT)')
    
    # Plot F component
    # Plot F component (ONLY OVH data)
    f_plot_idx = z_plot_idx + 1

    # Get OVH series directly from the cached arrays
    first_ovh_series = pd.Series(first_Fm_full, index=full_idx)
    second_ovh_series = pd.Series(second_Fm_full, index=full_idx)

    # Normalize OVH data for plotting
    if first_ovh_series.notna().any():
        first_mean = first_ovh_series.mean()
        first_F_normalized = first_ovh_series - first_mean
    else:
        first_F_normalized = first_ovh_series

    if second_ovh_series.notna().any():
        second_mean = second_ovh_series.mean()
        second_F_normalized = second_ovh_series - second_mean
    else:
        second_F_normalized = second_ovh_series

    # If same station, they should be identical after normalization
    if firstStation.upper() == secondStation.upper():
        print(f"[DEBUG F PLOT] Same station ({firstStation}), OVH signals should overlap perfectly", flush=True)

    # Plot OVH F data directly
    axs[f_plot_idx].plot(full_idx, first_F_normalized, color='blue', label=firstLabel)
    axs[f_plot_idx].plot(full_idx, second_F_normalized, color='red', label=secondLabel)

    axs[f_plot_idx].set_ylabel('F (nT) OVH')
    axs[f_plot_idx].grid(True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axs[f_plot_idx].minorticks_on()
    axs[f_plot_idx].grid(True)
    
    # Plot dF component only if we have valid dF data
    df_plot_idx = f_plot_idx + 1

    # Check if we have valid dF data to plot
    def has_meaningful_dF(df):
        if 'dF' not in df.columns or 'F_meas' not in df.columns:
            return False
        # Only consider dF valid if F_meas also exists at that time
        valid_mask = df['F_meas'].notna() & df['dF'].notna()
        return valid_mask.any()

    first_has_dF = has_meaningful_dF(firstData)
    second_has_dF = has_meaningful_dF(secondData)

    if first_has_dF or second_has_dF:
        # Create normalized dF series WITH NaN gaps preserved
        
        if first_has_dF:
            # Create a normalized copy of dF with NaN gaps preserved
            first_dF_normalized = firstData['dF'].copy()
            # Only normalize where we have valid dF
            valid_first = firstData['F_meas'].notna() & firstData['dF'].notna()
            if valid_first.any():
                first_dF_mean = firstData.loc[valid_first, 'dF'].mean()
                first_dF_normalized[valid_first] = firstData.loc[valid_first, 'dF'] - first_dF_mean
            # Plot the full series (matplotlib will skip NaN segments)
            axs[df_plot_idx].plot(firstData['dateTime'], first_dF_normalized, color='blue')
        
        if second_has_dF:
            # Create a normalized copy of dF with NaN gaps preserved
            second_dF_normalized = secondData['dF'].copy()
            # Only normalize where we have valid dF
            valid_second = secondData['F_meas'].notna() & secondData['dF'].notna()
            if valid_second.any():
                second_dF_mean = secondData.loc[valid_second, 'dF'].mean()
                second_dF_normalized[valid_second] = secondData.loc[valid_second, 'dF'] - second_dF_mean
            # Plot the full series (matplotlib will skip NaN segments)
            axs[df_plot_idx].plot(secondData['dateTime'], second_dF_normalized, color='red')
        
        axs[df_plot_idx].set_ylabel('dF (nT)')
        axs[df_plot_idx].grid(True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        axs[df_plot_idx].minorticks_on()
        axs[df_plot_idx].grid(True)
    else:
        # If no dF data, hide this subplot
        axs[df_plot_idx].set_visible(False)
    
    # Plot differences
    diff_plot_idx = df_plot_idx + 1
    axs[diff_plot_idx].plot(differences['dateTime'], differences['first_second_avgHDiff'], color='green')
    
    if 'first_second_avgDDiff' in differences.columns:
        axs[diff_plot_idx].plot(differences['dateTime'], differences['first_second_avgDDiff'], color='orange')
    
    axs[diff_plot_idx].plot(differences['dateTime'], differences['first_second_avgZDiff'], color='purple')
    axs[diff_plot_idx].set_ylabel('Diff')
    axs[diff_plot_idx].minorticks_on()
    axs[diff_plot_idx].grid(True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    axs[diff_plot_idx].grid(True)
    
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
        
    elif span_days <= 5:
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
    plt.suptitle(
        f"{firstLabel} vs {secondLabel}",
        fontsize=12,
        y=0.95
    )
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
    """Placeholder save function for GUI integration.

    Note: In GUI mode, plots are displayed rather than saved. The GUI
    handles save functionality through its own file dialog.

    Args:
        fig: Matplotlib figure object.
        station: Station code.
        instrument: Instrument name.
        start_date: Start date string.
        end_date: End date string.
    """
    print('Warning: save_plot called but plots are displayed in GUI', flush=True)
    return None


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def processFiles(
    base_path: str,
    first_station: str,
    first_instrument: str,
    second_station: str,
    second_instrument: str,
    start_date: str,
    end_date: str,
    start_time: str,
    end_time: str,
    mode: str = 'plot'
) -> plt.Figure:
    """Main processing function for multi-station geomagnetic data analysis.

    Orchestrates the complete data processing pipeline:
        1. Load data from FTP for both stations/instruments
        2. Handle different data formats (FGM, CTU, DTU, L251, OVH)
        3. Apply format-specific processing and conversions
        4. Aggregate data to consistent time resolution
        5. Load OVH total field data with station-level caching
        6. Apply baseline corrections
        7. Generate comparison plots

    Supported instruments:
        - FGM1/FGM2: Fluxgate magnetometers (minute data)
        - DTU: DI-flux theodolite data
        - L251: L025-type magnetometer
        - CTU: CTU format (requires conversion)
        - OVH: Overhauser magnetometer (F component only)

    Args:
        base_path: Base directory for local file operations.
        first_station: First station code (e.g., 'HER', 'HBK').
        first_instrument: First instrument name (e.g., 'FGM1', 'DTU').
        second_station: Second station code.
        second_instrument: Second instrument name.
        start_date: Start date string (format: 'YYYY-MM-DD').
        end_date: End date string (format: 'YYYY-MM-DD').
        start_time: Start time string (format: 'HH:MM').
        end_time: End time string (format: 'HH:MM').
        mode: Operation mode - 'plot', 'convert', or 'both' (default: 'plot').

    Returns:
        Matplotlib figure object containing the comparison plot.

    Raises:
        RuntimeError: If no data found for specified station/instrument/date range.
        Exception: Various exceptions for file I/O, parsing, or processing errors.
    """
        
    first_Fm = []
    second_Fm = []
    
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        def load_inst_df(station, instrument, dates):
            lst = []
            ftp_handler = get_ftp_handler()
            
            for single_date in pd.date_range(*dates):
                ds = single_date.strftime("%Y%m%d")
                
                # Update these conditionals to handle versioned instrument names
                if instrument.upper().startswith("FGM"):  # This already works for FGM1, FGM2
                    remote_path = ftp_handler.construct_remote_path(station, instrument, single_date)
                    
                    if ftp_handler.file_exists(remote_path):
                        local_path = ftp_handler.download_file(remote_path)
                        if local_path:
                            try:
                                df = load_fgm(local_path, station_name=station, instrument_name=instrument)
                            finally:
                                if os.path.exists(local_path):
                                    os.remove(local_path)
                        else:
                            df = None
                    else:
                        df = None
                        
                elif instrument.upper().startswith('DTU'):
                    remote_path = ftp_handler.construct_remote_path(station, instrument, single_date)
                    
                    if ftp_handler.file_exists(remote_path):
                        local_path = ftp_handler.download_file(remote_path)
                        if local_path:
                            try:
                                df = pd.read_csv(local_path, sep=r'\s+', skiprows=15, engine='python')
                                if 'DATE' in df.columns and 'TIME' in df.columns:
                                    df['dateTime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
                            except Exception as e:
                                print(f"Error reading DTU file {remote_path}: {e}")
                                df = None
                            finally:
                                if os.path.exists(local_path):
                                    os.remove(local_path)
                        else:
                            df = None
                    else:
                        print(f"DTU file not found: {remote_path}")
                        df = None
                        
                elif instrument.upper() == 'L251':  # This should work as is
                    remote_path = ftp_handler.construct_remote_path(station, instrument, single_date)
                    
                    if ftp_handler.file_exists(remote_path):
                        local_path = ftp_handler.download_file(remote_path)
                        if local_path:
                            try:
                                tmp = pd.read_csv(local_path, sep=r',\s*', header=None, engine='python')
                                if tmp.shape[1] >= 4:
                                    tmp.columns = ["dateTime_str", "MAGH", "MAGD", "MAGZ"] + [f"_rest{i}" for i in range(1, tmp.shape[1] - 3)]
                                    tmp['dateTime'] = pd.to_datetime(tmp['dateTime_str'], format="%Y\\%m\\%d %H:%M:%S")
                                    df = tmp
                                else:
                                    df = None
                            except Exception as e:
                                print(f"Error reading L251 file {remote_path}: {e}")
                                df = None
                            finally:
                                if os.path.exists(local_path):
                                    os.remove(local_path)
                        else:
                            df = None
                    else:
                        df = None
                    

                elif instrument.upper().startswith("CTU"):
                    # CTU processing branch  
                    remote_path = ftp_handler.construct_remote_path(station, instrument, single_date)
                    if ftp_handler.file_exists(remote_path):
                        local_path = ftp_handler.download_file(remote_path)
                        if local_path:
                            try:
                                # Pass save_file=True to enable conversion when requested
                                outdf, fig = process_ctu_file(local_path, base_path, save_file=(mode == 'convert'), plot=False, date_str=single_date.strftime('%Y-%m-%d'))
                                if outdf is not None:
                                    outdf['dateTime'] = pd.to_datetime(outdf['DATE'] + ' ' + outdf['TIME'])
                                    tmp = outdf[['dateTime', 'H', 'D', 'Z']].rename(columns={'H': 'MAGH', 'D': 'MAGD', 'Z': 'MAGZ'})
                                    for c in ('MAGH', 'MAGD', 'MAGZ'):
                                        if c not in tmp.columns: tmp[c] = np.nan
                                    df = tmp.copy()
                            except Exception as e:
                                print(f"Error processing CTU file {remote_path}: {e}", flush=True)
                                df = None
                            finally:
                                try:
                                    if os.path.exists(local_path): os.remove(local_path)
                                except Exception: pass
                    else:
                        df = None
                else:
                    df = None
                
                if df is not None:
                    # Remove duplicates and reindex (original logic)
                    df = df.loc[~df["dateTime"].duplicated()].copy()
                    times = df["dateTime"].sort_values().unique()
                    df = df.set_index("dateTime").reindex(times).reset_index()
                    lst.append(df)
            
            if not lst:
                raise RuntimeError(f"No {instrument} data for {station} in the specified date range")
            
            full = pd.concat(lst, ignore_index=True)
            
            # Rename columns to standard names (original logic)
            rename = {c: 'MAG'+c[-1] for c in full.columns if c[-1] in ['H','Z','D']}
            full.rename(columns=rename, inplace=True)
            
            return full
        
        # Load both datasets
        firstData = load_inst_df(first_station, first_instrument, (start_dt, end_dt))
        secondData = load_inst_df(second_station, second_instrument, (start_dt, end_dt))
        
        # Handle different data resolutions (FGM vs others)
        first_is_fgm = first_instrument.upper().startswith('FGM')
        second_is_fgm = second_instrument.upper().startswith('FGM')
        
        def infer_median_cad_seconds_df(df):
            try:
                diffs = pd.to_datetime(df['dateTime']).sort_values().diff().dropna().dt.total_seconds()
                return float(diffs.median()) if len(diffs) else None
            except Exception:
                return None
        
        cad_first = infer_median_cad_seconds_df(firstData) if (firstData is not None and 'dateTime' in firstData.columns) else None
        cad_second = infer_median_cad_seconds_df(secondData) if (secondData is not None and 'dateTime' in secondData.columns) else None
        
        # Aggregate seconds to minutes if needed
        need_agg = ((first_is_fgm and cad_second is not None and cad_second < 60) or
                   (second_is_fgm and cad_first is not None and cad_first < 60))
        
        # Handle data resolution - FGM data should NOT go through aggressive processing
        need_agg = False
        first_min = None
        second_min = None

        # Only aggregate non-FGM seconds data
        if not first_is_fgm and cad_first is not None and cad_first < 60:
            print(f"[INFO] Aggregating {first_station} seconds data to minutes", flush=True)
            # Use conservative cleaning for non-FGM data
            first_clean = firstData.copy()
            for col in ('MAGH', 'MAGZ', 'MAGF'):  # Skip MAGD from aggressive cleaning
                if col in first_clean.columns:
                    first_clean[col] = _mask_sentinels_and_extremes(first_clean[col])
            # Only basic sentinel masking for D component
            if 'MAGD' in first_clean.columns:
                first_clean['MAGD'] = _mask_sentinels_and_extremes(first_clean['MAGD'])
            
            first_min = minute_aggregate_exclude_nans(first_clean, ts_col='dateTime')
            if 'dateTime' in first_min.columns:
                first_min['dateTime'] = pd.to_datetime(first_min['dateTime'], errors='coerce')

        if not second_is_fgm and cad_second is not None and cad_second < 60:
            print(f"[INFO] Aggregating {second_station} seconds data to minutes", flush=True)
            # Use conservative cleaning for non-FGM data
            second_clean = secondData.copy()
            for col in ('MAGH', 'MAGZ', 'MAGF'):  # Skip MAGD from aggressive cleaning
                if col in second_clean.columns:
                    second_clean[col] = _mask_sentinels_and_extremes(second_clean[col])
            # Only basic sentinel masking for D component
            if 'MAGD' in second_clean.columns:
                second_clean['MAGD'] = _mask_sentinels_and_extremes(second_clean['MAGD'])
            
            second_min = minute_aggregate_exclude_nans(second_clean, ts_col='dateTime')
            if 'dateTime' in second_min.columns:
                second_min['dateTime'] = pd.to_datetime(second_min['dateTime'], errors='coerce')

        # Replace data with aggregated versions for non-FGM seconds data
        if first_min is not None:
            firstData = first_min.copy()
        if second_min is not None:
            secondData = second_min.copy()

        # Now handle FGM data with simple reindexing (like single script)
        start_ts = pd.to_datetime(f"{start_date} {start_time}")
        end_ts = pd.to_datetime(f"{end_date} {end_time}")
        target_idx = pd.date_range(start=start_ts, end=end_ts, freq='60s')

        if first_is_fgm and firstData is not None and not firstData.empty:
            if 'dateTime' in firstData.columns:
                firstData = firstData.set_index('dateTime')
                firstData = firstData.reindex(target_idx)
                firstData = firstData.reset_index().rename(columns={'index': 'dateTime'})
                print(f"[DEBUG] Reindexed first FGM data to {len(firstData)} minute records", flush=True)

        if second_is_fgm and secondData is not None and not secondData.empty:
            if 'dateTime' in secondData.columns:
                secondData = secondData.set_index('dateTime')
                secondData = secondData.reindex(target_idx)
                secondData = secondData.reset_index().rename(columns={'index': 'dateTime'})
                print(f"[DEBUG] Reindexed second FGM data to {len(secondData)} minute records", flush=True)
        
        # Ensure datetime columns
        for df in [firstData, secondData]:
            if 'dateTime' not in df.columns and df.index.name == 'dateTime':
                df.reset_index(inplace=True)
            if 'dateTime' in df.columns:
                df['dateTime'] = pd.to_datetime(df['dateTime'], errors='coerce')
        
        # Create target minute index
        target_idx = pd.date_range(
            start=pd.to_datetime(f"{start_date} {start_time}"),
            end=pd.to_datetime(f"{end_date} {end_time}"),
            freq='60s'
        )
        
        # Reindex to target timeline
        if firstData is not None and not firstData.empty:
            if 'dateTime' in firstData.columns:
                firstData = firstData.set_index('dateTime').reindex(target_idx).rename_axis('dateTime').reset_index()
        
        if secondData is not None and not secondData.empty:
            if 'dateTime' in secondData.columns:
                secondData = secondData.set_index('dateTime').reindex(target_idx).rename_axis('dateTime').reset_index()
        
        # Load OVH data for F values using FTP

        # Cache OVH data per station to avoid redundant loading
        _ovh_cache = {}

        def get_cached_ovh(station, start_dt, end_dt, base_path):
            """Get OVH data from cache or load it"""
            station_key = station.upper()
            if station_key not in _ovh_cache:
                ovh_df = load_ovh_for_range(station_key, start_dt, end_dt, base_path)
                if not ovh_df.empty:
                    # Aggregate to minutes
                    ovh_min = minute_aggregate_exclude_nans(ovh_df[['dateTime', 'MAGF']], ts_col='dateTime')
                    ovh_min['MAGF'] = pd.to_numeric(ovh_min['MAGF'], errors='coerce')
                    # Store the raw minute data
                    _ovh_cache[station_key] = ovh_min
                else:
                    _ovh_cache[station_key] = pd.DataFrame(columns=['dateTime', 'MAGF'])
            
            return _ovh_cache[station_key]

        # Load OVH data for F values using FTP - USE CACHED VERSION
        ovh_min = get_cached_ovh(first_station, start_dt, end_dt, base_path)
            
        if not ovh_min.empty:
            # Reindex to target timeline
            ovh_min = ovh_min.set_index('dateTime').reindex(target_idx).rename_axis('dateTime').reset_index()
            if 'MAGF' in ovh_min.columns:
                ovh_min = ovh_min.rename(columns={'MAGF': 'F_meas'})
            else:
                ovh_min['F_meas'] = np.nan
        else:
            ovh_min = pd.DataFrame({'dateTime': target_idx, 'F_meas': [np.nan]*len(target_idx)})

        # Build ovh_min for the second station as well - USE CACHED VERSION
        ovh_min_second = get_cached_ovh(second_station, start_dt, end_dt, base_path)

        if not ovh_min_second.empty and 'MAGF' in ovh_min_second.columns:
            ovh_min_second = minute_aggregate_exclude_nans(ovh_min_second[['dateTime','MAGF']], ts_col='dateTime')
            ovh_min_second = ovh_min_second.rename(columns={'MAGF':'F_meas'})
            # Reindex to target timeline
            ovh_min_second = ovh_min_second.set_index('dateTime').reindex(target_idx).rename_axis('dateTime').reset_index()
        else:
            ovh_min_second = pd.DataFrame()
            
        # Apply baselines and compute F values
        def finalize_df_F(df, station, instrument, ovh_min=None):
            if df is None:
                return df
            
            if 'dateTime' not in df.columns and pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.reset_index().rename(columns={'index': 'dateTime'})
            
            if 'dateTime' in df.columns:
                df['dateTime'] = pd.to_datetime(df['dateTime'], errors='coerce')
            
            for c in ('MAGH', 'MAGD', 'MAGZ', 'MAGF', 'F_meas', 'F_calc'):
                if c not in df.columns:
                    df[c] = np.nan
                else:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
        
            for c in ('MAGH','MAGD','MAGZ','MAGF','F_meas'):
                df[c] = _mask_sentinels_and_extremes(df[c])
                    
            try:
                H0, D0, Z0 = loadBaselineValues(station, instrument.upper())
            except Exception:
                H0, D0, Z0 = 0.0, 0.0, 0.0
            
            df['MAGH'] = df['MAGH'] + float(H0)
            df['MAGD'] = df['MAGD'] + float(D0)
            df['MAGZ'] = df['MAGZ'] + float(Z0)
            
            denom = df['MAGH'].abs().replace({0.0: np.nan})
            df['MAGD'] = np.where(denom.notna(), df['MAGD'] * 3438.0 / denom, np.nan)
            
            df['F_calc'] = np.sqrt(df['MAGH'].pow(2) + df['MAGZ'].pow(2))
            df['Fc'] = df['F_calc']
            
            df['MAGF'] = pd.to_numeric(df.get('MAGF'), errors='coerce')
            existing_fmeas = pd.to_numeric(df.get('F_meas'), errors='coerce')
            df['F_meas'] = existing_fmeas.combine_first(df['MAGF'])

            # mask sentinel / absurd values just in case
            df['MAGF'] = _mask_sentinels_and_extremes(df['MAGF'])
            df['F_meas'] = _mask_sentinels_and_extremes(df['F_meas'])
            
            if df['F_meas'].isna().all() and ovh_min is not None and not ovh_min.empty:
                if 'dateTime' in ovh_min.columns:
                    ovh_min['dateTime'] = pd.to_datetime(ovh_min['dateTime'], errors='coerce')
                if 'F_meas' not in ovh_min.columns and 'MAGF' in ovh_min.columns:
                    ovh_min = ovh_min.rename(columns={'MAGF': 'F_meas'})
                
                ovh_min = ovh_min.set_index('dateTime')
                ovh_min['F_meas'] = pd.to_numeric(ovh_min['F_meas'], errors='coerce')
                
                if 'dateTime' in df.columns:
                    df = df.set_index(pd.to_datetime(df['dateTime'], errors='coerce'))
                
                try:
                    ov_filled = ovh_min['F_meas'].reindex(df.index, method='ffill')
                    df['F_meas'] = df['F_meas'].combine_first(ov_filled)
                except Exception:
                    pass
                
                if 'dateTime' not in df.columns:
                    df = df.reset_index().rename(columns={'index': 'dateTime'})
            
            df['dF'] = np.where(df['F_meas'].notna() & df['Fc'].notna(), df['F_meas'] - df['Fc'], np.nan)
            return df
        
        # Build ovh_min for the second station as well (so F_meas fallback is symmetric)
        try:
            ovh_min_second = load_ovh_for_range(second_station.upper(), start_dt, end_dt, base_path)
            if not ovh_min_second.empty and 'MAGF' in ovh_min_second.columns:
                ovh_min_second = minute_aggregate_exclude_nans(ovh_min_second[['dateTime','MAGF']], ts_col='dateTime')
                ovh_min_second = ovh_min_second.rename(columns={'MAGF':'F_meas'})
            else:
                ovh_min_second = pd.DataFrame()
        except Exception:
            ovh_min_second = pd.DataFrame()
        
        firstData = finalize_df_F(firstData, first_station, first_instrument, ovh_min=ovh_min)
        secondData = finalize_df_F(secondData, second_station, second_instrument, ovh_min=ovh_min_second)
        
        # Apply time filter
        start_ts = pd.to_datetime(f"{start_date} {start_time}")
        end_ts = pd.to_datetime(f"{end_date} {end_time}")
        
        if 'dateTime' in firstData.columns:
            firstData = firstData[(firstData['dateTime'] >= start_ts) & (firstData['dateTime'] <= end_ts)]
        if 'dateTime' in secondData.columns:
            secondData = secondData[(secondData['dateTime'] >= start_ts) & (secondData['dateTime'] <= end_ts)]

        '''        
        # Load OVH F values using FTP for both stations
        ftp_handler = get_ftp_handler()
        first_Fm = []
        second_Fm = []
        
        for single_date in pd.date_range(start_dt, end_dt):
            # Load first station OVH data
            remote_path1 = ftp_handler.construct_remote_path(first_station, 'OVH', single_date)
            if ftp_handler.file_exists(remote_path1):
                local_path = ftp_handler.download_file(remote_path1)
                if local_path:
                    try:
                        df1 = pd.read_csv(local_path, sep=r'\s+', header=None, engine='python')
                        if df1.shape[1] >= 2:
                            first_Fm.extend(pd.to_numeric(df1.iloc[:, 1], errors='coerce'))
                    finally:
                        if os.path.exists(local_path):
                            os.remove(local_path)
            
            # Load second station OVH data  
            remote_path2 = ftp_handler.construct_remote_path(second_station, 'OVH', single_date)
            if ftp_handler.file_exists(remote_path2):
                local_path = ftp_handler.download_file(remote_path2)
                if local_path:
                    try:
                        df2 = pd.read_csv(local_path, sep=r'\s+', header=None, engine='python')
                        if df2.shape[1] >= 2:
                            second_Fm.extend(pd.to_numeric(df2.iloc[:, 1], errors='coerce'))
                    finally:
                        if os.path.exists(local_path):
                            os.remove(local_path)
        '''        
        # Create output filename
        applyBias = (second_station.lower() == "sqd")

        
        firstLabel = f"{first_station.upper()}{first_instrument}"
        secondLabel = f"{second_station.upper()}{second_instrument}"
        
        # Generate plot
        fig = plotData(
            firstData, first_station,
            secondData, second_station,
            firstLabel, secondLabel,
            f"{start_date} {start_time}",
            f"{end_date} {end_time}",
            None, applyBias,
            [], []
        )

        return fig
        
    except Exception:
        traceback.print_exc()
        raise


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process and plot multi-station/instrument magnetic field data.'
    )
    parser.add_argument(
        'base_path',
        help='Base directory containing data files'
    )
    parser.add_argument(
        'first_station',
        help='First station code (e.g., HER)'
    )
    parser.add_argument(
        'first_instrument',
        help='First instrument (e.g., OVH, L251, FGM1)'
    )
    parser.add_argument(
        'second_station',
        help='Second station code (e.g., HBK)'
    )
    parser.add_argument(
        'second_instrument',
        help='Second instrument (e.g., OVH, L251, FGM1)'
    )
    parser.add_argument(
        'start_date',
        help='Start date in YYYY-MM-DD format'
    )
    parser.add_argument(
        'end_date',
        help='End date in YYYY-MM-DD format'
    )
    parser.add_argument(
        'start_time',
        help='Start time in HH:MM format'
    )
    parser.add_argument(
        'end_time',
        help='End time in HH:MM format'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='plot',
        choices=['plot', 'convert', 'both'],
        help='Operation mode: plot, convert, or both'
    )

    args = parser.parse_args()

    processFiles(
        args.base_path,
        args.first_station,
        args.first_instrument,
        args.second_station,
        args.second_instrument,
        args.start_date,
        args.end_date,
        args.start_time,
        args.end_time,
        mode=args.mode
    )
