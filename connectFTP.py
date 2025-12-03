"""FTP data handler for SANSA magnetic instrument data.

This module provides FTP connectivity to SANSA (South African National Space Agency)
data servers for downloading geomagnetic instrument data files. It handles connection
management, file path construction for various instrument types, and automatic
reconnection on connection loss.

Supported instruments:
    - OVH: Overhauser magnetometer (scalar F measurements)
    - DTU: DTU-Space fluxgate magnetometer
    - FGM1/FGM2: Fluxgate magnetometers
    - L251: LEMI-025 magnetometer
    - SQD: SQUID magnetometer
    - CTU: CTU magnetometer
"""

import ftplib
import os
import re
import tempfile
from datetime import datetime
from typing import Optional

# FTP Configuration
FTP_CONFIG: dict[str, str] = {
    'host': 'storage4.her.sansa.org.za',
    'username': 'anonymous',
    'password': 'dap@sansa.org.za',
    'base_path': '/SANSA_INSTRUMENTS'
}

# Station version mapping - defines year ranges for each instrument version
STATION_VERSIONS: dict[str, dict[str, dict[str, tuple[int, int]]]] = {
    'HBK': {
        'OVH': {
            'HBKOVH1': (2008, 2018),
            'HBKOVH2': (2010, 2017),
            'HBKOVH3': (2018, 2025),
            'HBKOVH4': (2018, 2025),
        }
    },
    'KMH': {
        'OVH': {
            'KMHOVH1': (2006, 2025),
            'KMHOVH2': (2010, 2025),
        }
    },
    'HER': {
        'OVH': {
            'HEROVH1': (2009, 2025),
            'HEROVH2': (2019, 2025),
        }
    },
    'TSU': {
        'OVH': {
            'TSUOVH1': (2006, 2025),
            'TSUOVH2': (2010, 2025),
        }
    },
    'SNA': {
        'OVH': {
            'SNAOVH1': (2008, 2014),
            'SNAOVH2': (2015, 2025),
        }
    }
}


class FTPDataHandler:
    """Handler for FTP connections to SANSA data servers.

    This class manages FTP connections with automatic reconnection on failure
    and provides methods for navigating the SANSA instrument data directory
    structure.

    Attributes:
        ftp: The underlying FTP connection object.
        last_activity: Timestamp of the last FTP activity for staleness detection.
    """

    def __init__(self) -> None:
        """Initialize the FTP handler and establish connection."""
        self.ftp: Optional[ftplib.FTP] = None
        self.last_activity: Optional[datetime] = None
        self.connect()

    def connect(self) -> None:
        """Establish FTP connection with retry logic.

        Raises:
            Exception: If connection cannot be established after retries.
        """
        try:
            if self.ftp:
                try:
                    self.ftp.quit()
                except Exception:
                    pass
                self.ftp = None

            self.ftp = ftplib.FTP(FTP_CONFIG['host'])
            self.ftp.login(FTP_CONFIG['username'], FTP_CONFIG['password'])
            self.ftp.cwd(FTP_CONFIG['base_path'])
            self.last_activity = datetime.now()
            print('FTP connection established successfully')
        except Exception as e:
            print(f'FTP connection failed: {e}')
            self.ftp = None
            raise

    def ensure_connection(self) -> None:
        """Ensure FTP connection is alive, reconnect if needed.

        Checks if the connection is stale (no activity for 30 seconds) or
        lost, and reconnects if necessary.
        """
        try:
            # Check if connection is stale (no activity for 30 seconds)
            if (self.last_activity and
                    (datetime.now() - self.last_activity).total_seconds() > 30):
                print('FTP connection stale, reconnecting...')
                self.connect()
                return

            # Test connection with a simple command
            if self.ftp:
                self.ftp.voidcmd('NOOP')
                self.last_activity = datetime.now()
        except Exception:
            print('FTP connection lost, reconnecting...')
            self.connect()

    def get_ovh_version(self, station: str, date: datetime) -> Optional[str]:
        """Determine the appropriate OVH instrument version for a given date.

        Selects the OVH version by preferring the lowest-numbered version
        (OVH1 first). For a specific date, tries OVH1, then OVH2, etc.,
        returning the first version that has data for that date.

        Args:
            station: Station code (e.g., 'HER', 'HBK').
            date: Date for which to find the OVH version.

        Returns:
            Version name string (e.g., 'HEROVH1') or None if not found.
        """
        self.ensure_connection()
        year = date.year
        month = f'{date.month:02d}'

        # Build possible_versions based on STATION_VERSIONS (or default list)
        possible_versions: list[str] = []
        station_data = STATION_VERSIONS.get(station.upper(), {})
        ovh_versions = station_data.get('OVH', {})

        # Collect version names and filter by year range
        for version_name, (start_year, end_year) in ovh_versions.items():
            if start_year <= year <= end_year:
                possible_versions.append(version_name)

        if not possible_versions:
            # Fallback defaults if no mapping present
            possible_versions = [f'{station.upper()}OVH{v}' for v in [1, 2, 3, 4]]

        # Sort by trailing integer (OVH1, OVH2, ...) ascending
        def _ver_num(vname: str) -> int:
            """Extract trailing digits from version name."""
            m = re.search(r'(\d+)$', vname)
            return int(m.group(1)) if m else 9999

        possible_versions = sorted(possible_versions, key=_ver_num)

        # Return the first version that actually exists for this date
        for version_name in possible_versions:
            remote_path = (
                f'OVH/{version_name}/raw/{year}/{month}/'
                f'{version_name}-{date.strftime("%Y%m%d")}'
            )
            if self.file_exists(remote_path):
                return version_name

        return None

    def get_station_version(
        self,
        station: str,
        instrument: str,
        date: datetime
    ) -> str:
        """Determine station version string for path construction.

        Only OVH uses date-based version selection; other instruments
        use the user-selected instrument name directly.

        Args:
            station: Station code (e.g., 'HER', 'HBK').
            instrument: Instrument type (e.g., 'OVH', 'DTU1', 'FGM1').
            date: Date for version determination.

        Returns:
            Version string for path construction (e.g., 'HEROVH1', 'HERFGM1').
        """
        if instrument.upper() == 'OVH':
            return self.get_ovh_version(station, date) or f'{station.upper()}OVH1'
        else:
            return f'{station.upper()}{instrument}'

    def construct_remote_path(
        self,
        station: str,
        instrument: str,
        date: datetime
    ) -> str:
        """Construct the remote file path for a given station, instrument, and date.

        Handles the different directory structures used by various instruments
        including special cases for SQD/CTU data.

        Args:
            station: Station code (e.g., 'HER', 'HBK', 'SNA').
            instrument: Instrument type (e.g., 'OVH', 'DTU1', 'FGM1', 'SQD').
            date: Date of the data file.

        Returns:
            Remote path string relative to the base SANSA_INSTRUMENTS directory.
        """
        self.ensure_connection()
        year = date.year
        month = f'{date.month:02d}'
        day = f'{date.day:02d}'

        # Handle HER SQD and CTU directly (no station-version subfolders)
        if station.upper() == 'HER' and instrument.upper() in ('SQD', 'CTU'):
            instrument_folder = 'SQD'
            ext = 'squid' if instrument.upper() == 'SQD' else 'ctumag'
            filename = f'{date.strftime("%Y-%m-%d")}.{ext}'
            return f'{instrument_folder}/{filename}'

        station_version = self.get_station_version(station, instrument, date)

        if instrument.upper() == 'DTU1':
            filename = f'{station.lower()}{date.strftime("%Y%m%d")}vsec.sec'
            remote_path = f'DTU/{station_version}/raw/{year}/{month}/{day}/{filename}'

        elif instrument.upper() == 'L251':
            filename = f'{station_version}-{date.strftime("%Y%m%d")}'
            remote_path = f'L25/{station_version}/raw/{year}/{month}/{filename}'

        elif instrument.upper().startswith('FGM'):
            filename = f'{station_version}-{date.strftime("%Y%m%d")}'
            remote_path = f'FGM/{station_version}/raw/{year}/{month}/{filename}'

        elif instrument.upper() == 'OVH':
            filename = f'{station_version}-{date.strftime("%Y%m%d")}'
            remote_path = f'OVH/{station_version}/raw/{year}/{month}/{filename}'

        else:
            instrument_ftp = instrument.upper()
            filename = f'{station_version}-{date.strftime("%Y%m%d")}'
            remote_path = f'{instrument_ftp}/{station_version}/raw/{year}/{month}/{filename}'

        return remote_path

    def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists on the FTP server.

        Uses multiple methods for reliability: SIZE command, NLST directory
        listing, and RETR attempt.

        Args:
            remote_path: Path to check relative to the base directory.

        Returns:
            True if file exists, False otherwise.
        """
        self.ensure_connection()
        try:
            # Method 1: Try SIZE command (fastest if supported)
            try:
                self.ftp.size(remote_path)
                self.last_activity = datetime.now()
                return True
            except Exception:
                pass

            # Method 2: Use NLST to list directory and check for file
            directory = '/'.join(remote_path.split('/')[:-1])
            filename = remote_path.split('/')[-1]

            try:
                files = self.ftp.nlst(directory)
                for file in files:
                    if filename in file:
                        self.last_activity = datetime.now()
                        return True
            except Exception:
                pass

            # Method 3: Try to retrieve the file (most reliable but slowest)
            try:
                self.ftp.voidcmd('TYPE I')
                socket = self.ftp.transfercmd(f'RETR {remote_path}')
                socket.close()
                self.ftp.voidresp()
                self.last_activity = datetime.now()
                return True
            except Exception:
                return False

        except Exception as e:
            print(f'File existence check failed for {remote_path}: {e}')
            return False

    def find_file_in_directory(
        self,
        directory: str,
        pattern: str
    ) -> list[str]:
        """Find files in a directory matching a pattern.

        Args:
            directory: Remote directory path to search.
            pattern: Substring pattern to match in filenames.

        Returns:
            List of matching file paths.
        """
        self.ensure_connection()
        try:
            files = self.ftp.nlst(directory)
            self.last_activity = datetime.now()
            matching_files = [f for f in files if pattern in f]
            return matching_files
        except Exception:
            return []

    def download_file(
        self,
        remote_path: str,
        local_path: Optional[str] = None
    ) -> Optional[str]:
        """Download a file from FTP to local storage.

        Args:
            remote_path: Path to the file on the FTP server.
            local_path: Optional local path for the downloaded file.
                       If None, a temporary file is created.

        Returns:
            Local path to the downloaded file, or None on failure.
        """
        self.ensure_connection()
        if local_path is None:
            fd, local_path = tempfile.mkstemp()
            os.close(fd)

        try:
            with open(local_path, 'wb') as f:
                self.ftp.retrbinary(f'RETR {remote_path}', f.write)
            self.last_activity = datetime.now()
            return local_path
        except Exception as e:
            print(f'Error downloading {remote_path}: {e}')
            if os.path.exists(local_path):
                os.remove(local_path)
            return None

    def list_directory(self, remote_path: str) -> list[str]:
        """List contents of a remote directory.

        Args:
            remote_path: Path to the directory on the FTP server.

        Returns:
            List of directory contents (in FTP DIR format).
        """
        self.ensure_connection()
        try:
            contents: list[str] = []
            self.ftp.dir(remote_path, contents.append)
            self.last_activity = datetime.now()
            return contents
        except Exception:
            return []

    def close(self) -> None:
        """Close FTP connection gracefully."""
        if self.ftp:
            try:
                self.ftp.quit()
            except Exception:
                pass
            self.ftp = None


# Global FTP handler instance
_ftp_handler: Optional[FTPDataHandler] = None


def get_ftp_handler() -> FTPDataHandler:
    """Get or create the global FTP handler instance.

    Creates a new FTPDataHandler if one doesn't exist, or ensures the
    existing connection is still alive.

    Returns:
        The global FTPDataHandler instance.
    """
    global _ftp_handler
    if _ftp_handler is None:
        _ftp_handler = FTPDataHandler()
    else:
        _ftp_handler.ensure_connection()
    return _ftp_handler


def reset_ftp_connection() -> FTPDataHandler:
    """Force reset the FTP connection.

    Call this if connection issues persist after normal reconnection attempts.

    Returns:
        A fresh FTPDataHandler instance.
    """
    global _ftp_handler
    if _ftp_handler:
        _ftp_handler.close()
        _ftp_handler = None
    return get_ftp_handler()
