"""Main GUI application for SANSA Geomagnetic Data Processing.

This module provides a PyQt5-based graphical user interface for:
- Single-station and multi-station magnetic field data plotting
- SQD/CTU data conversion to IAGA-2002 format
- Data quality visualization and comparison

The application connects to SANSA FTP servers for data retrieval and
supports various magnetometer instruments including DTU, FGM, L251, SQD, and CTU.
"""

import os
import subprocess
import sys
import traceback

from PyQt5.QtCore import (
    QDate,
    QEasingCurve,
    QProcess,
    QPropertyAnimation,
    QRect,
    Qt,
    QThread,
    QTime,
    QTimer,
    pyqtProperty,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QGuiApplication,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDesktopWidget,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from connectFTP import get_ftp_handler, FTP_CONFIG

# Import plotting functions with fallback
try:
    # Use ONLY the convert scripts for all plotting (they handle all instruments)
    from plot_convert_SingleOnGUI_FTP import (loadBaselineValues, _mask_sentinels_and_extremes, 
                          clean_component_at_original_resolution, minute_aggregate_clean_data, 
                          load_fgm, load_ovh_for_station, calculate_Fc_corrected, 
                          plotSingleStation, process_single_station)

    from plot_convert_MultipleOnGUI_FTP import (_mask_sentinels_and_extremes, despike_series, 
                            minute_aggregate_exclude_nans, window_majority_despike, 
                            aggregate_seconds_to_minute, remove_derivative_spikes, 
                            load_ovh_for_range, clean_component, loadBaselineValues, 
                            apply_baseline_and_compute_F, load_fgm, plotData, processFiles)
except ImportError:
    print("Warning: Could not import plotting functions from convert scripts")
    
    def plotSingleStation(*args, **kwargs):
        print("plotSingleStation called but not implemented")
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.plot([0, 1, 2], [0, 1, 0])
        ax.set_title("Placeholder Plot - Integration Required")
        return fig
    
    def processFiles(*args, **kwargs):
        print("processFiles called but not implemented")
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.plot([0, 1, 2], [0, 1, 0])
        ax.set_title("Placeholder Plot - Integration Required")
        return fig

# Determine script directory
try:
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

# Instrument folder mapping for FTP paths
INSTRUMENT_FOLDER: dict[str, str] = {
    'DTU': 'DTU',
    'L025': 'L25',
    'FGM1': 'FGM1',
    'FGM2': 'FGM2'
}


class PlotWorker(QThread):
    """Worker thread for non-blocking plot generation with FTP support.

    This class handles data retrieval and plot generation in a background
    thread to keep the GUI responsive during long-running operations.

    Signals:
        finished_signal: Emitted with the figure when plot generation succeeds.
        error_signal: Emitted with error message on failure.
        progress_signal: Emitted with progress updates during processing.
    """

    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)

    def __init__(
        self,
        plot_type: str,
        base_path: str,
        station1: str,
        inst1: str,
        station2: str = None,
        inst2: str = None,
        start_date: str = None,
        end_date: str = None,
        start_time: str = None,
        end_time: str = None
    ) -> None:
        """Initialize the plot worker.

        Args:
            plot_type: Either 'single' or 'multi' for station mode.
            base_path: Base path for data (or 'FTP' for remote).
            station1: First station code.
            inst1: First instrument type.
            station2: Second station code (multi mode only).
            inst2: Second instrument type (multi mode only).
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            start_time: Start time in HH:MM:SS format.
            end_time: End time in HH:MM:SS format.
        """
        super().__init__()
        self.plot_type = plot_type
        self.base_path = base_path
        self.station1 = station1
        self.inst1 = inst1
        self.station2 = station2
        self.inst2 = inst2
        self.start_date = start_date
        self.end_date = end_date
        self.start_time = start_time
        self.end_time = end_time
        self._is_running = True

    def run(self):
        try:
            # Initialize FTP connection
            self.progress_signal.emit("Connecting to FTP storage...")
            ftp_handler = get_ftp_handler()
            self.progress_signal.emit("FTP connection established")
            
            self.progress_signal.emit("Loading data...")
            fig = None
            
            # ALWAYS use convert scripts (they handle all instruments, not just SQD/CTU)
            if self.plot_type == 'single':
                from plot_convert_SingleOnGUI_FTP import process_single_station
                self.progress_signal.emit(f"Processing {self.station1}/{self.inst1}...")
                print(f'Using plot_convert_SingleOnGUI_FTP for single station {self.station1}/{self.inst1}')
                
                # Parse output format from instrument name if present (e.g., 'SQD-XYZF')
                inst_base = self.inst1
                output_format = None
                if self.inst1 and '-' in self.inst1:
                    inst_base, maybe_fmt = self.inst1.split('-', 1)
                    output_format = maybe_fmt.upper()
                
                fig = process_single_station(
                    base_path=self.base_path,
                    station=self.station1,
                    instrument=inst_base,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    start_time=self.start_time,
                    end_time=self.end_time,
                    mode='plot',
                    output_format=output_format,
                    compare_dtu=True,
                    gui=True
                )
            else:  # multi-station
                from plot_convert_MultipleOnGUI_FTP import processFiles
                self.progress_signal.emit(f"Processing {self.station1}/{self.inst1} vs {self.station2}/{self.inst2}...")
                print(f'Using plot_convert_MultipleOnGUI_FTP for multiple stations {self.station1}/{self.inst1} vs {self.station2}/{self.inst2}')
                
                fig = processFiles(
                    base_path=self.base_path,
                    first_station=self.station1,
                    first_instrument=self.inst1,
                    second_station=self.station2,
                    second_instrument=self.inst2,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    start_time=self.start_time,
                    end_time=self.end_time,
                    mode='plot'
                )
            
            if self._is_running:
                if fig is not None:
                    self.progress_signal.emit("Plot generation complete")
                    self.finished_signal.emit(fig)
                else:
                    self.error_signal.emit("No plot was generated - check data availability")
                    
        except ImportError as e:
            self.error_signal.emit(f"Import error: {str(e)}")
        except Exception as e:
            self.error_signal.emit(f"Plotting error: {str(e)}")
        finally:
            self._is_running = False

    def stop(self) -> None:
        """Stop the thread gracefully."""
        self._is_running = False
        self.terminate()
        self.wait(2000)


class ToggleButton(QPushButton):
    """Animated toggle button with sliding thumb indicator.

    A custom toggle button that displays an animated sliding thumb
    when toggled between states.
    """

    def __init__(self, parent=None) -> None:
        """Initialize the toggle button."""
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(80, 40)

        # Colors
        self.bg_off = QColor(220, 220, 220)
        self.bg_on = QColor(220, 220, 220)
        self.thumb_color = QColor('#09597d')
        self.pen_color = QColor('#09597d')

        # Thumb animation state
        self._thumb_pos = 3
        self.anim = QPropertyAnimation(self, b'thumbPos', self)
        self.anim.setDuration(200)

        self.setFont(QFont('Arial', 10, QFont.Bold))
        self.toggled.connect(self._onToggled)
        self._onToggled(self.isChecked())

    def paintEvent(self, event) -> None:
        """Paint the toggle button."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.pen_color)
        pen.setWidth(2)
        p.setPen(pen)

        # Background
        bg = self.bg_on if self.isChecked() else self.bg_off
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, 19, 19)

        # Thumb
        p.setBrush(QBrush(self.thumb_color))
        p.drawEllipse(self._thumb_pos, 3, 34, 34)

    def _onToggled(self, checked: bool) -> None:
        """Animate thumb position on toggle."""
        end = self.width() - 37 if checked else 3
        self.anim.stop()
        self.anim.setStartValue(self._thumb_pos)
        self.anim.setEndValue(end)
        self.anim.start()

    def get_thumbPos(self) -> int:
        """Get current thumb position."""
        return self._thumb_pos

    def set_thumbPos(self, v: int) -> None:
        """Set thumb position and update display."""
        self._thumb_pos = v
        self.update()

    thumbPos = pyqtProperty(int, get_thumbPos, set_thumbPos)


class SQDProcessingWidget(QWidget):
    """Widget for SQD/CTU data conversion to IAGA-2002 format.

    Provides UI controls for converting raw SQUID and CTU magnetometer
    data files to the standard IAGA-2002 format.
    """

    def __init__(self, parent=None) -> None:
        """Initialize the SQD processing widget."""
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        """Build the user interface."""
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setVerticalSpacing(5)
        grid.setHorizontalSpacing(10)

        # Station (fixed to HER for SQD)
        lbl_station = QLabel('Station')
        grid.addWidget(lbl_station, 0, 0)
        self.stationLabel = QComboBox()
        self.stationLabel.addItems(['HER'])
        self.stationLabel.setEnabled(False)
        grid.addWidget(self.stationLabel, 0, 1, 1, 2)

        # Instrument selection (SQD/CTU)
        lbl_instrument = QLabel('Instrument')
        grid.addWidget(lbl_instrument, 1, 0)
        self.instrumentCombo = QComboBox()
        self.instrumentCombo.addItems(['SQD', 'CTU'])
        grid.addWidget(self.instrumentCombo, 1, 1, 1, 2)

        # Format selection
        lbl_format = QLabel('Output Format')
        grid.addWidget(lbl_format, 2, 0)
        self.formatCombo = QComboBox()
        self.formatCombo.addItems(['HDZF', 'DHZF', 'XYZF'])
        grid.addWidget(self.formatCombo, 2, 1, 1, 2)

        lbl_start = QLabel('Start Date')
        grid.addWidget(lbl_start, 4, 0)
        self.startDate = QDateEdit(calendarPopup=True)
        self.startDate.setDate(QDate.currentDate().addDays(-1))
        grid.addWidget(self.startDate, 4, 1, 1, 2)

        lbl_end = QLabel('End Date')
        grid.addWidget(lbl_end, 5, 0)
        self.endDate = QDateEdit(calendarPopup=True)
        self.endDate.setDate(QDate.currentDate())
        grid.addWidget(self.endDate, 5, 1, 1, 2)

        # Status label
        self.status_label = QLabel('Ready')
        self.status_label.setStyleSheet('color: green;')
        grid.addWidget(self.status_label, 6, 0, 1, 3)

        # Buttons layout
        btn_layout = QHBoxLayout()
        self.convertBtn = QPushButton('Convert SQD/CTU')
        self.convertBtn.clicked.connect(self._run_conversion)
        btn_layout.addWidget(self.convertBtn)

        grid.addLayout(btn_layout, 7, 0, 1, 3)

        bigFont = QFont('Aptos', 12)
        for w in [
            lbl_station, lbl_instrument, lbl_format, lbl_start, lbl_end,
            self.stationLabel, self.instrumentCombo, self.formatCombo,
            self.startDate, self.endDate, self.convertBtn
        ]:
            w.setFont(bigFont)

        layout.addLayout(grid)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self.instrumentCombo.currentTextChanged.connect(self._on_instrument_changed)
        self._on_instrument_changed('SQD')

    def _on_instrument_changed(self, instrument: str) -> None:
        """Update UI based on selected instrument.

        Args:
            instrument: Selected instrument type ('SQD' or 'CTU').
        """
        if instrument == 'CTU':
            # Disable XYZF for CTU
            self.formatCombo.setCurrentText('HDZF')
            index = self.formatCombo.findText('XYZF')
            if index >= 0:
                self.formatCombo.model().item(index).setEnabled(False)
        else:
            # Enable all formats for SQD
            for i in range(self.formatCombo.count()):
                self.formatCombo.model().item(i).setEnabled(True)

    def _run_conversion(self) -> None:
        """Run SQD/CTU data conversion."""
        instrument = self.instrumentCombo.currentText()
        output_format = self.formatCombo.currentText()
        compare_dtu = True
        start_date = self.startDate.date().toString('yyyy-MM-dd')
        end_date = self.endDate.date().toString('yyyy-MM-dd')
        out_base = os.path.expanduser(r'~/Documents/SANSA_Processed_Data')

        print(f'DEBUG SQDProcessingWidget: Instrument={instrument}, Format={output_format}')
        print(f'DEBUG SQDProcessingWidget: Date Range={start_date} to {end_date}')

        self.status_label.setText('Converting...')
        self.status_label.setStyleSheet('color: blue;')
        QApplication.processEvents()

        try:
            from plot_convert_SingleOnGUI_FTP import process_single_station

            results = process_single_station(
                start_date=start_date,
                end_date=end_date,
                out_base=out_base,
                mode='convert',
                instruments=[instrument],
                output_format=output_format,
                compare_dtu=True,
                gui=True
            )

            self.status_label.setText('Conversion complete')
            self.status_label.setStyleSheet('color: green;')

            format_dir = output_format.replace('F', '')
            target_dir = os.path.join(out_base, format_dir)
            os.makedirs(target_dir, exist_ok=True)

            QMessageBox.information(
                self, 'Success',
                f'Conversion completed successfully for {instrument}\n\n'
                f'{instrument} data saved at: {target_dir}'
            )

        except subprocess.CalledProcessError as e:
            error_msg = (
                f'Conversion failed with exit code {e.returncode}\n\n'
                f'STDERR:\n{e.stderr}\n\nSTDOUT:\n{e.stdout}'
            )
            self.status_label.setText('Conversion failed')
            self.status_label.setStyleSheet('color: red;')
            QMessageBox.critical(self, 'Error', error_msg)
        except subprocess.TimeoutExpired:
            self.status_label.setText('Conversion timed out')
            self.status_label.setStyleSheet('color: red;')
            QMessageBox.critical(self, 'Error', 'Conversion timed out after 5 minutes')
        except Exception as e:
            error_msg = f'Unexpected error: {str(e)}'
            self.status_label.setText('Conversion failed')
            self.status_label.setStyleSheet('color: red;')
            QMessageBox.critical(self, 'Error', error_msg)
            print(f'DEBUG: Unexpected error: {traceback.format_exc()}')


class DataQualityCheckWidget(QWidget):
    """Widget for data quality checking and visualization.

    Provides controls for selecting stations, instruments, and date ranges
    to generate comparison plots for geomagnetic data quality assessment.
    """

    def __init__(self) -> None:
        """Initialize the data quality check widget."""
        super().__init__()
        self._setup_ui()
        self._connect_signals()
        self._toggle_single_station_mode(self.stationToggle.isChecked())
        self._toggle_single_day_mode(self.dayToggle.isChecked())
        self.plotProc = QProcess(self)
        self.plotProc.setProcessChannelMode(QProcess.MergedChannels)
        self.plotProc.readyReadStandardOutput.connect(self._forwardPlotOutput)
        self.plotProc.finished.connect(self._onPlotFinished)
        self.plotProc.errorOccurred.connect(self._onPlotError)

        # Test FTP connection on startup
        QTimer.singleShot(1000, self.test_ftp_connection)

    def test_ftp_connection(self) -> None:
        """Test FTP connection and update status label."""
        try:
            from connectFTP import get_ftp_handler
            ftp_handler = get_ftp_handler()
            if ftp_handler.ftp is not None:
                self.status_label.setText('Connected, Ready to Plot')
                self.status_label.setStyleSheet('color: green;')
            else:
                self.status_label.setText('Storage connection failed')
                self.status_label.setStyleSheet('color: red;')
        except Exception as e:
            self.status_label.setText(f'Storage error: {str(e)}')
            self.status_label.setStyleSheet('color: red;')

    def reset_ftp_connection(self) -> None:
        """Reset FTP connection to fix stale connections."""
        try:
            from connectFTP import reset_ftp_connection
            reset_ftp_connection()
            self.status_label.setText('FTP connection reset')
            self.status_label.setStyleSheet('color: green;')
        except Exception as e:
            self.status_label.setText(f'FTP reset failed: {str(e)}')
            self.status_label.setStyleSheet('color: red;')

    def _forwardPlotOutput(self) -> None:
        """Forward plot process output to console."""
        ba = self.plotProc.readAllStandardOutput()
        text = bytes(ba).decode('utf-8', errors='ignore')
        print(text, end='')

    def closeEvent(self, event) -> None:
        """Ensure worker thread is stopped when widget closes."""
        if hasattr(self, 'plot_worker') and self.plot_worker.isRunning():
            self.plot_worker.stop()
        event.accept()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle('Data Quality Check')

        # Create all widgets
        self.stationToggle = ToggleButton()
        self.stationLabelOff = QLabel('Single-Station Single-Instrument Mode')
        self.stationLabelOn = QLabel('Multi-Station Multi-Instrument Mode')

        self.dayToggle = ToggleButton()
        self.dayLabelOff = QLabel('Single-Day Mode')
        self.dayLabelOn = QLabel('Multi-Day Mode')

        self.compareCombo1 = QComboBox()
        self.compareCombo1.addItems(['HER', 'HBK', 'TSU', 'KMH', 'SNA', 'MRN'])
        self.labelCombo1 = QComboBox()
        self.compareCombo2 = QComboBox()
        self.compareCombo2.addItems(['HER', 'HBK', 'TSU', 'KMH', 'SNA', 'MRN'])
        self.labelCombo2 = QComboBox()

        self.startDate = QDateEdit(calendarPopup=True)
        self.startDate.setDate(QDate.currentDate())
        self.startHour = QComboBox()
        self.startHour.addItems([f'{h:02d}:00' for h in range(25)])
        self.endDate = QDateEdit(calendarPopup=True)
        self.endDate.setDate(QDate.currentDate())
        self.endHour = QComboBox()
        self.endHour.addItems([f'{h:02d}:00' for h in range(25)])
        self.endHour.setCurrentText('24:00')

        self.plotBtn = QPushButton('Plot Comparison/Single Station')
        self.save_button = QPushButton('Save Plot')

        bigFont = QFont('Aptos', 12)
        for w in (
            self.stationLabelOff, self.stationLabelOn,
            self.dayLabelOff, self.dayLabelOn,
            self.compareCombo1, self.labelCombo1,
            self.compareCombo2, self.labelCombo2,
            self.startDate, self.startHour,
            self.endDate, self.endHour,
            self.plotBtn, self.save_button,
        ):
            w.setFont(bigFont)

        self.plotBtn.clicked.connect(self.plotComparison)
        self.save_button.clicked.connect(self.save_plot)

        # Layout
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(12)

        # Toggle Row #1
        row1 = QHBoxLayout()
        row1.addWidget(self.stationLabelOff)
        row1.addStretch()
        row1.addWidget(self.stationToggle)
        row1.addStretch()
        row1.addWidget(self.stationLabelOn)
        vbox.addLayout(row1)

        # Station / Instrument Group
        stationGroup = QGroupBox('Choose Station(s)')
        stationGroup.setFont(bigFont)
        sg = QGridLayout(stationGroup)
        sg.setContentsMargins(5, 5, 5, 5)
        sg.setHorizontalSpacing(10)
        sg.setVerticalSpacing(4)
        sg.addWidget(QLabel('Station 1:'), 0, 0, Qt.AlignRight)
        sg.addWidget(self.compareCombo1, 0, 1)
        sg.addWidget(QLabel('Instrument 1:'), 0, 2, Qt.AlignRight)
        sg.addWidget(self.labelCombo1, 0, 3)
        sg.addWidget(QLabel('Station 2:'), 1, 0, Qt.AlignRight)
        sg.addWidget(self.compareCombo2, 1, 1)
        sg.addWidget(QLabel('Instrument 2:'), 1, 2, Qt.AlignRight)
        sg.addWidget(self.labelCombo2, 1, 3)
        vbox.addWidget(stationGroup)

        # Toggle Row #2
        row2 = QHBoxLayout()
        row2.addWidget(self.dayLabelOff)
        row2.addStretch()
        row2.addWidget(self.dayToggle)
        row2.addStretch()
        row2.addWidget(self.dayLabelOn)
        vbox.addLayout(row2)

        # Dates Group
        dateGroup = QGroupBox('Choose Date(s)')
        dateGroup.setFont(bigFont)
        dg = QGridLayout(dateGroup)
        dg.setContentsMargins(5, 5, 5, 5)
        dg.setHorizontalSpacing(10)
        dg.setVerticalSpacing(4)
        dg.addWidget(QLabel('Start Date:'), 0, 0, Qt.AlignRight)
        dg.addWidget(self.startDate, 0, 1)
        dg.addWidget(QLabel('Start Hour:'), 0, 2, Qt.AlignRight)
        dg.addWidget(self.startHour, 0, 3)
        dg.addWidget(QLabel('End Date:'), 1, 0, Qt.AlignRight)
        dg.addWidget(self.endDate, 1, 1)
        dg.addWidget(QLabel('End Hour:'), 1, 2, Qt.AlignRight)
        dg.addWidget(self.endHour, 1, 3)
        vbox.addWidget(dateGroup)

        # Buttons Row
        btnRow = QHBoxLayout()
        btnRow.addWidget(self.plotBtn)
        btnRow.addStretch()
        btnRow.addWidget(self.save_button)
        vbox.addLayout(btnRow)

        # Status Label
        self.status_label = QLabel('Connecting to storage...')
        self.status_label.setFont(QFont('Aptos', 10))
        self.status_label.setStyleSheet('color: orange;')
        vbox.addWidget(self.status_label)

        # Initial state
        self.stationToggle.setChecked(False)
        self.dayToggle.setChecked(True)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        # Station-instrument mappings
        self.single_station_mapping = {
            'SNA': ['DTU1', 'FGM1'],
            'HER': ['DTU1', 'FGM1', 'FGM2', 'L251', 'SQD-HDZF', 'SQD-XYZF', 'CTU'],
            'MRN': ['LE011'],
            'HBK': ['DTU1', 'FGM1', 'L251'],
            'TSU': ['DTU1', 'FGM1'],
            'KMH': ['DTU1', 'FGM1']
        }

        self.multi_station_mapping = {
            'SNA': ['DTU1', 'FGM1'],
            'HER': ['DTU1', 'FGM1', 'FGM2', 'L251', 'CTU'],
            'MRN': ['LE011'],
            'HBK': ['DTU1', 'FGM1', 'L251'],
            'TSU': ['DTU1', 'FGM1'],
            'KMH': ['DTU1', 'FGM1']
        }

        self.compareCombo1.currentTextChanged.connect(
            lambda st: self._update_instruments(st, self.labelCombo1)
        )
        self.compareCombo2.currentTextChanged.connect(
            lambda st: self._update_instruments(st, self.labelCombo2)
        )
        self.stationToggle.toggled.connect(self._toggle_single_station_mode)
        self.dayToggle.toggled.connect(self._toggle_single_day_mode)

        # Update both dropdowns initially
        self._update_instruments(self.compareCombo1.currentText(), self.labelCombo1)
        self._update_instruments(self.compareCombo2.currentText(), self.labelCombo2)

        self.dayToggle.toggled.connect(
            lambda ch: self._sync_end_date_if_single(self.startDate.date())
        )
        self.startDate.dateChanged.connect(self._sync_end_date_if_single)

    def _toggle_single_station_mode(self, checked: bool) -> None:
        """Toggle between single and multi station mode.

        Args:
            checked: True for multi-station mode, False for single-station.
        """
        self.compareCombo2.setEnabled(checked)
        self.labelCombo2.setEnabled(checked)

        station1 = self.compareCombo1.currentText()
        station2 = self.compareCombo2.currentText()

        self._update_instruments(station1, self.labelCombo1)
        self._update_instruments(station2, self.labelCombo2)

        is_multi_station = checked

        for combo in [self.labelCombo1, self.labelCombo2]:
            current_text = combo.currentText()
            for i in range(combo.count()):
                item_text = combo.itemText(i)
                if item_text.upper().startswith('SQD'):
                    combo.model().item(i).setEnabled(not is_multi_station)

            if is_multi_station and current_text.upper().startswith('SQD'):
                for i in range(combo.count()):
                    if not combo.itemText(i).upper().startswith('SQD'):
                        combo.setCurrentIndex(i)
                        break

    def _toggle_single_day_mode(self, checked: bool) -> None:
        """Toggle between single and multi day mode.

        Args:
            checked: True for multi-day mode, False for single-day.
        """
        self.startHour.setEnabled(not checked)
        self.endHour.setEnabled(not checked)
        self.endDate.setEnabled(checked)
        if not checked:
            self.endDate.setDate(self.startDate.date())

    def _sync_end_date_if_single(self, new_start_date) -> None:
        """Sync end date to start date in single-day mode."""
        if not self.dayToggle.isChecked():
            self.endDate.setDate(new_start_date)

    def _update_instruments(self, station: str, combo: QComboBox) -> None:
        """Update instrument dropdown based on station selection.

        Args:
            station: Selected station code.
            combo: Instrument combo box to update.
        """
        if self.stationToggle.isChecked():
            mapping = self.multi_station_mapping
        else:
            mapping = self.single_station_mapping

        current_selection = combo.currentText()
        combo.clear()
        combo.addItems(mapping.get(station, []))

        if current_selection in mapping.get(station, []):
            combo.setCurrentText(current_selection)

    def show_error(self, message: str) -> None:
        """Show error message dialog.

        Args:
            message: Error message to display.
        """
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle('Error')
        msg.exec_()

    def plotComparison(self) -> None:
        """Generate and display comparison plot."""
        self.reset_ftp_connection()

        station1 = self.compareCombo1.currentText()
        inst1 = self.labelCombo1.currentText()

        if not self.stationToggle.isChecked():
            station2, inst2 = station1, inst1
            plot_type = 'single'
        else:
            station2 = self.compareCombo2.currentText()
            inst2 = self.labelCombo2.currentText()
            plot_type = 'multi'

        sd_iso = self.startDate.date().toString('yyyy-MM-dd')
        ed_iso = self.endDate.date().toString('yyyy-MM-dd')

        if not self.dayToggle.isChecked():
            raw_start = self.startHour.currentText()
            raw_end = self.endHour.currentText()
            start_time = raw_start + ':00'
            end_time = '23:59:59' if raw_end == '24:00' else raw_end + ':00'
        else:
            start_time, end_time = '00:00:00', '23:59:59'

        base_path = 'FTP'

        folder1 = INSTRUMENT_FOLDER.get(inst1, inst1)
        folder2 = INSTRUMENT_FOLDER.get(inst2, inst2)

        self.plotBtn.setEnabled(False)
        self.plotBtn.setText('Plotting...')

        if hasattr(self, 'plot_worker') and self.plot_worker.isRunning():
            self.plot_worker.stop()

        self.plot_worker = PlotWorker(
            plot_type=plot_type,
            base_path=base_path,
            station1=station1,
            inst1=folder1,
            station2=station2 if plot_type == 'multi' else None,
            inst2=folder2 if plot_type == 'multi' else None,
            start_date=sd_iso,
            end_date=ed_iso,
            start_time=start_time,
            end_time=end_time
        )

        self.plot_worker.finished_signal.connect(self.on_plot_finished)
        self.plot_worker.error_signal.connect(self.on_plot_error)
        self.plot_worker.progress_signal.connect(self.on_plot_progress)

        self.plot_worker.start()

        QTimer.singleShot(600000, self.on_plot_timeout)

    def on_convert_only_clicked(self) -> None:
        """Handle convert-only button click for SQD/CTU data."""
        single_mode = not self.stationToggle.isChecked()
        inst1 = self.labelCombo1.currentText().upper()
        inst2 = None
        if not single_mode:
            inst2 = self.labelCombo2.currentText().upper()

        instruments_to_handle = []
        if inst1 in ('SQD', 'CTU'):
            instruments_to_handle.append(inst1)
        if inst2 and inst2 in ('SQD', 'CTU'):
            instruments_to_handle.append(inst2)

        if not instruments_to_handle:
            QMessageBox.information(
                self, 'Nothing to convert',
                'Convert Only is active only for SQD or CTU instruments.'
            )
            return

        sd_iso = self.startDate.date().toString('yyyy-MM-dd')
        ed_iso = self.endDate.date().toString('yyyy-MM-dd')
        outbase = os.path.expanduser(r'~/Documents/SANSA_Processed_Data')

        converter_path = os.path.join(SCRIPT_DIR, 'plot_convert_SingleOnGUI_FTP.py')
        if not os.path.exists(converter_path):
            QMessageBox.critical(
                self, 'Converter missing',
                f'Converter script not found at:\n{converter_path}'
            )
            return

        if single_mode:
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, 'plot_convert_SingleOnGUI_FTP.py'),
                sd_iso, ed_iso,
                '--mode', 'convert',
                '--outbase', outbase,
                '--instruments'
            ] + instruments_to_handle
        else:
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, 'plot_convert_MultipleOnGUI_FTP.py'),
                outbase,
                self.compareCombo1.currentText(),
                self.labelCombo1.currentText(),
                self.compareCombo2.currentText(),
                self.labelCombo2.currentText(),
                sd_iso, ed_iso,
                '00:00:00', '23:59:59',
                '--mode', 'convert'
            ]

        print(f'DEBUG: Executing command: {" ".join(cmd)}')

        self.status_label.setText('Starting conversion from FTP...')
        QApplication.processEvents()

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f'DEBUG: Command output: {result.stdout}')
            if result.stderr:
                print(f'DEBUG: Command errors: {result.stderr}')
            subprocess.run(cmd, check=True)
            self.status_label.setText('Conversion complete')
            self.status_label.setStyleSheet('color: green;')
        except subprocess.CalledProcessError:
            self.status_label.setText('Conversion failed')
            self.status_label.setStyleSheet('color: red;')
        except Exception:
            self.status_label.setText('Conversion error')
            self.status_label.setStyleSheet('color: red;')

    def on_plot_finished(self, fig) -> None:
        """Handle successful plot generation.

        Args:
            fig: Generated matplotlib figure.
        """
        try:
            self.plotBtn.setEnabled(True)
            self.plotBtn.setText('Plot Comparison/Single Station')

            if fig is not None:
                self.display_figure(fig)
                self.status_label.setText('Ready to Save/Plot Again')
                self.status_label.setStyleSheet('color: green;')
            else:
                self.status_label.setText('Ready - No plot generated')
                self.status_label.setStyleSheet('color: orange;')
                QMessageBox.warning(self, 'Warning', 'No plot was generated')

        except Exception as e:
            self.on_plot_error(f'Error displaying plot: {str(e)}')

    def on_plot_error(self, error_message: str) -> None:
        """Handle plot generation error.

        Args:
            error_message: Error message to display.
        """
        self.plotBtn.setEnabled(True)
        self.plotBtn.setText('Plot Comparison/Single Station')

        self.status_label.setText('Ready - Error occurred')
        self.status_label.setStyleSheet('color: red;')

        QMessageBox.critical(self, 'Plotting Error', error_message)
        print(f'Plotting error: {error_message}')

    def on_plot_progress(self, progress_message: str) -> None:
        """Handle progress updates from worker thread.

        Args:
            progress_message: Progress message to display.
        """
        print(f'Plot progress: {progress_message}')
        self.status_label.setText(progress_message)

        if 'error' in progress_message.lower() or 'failed' in progress_message.lower():
            self.status_label.setStyleSheet('color: red;')
        elif 'complete' in progress_message.lower() or 'ready' in progress_message.lower():
            self.status_label.setStyleSheet('color: green;')
        elif 'plotting' in progress_message.lower() or 'loading' in progress_message.lower():
            self.status_label.setStyleSheet('color: blue;')
        else:
            self.status_label.setStyleSheet('color: black;')

    def on_plot_timeout(self) -> None:
        """Handle plot thread timeout."""
        if hasattr(self, 'plot_worker') and self.plot_worker.isRunning():
            print('Plot thread timeout - stopping...')
            self.plot_worker.stop()
            self.on_plot_error('Plotting timed out after 10 minutes')

    def display_figure(self, fig) -> None:
        """Display matplotlib figure in main window.

        Args:
            fig: Matplotlib figure to display.
        """
        main_window = self.window()
        if not hasattr(main_window, 'display_figure_in_main'):
            QMessageBox.warning(self, 'Error', 'Cannot display plot - main window not found')
            return

        main_window.display_figure_in_main(fig)

    def clear_plot_area(self) -> None:
        """Clear the plot area in the main window."""
        main_window = self.window()
        if hasattr(main_window, 'clear_plot_area'):
            main_window.clear_plot_area()

    @pyqtSlot(int, QProcess.ExitStatus)
    def _onPlotFinished(self, exitCode: int, exitStatus: QProcess.ExitStatus) -> None:
        """Handle plot process completion."""
        out = self._stdout.decode('utf-8', 'replace')
        err = self._stderr.decode('utf-8', 'replace')

        if exitStatus == QProcess.NormalExit and exitCode == 0:
            QMessageBox.information(self, 'Success', 'Plot created successfully.')
            if hasattr(self, 'plot_display_requested'):
                self.plot_display_requested.emit()
        else:
            QMessageBox.critical(
                self, 'Plot Failed',
                f'Exit code: {exitCode}\n\n--- STDOUT ---\n{out}\n\n--- STDERR ---\n{err}'
            )
        self.plotBtn.setEnabled(True)

    @pyqtSlot(QProcess.ProcessError)
    def _onPlotError(self, error: QProcess.ProcessError) -> None:
        """Handle QProcess error."""
        err_str = self.plotProc.errorString()
        print(f'[DEBUG] QProcess error enum={error}, string={err_str}')
        QMessageBox.critical(
            self, 'Process Error',
            f'Could not launch plot process:\n{err_str}'
        )

    @pyqtSlot()
    def _onStdOut(self) -> None:
        """Collect stdout from process."""
        self._stdout += bytes(self.plotProc.readAllStandardOutput())

    @pyqtSlot()
    def _onStdErr(self) -> None:
        """Collect stderr from process."""
        self._stderr += bytes(self.plotProc.readAllStandardError())

    def save_plot(self) -> None:
        """Save the current plot to a file."""
        main_window = self.window()
        if not hasattr(main_window, 'current_figure') or main_window.current_figure is None:
            QMessageBox.warning(self, 'Warning', 'No plot to save')
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save Plot', '',
            'PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)'
        )

        if filename:
            try:
                main_window.current_figure.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, 'Success', f'Plot saved to {filename}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Could not save plot: {str(e)}')


class MainWindow(QWidget):
    """Main application window for SANSA Geomagnetic Data Processing.

    This window provides the primary interface with:
    - Left panel: Data quality check and conversion controls
    - Right panel: Plot display area
    """

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self.current_figure = None
        self.current_canvas = None
        self.current_plot_container = None
        self.initUI()

    def initUI(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle('Geomagnetic Data Processing')
        self.center()

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left column (50%)
        left_column = QVBoxLayout()
        left_column.setSpacing(10)

        # Right column (50%) - for plot display
        self.right_column = QVBoxLayout()
        self.right_column.setSpacing(10)

        # Create sections for left column
        self.create_plot_section()
        self.create_bottom_section()

        left_column.addWidget(self.plot_section, 6)
        left_column.addWidget(self.bottom_section, 4)

        self.create_plot_area()

        main_layout.addLayout(left_column, 1)
        main_layout.addLayout(self.right_column, 1)

        self.setLayout(main_layout)
        self.showMaximized()

    def center(self) -> None:
        """Center the window on the screen."""
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def create_plot_section(self) -> None:
        """Create the plot comparison/single station section."""
        self.plot_section = QGroupBox('Plot Data (Comparison / Single Station)')
        self.plot_section.setStyleSheet('QGroupBox { font-weight: bold; color: #09597d; }')

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        self.data_quality_widget = DataQualityCheckWidget()
        scroll.setWidget(self.data_quality_widget)

        layout = QVBoxLayout(self.plot_section)
        layout.addWidget(scroll)

    def create_bottom_section(self) -> None:
        """Create the bottom section with conversion and info panels."""
        self.bottom_section = QWidget()
        bottom_layout = QHBoxLayout(self.bottom_section)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)

        # Left column: Data Conversion + Logo
        left_column = QVBoxLayout()
        left_column.setSpacing(10)

        self.create_conversion_section()
        self.create_logo_section()

        left_column.addWidget(self.conversion_section, 3)
        left_column.addWidget(self.logo_section, 1)

        # Right column: Notes
        self.create_notes_section()

        bottom_layout.addLayout(left_column, 1)
        bottom_layout.addWidget(self.notes_section, 1)

    def create_notes_section(self) -> None:
        """Create the notes/instructions section."""
        self.notes_section = QGroupBox('Notes / About')
        self.notes_section.setStyleSheet('QGroupBox { font-weight: bold; color: #09597d; }')

        layout = QVBoxLayout(self.notes_section)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        notes_content = QWidget()
        notes_layout = QVBoxLayout(notes_content)

        # Plot Comparison Instructions
        plot_instructions = QLabel(
            '<h3>Plot Comparison / Single Station</h3>'
            '<p><b>Single-Station Mode:</b> Plot data from one station and one instrument.</p>'
            '<ul>'
            '<li>Available for all instruments including SQD and CTU</li>'
            '<li>SQD data automatically includes comparison with DTU1 reference data</li>'
            '<li>Other instruments (DTU1, FGM1, FGM2, L251, CTU) show standard variometer '
            'plots with OVH F-measurements</li>'
            '</ul>'
            '<p><b>Multi-Station Mode:</b> Compare different stations or instruments.</p>'
            '<ul>'
            '<li>Available for standard instruments (DTU1, FGM1, FGM2, L251) and CTU</li>'
            '<li><b>SQD instruments are disabled</b> in multi-station mode</li>'
            '<li>Compare different stations with same instruments</li>'
            '<li>Compare different instruments at same station</li>'
            '</ul>'
            '<p><b>Single-Day Mode:</b> View data for one day with custom time range.</p>'
            '<p><b>Multi-Day Mode:</b> View data across multiple full days.</p>'
            '<p><b>How to use:</b></p>'
            '<p>1. Choose station(s) and instrument(s)</p>'
            '<p>2. Select date range and time (if single-day mode)</p>'
            '<p>3. Click "Plot Comparison/Single Station"</p>'
            '<p>4. View plot in right panel</p>'
            '<p>5. Use "Save Plot" to export when satisfied</p>'
            '<br>'
        )
        plot_instructions.setWordWrap(True)
        plot_instructions.setTextFormat(Qt.RichText)

        # Data Conversion Instructions
        conversion_instructions = QLabel(
            '<h3>Data Conversion</h3>'
            '<p>Convert raw SQD/CTU magnetometer files to IAGA-2002 format</p>'
            '<p><b>How to use:</b></p>'
            '<p>1. Select either SQD or CTU instrument</p>'
            '<p>2. Choose output format (HDZF, DHZF, XYZF)</p>'
            '<p>3. Select day(s) to convert the file(s)</p>'
            '<p>4. Click "Convert SQD/CTU"</p>'
            '<p>5. Output directory: Documents/SANSA Processed Data</p>'
            '<p><b>Note:</b> This feature converts proprietary format data to standard '
            'IAGA format for analysis and sharing.</p>'
            '<br>'
        )
        conversion_instructions.setWordWrap(True)
        conversion_instructions.setTextFormat(Qt.RichText)

        # About Section
        about_section = QLabel(
            '<h3>About</h3>'
            '<p><b>Geomagnetic Data Quality Checks Application</b></p>'
            '<p>Version 2.0</p>'
            '<p>This application is designed for processing and quality checking '
            'geomagnetic data from SANSA\'s magnetometer network.</p>'
            '<p><b>Developed By:</b></p>'
            '<ul>'
            '<li>Sakhile Mkhize - Developer (2025)</li>'
            '<li>Mfezeko Rataza - Developer (2024)</li>'
            '<li>Dr. Emmanuel Nahayo - Geomagnetic Data Specialist</li>'
            '</ul>'
            '<p><b>Copyright 2025 SANSA Space Science</b></p>'
            '<p>South African National Space Agency</p>'
            '<p>Space Science Directorate</p>'
            '<p>Engineering & Data Acquisition (EDA)</p>'
            '<p><b>Licensed under the SANSA Internal Use License</b></p>'
            '<p>All rights reserved.</p>'
        )
        about_section.setWordWrap(True)
        about_section.setTextFormat(Qt.RichText)

        notes_layout.addWidget(plot_instructions)
        notes_layout.addWidget(conversion_instructions)
        notes_layout.addWidget(about_section)
        notes_layout.addStretch()

        scroll.setWidget(notes_content)
        layout.addWidget(scroll)

    def create_conversion_section(self) -> None:
        """Create the data conversion section."""
        self.conversion_section = QGroupBox('Convert Data (SQD/CTU to IAGA-2002)')
        self.conversion_section.setStyleSheet('QGroupBox { font-weight: bold; color: #09597d; }')

        self.data_conversion_widget = SQDProcessingWidget()

        layout = QVBoxLayout(self.conversion_section)
        layout.addWidget(self.data_conversion_widget)

    def create_logo_section(self) -> None:
        """Create the logo section."""
        self.logo_section = QGroupBox()
        self.logo_section.setStyleSheet('QGroupBox { border: 0px; }')

        layout = QHBoxLayout(self.logo_section)
        layout.setContentsMargins(5, 5, 5, 5)

        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setMinimumHeight(60)

        self.logo_label.setText('Company Logo')
        self.logo_label.setStyleSheet('''
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 5px;
                color: #666666;
                font-size: 14px;
                padding: 10px;
            }
        ''')

        layout.addWidget(self.logo_label)

    def set_logo_image(self, image_path: str) -> None:
        """Set the company logo image.

        Args:
            image_path: Path to the logo image file.
        """
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(200, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_pixmap)
            self.logo_label.setText('')
        else:
            print(f'Logo image not found: {image_path}')
            self.logo_label.setText('Logo Not Found')

    def create_plot_area(self) -> None:
        """Create the plot area in the right column."""
        self.right_container = QWidget()
        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(0)

        self.plot_placeholder = QLabel(
            'Plot will appear here\n\nGenerate a plot to view it in this area'
        )
        self.plot_placeholder.setAlignment(Qt.AlignCenter)
        self.plot_placeholder.setStyleSheet('''
            color: gray;
            font-size: 16px;
            border: 2px dashed gray;
            padding: 80px;
            background-color: #f8f9fa;
        ''')
        self.plot_placeholder.setMinimumSize(400, 600)

        self.right_layout.addWidget(self.plot_placeholder)
        self.right_column.addWidget(self.right_container)

    def clear_plot_area(self) -> None:
        """Clear the plot area."""
        self.current_figure = None
        self.current_canvas = None

        for i in reversed(range(self.right_column.count())):
            widget = self.right_column.itemAt(i).widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

        self.plot_placeholder = QLabel(
            'Plot will appear here\n\nGenerate a plot to view it in this area'
        )
        self.plot_placeholder.setAlignment(Qt.AlignCenter)
        self.plot_placeholder.setStyleSheet(
            'color: gray; font-size: 16px; border: 2px dashed gray; padding: 20px;'
        )
        self.right_column.addWidget(self.plot_placeholder)

    def display_figure_in_main(self, fig) -> None:
        """Display matplotlib figure in the right column.

        Args:
            fig: Matplotlib figure to display.
        """
        self.clear_plot_area()

        if hasattr(self, 'plot_placeholder'):
            self.plot_placeholder.setParent(None)
            self.plot_placeholder = None

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)

        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, plot_container)

        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(canvas)

        self.right_column.addWidget(plot_container)

        self.current_figure = fig
        self.current_canvas = canvas
        self.current_plot_container = plot_container

        canvas.draw()


def main() -> None:
    """Main application entry point."""
    app = QApplication(sys.argv)
    main_window = MainWindow()

    logo_path = os.path.join(SCRIPT_DIR, 'SANSA (004) (2).png')
    main_window.set_logo_image(logo_path)

    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
