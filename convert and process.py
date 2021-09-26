# coding: utf-8
import configparser
import os
import sys
from pathlib import Path
from typing import List, Tuple, Union, Any, Dict, BinaryIO, cast, Mapping

import numpy as np
import psutil  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from numpy.typing import NDArray

if sys.version_info >= (3, 8):
    from typing import Final
else:
    # stub
    class Final:
        @staticmethod
        def __getitem__(item):
            return item

import utils

INI_SUFFIX: Final[str] = '.ini'
DAT_SUFFIX: Final[str] = '.dat'
PAR_SUFFIX: Final[str] = '.par'
CSV_SUFFIX: Final[str] = '.csv'

SOUND_SPEED: Final[float] = 1500.0


class Config:
    @staticmethod
    def _un_escape(s: str) -> str:
        return s.encode().decode('unicode_escape')

    @staticmethod
    def _unit_to_inches(s: str) -> float:
        s = s.strip()
        conversions: Final[Dict[str, float]] = {
            '″': 1.0,
            '′′': 1.0,
            '"': 1.0,
            "''": 1.0,
            'in': 1.0,
            'inch': 1.0,
            'inches': 1.0,
            'ft': 1.0 / 12.0,
            '′': 1.0 / 12.0,
            "'": 1.0 / 12.0,
            'yd': 1.0 / 36.0,
            'mm': 25.4,
            'cm': 2.54,
            'm': 0.0254,
            'pt': 72.0,
        }
        key: str
        value: float
        for key, value in conversions.items():
            if s.endswith(key):
                return float(s[:-len(key)].strip()) / value
        return float(s)

    def __init__(self, config_filename: Union[str, Path]) -> None:
        c: configparser.ConfigParser = configparser.ConfigParser(default_section='general', interpolation=None)
        c.read(config_filename, encoding='utf-8')

        self.verbosity: Final[int] = c.getint('general', 'verbosity', fallback=1)
        self.recursive_search: Final[bool] = c.getboolean('general', 'recursive search', fallback=False)

        self.filename_mask: Final[str] = c.get('reading', 'filename mask', fallback='*')
        self.sync_channel: Final[int] = c.getint('reading', 'sync channel', fallback=1) - 1
        self.signal_channel: Final[int] = c.getint('reading', 'signal channel', fallback=2) - 1
        self.sync_threshold: Final[float] = c.getfloat('reading', 'sync threshold', fallback=1.0)
        self.calibration_factor: Final[float] = c.getfloat('reading', 'calibration factor', fallback=1.0)
        self.min_depth: Final[float] = c.getfloat('reading', 'min depth', fallback=0.0)
        self.max_depth: Final[float] = c.getfloat('reading', 'max depth', fallback=np.inf)

        self.delimiter: Final[str] = self._un_escape(c.get('saving', 'delimiter', fallback='\t'))
        self.saving_directory: Final[Path] = Path(c.get('saving', 'directory', fallback='Imp'))
        self.img_directory: Final[Path] = Path(c.get('saving', 'graphics sub-directory', fallback='img'))
        self.saving_extension: Final[str] = c.get('saving', 'file name extension', fallback=CSV_SUFFIX)
        self.header: Final[bool] = c.getboolean('saving', 'header', fallback=True)

        self.save_initial_signal: Final[bool] = c.getboolean('initial signal', 'save', fallback=False)
        self.save_initial_signal_file_name: Final[str] = c.get('initial signal', 'file name',
                                                               fallback='IMPV')

        self.peak_files: Final[bool] = c.getboolean('peak files', 'save', fallback=True)
        self.peak_files_prefix: Final[str] = c.get('peak files', 'file name prefix', fallback='ALT_')

        self.peak_plot: Final[bool] = c.getboolean('peak plot', 'save', fallback=False)
        self.peak_plot_number: Final[float] = c.getint('peak plot', 'peak number', fallback=1)
        self.peak_plot_width: Final[float] = self._unit_to_inches(c.get('peak plot', 'width', fallback='6.4″'))
        self.peak_plot_height: Final[float] = self._unit_to_inches(c.get('peak plot', 'height', fallback='4.8″'))
        self.peak_plot_dpi: Final[float] = c.getfloat('peak plot', 'dpi', fallback=100.0)
        self.peak_plot_x_grid: Final[bool] = c.getboolean('peak plot', 'x grid', fallback=False)
        self.peak_plot_y_grid: Final[bool] = c.getboolean('peak plot', 'y grid', fallback=False)
        self.peak_plot_line_color: Final[str] = c.get('peak plot', 'line color', fallback='C0')
        self.peak_plot_file_name: Final[str] = c.get('peak plot', 'file name', fallback='peak plot')
        self.peak_plot_file_format: Final[str] = c.get('peak plot', 'file format', fallback='png')

        self.peak_parameters: Final[bool] = c.getboolean('peak parameters', 'save', fallback=True)
        self.peak_parameters_file_name: Final[str] = c.get('peak parameters', 'file name',
                                                           fallback='!ALT_PAR')

        self.peak_parameters_plots: Final[bool] = c.getboolean('peak parameters plots', 'save', fallback=False)
        self.peak_parameters_plots_width: Final[float] = \
            self._unit_to_inches(c.get('peak parameters plots', 'width', fallback='6.4″'))
        self.peak_parameters_plots_height: Final[float] = \
            self._unit_to_inches(c.get('peak parameters plots', 'height', fallback='4.8″'))
        self.peak_parameters_plots_dpi: Final[float] = c.getfloat('peak parameters plots', 'dpi', fallback=100.0)
        self.peak_parameters_plots_x_grid: Final[bool] = c.getboolean('peak parameters plots', 'x grid', fallback=False)
        self.peak_parameters_plots_y_grid: Final[bool] = c.getboolean('peak parameters plots', 'y grid', fallback=False)
        self.peak_parameters_plots_line_color: Final[str] = c.get('peak parameters plots', 'line color', fallback='C0')
        self.peak_parameters_plots_file_format: Final[str] = c.get('peak parameters plots', 'file format',
                                                                   fallback='png')

        self.averaged_peaks: Final[bool] = c.getboolean('averaged peaks', 'save', fallback=True)
        self.averaged_peaks_window: Final[int] = c.getint('averaged peaks', 'averaging window', fallback=0)
        self.averaged_peaks_files_prefix: Final[str] = c.get('averaged peaks', 'peak files prefix', fallback='!AV_ALT_')

        self.averaged_peaks_parameters: Final[bool] = c.getboolean('averaged peaks parameters', 'save', fallback=True)
        self.averaged_peaks_parameters_window: Final[int] = c.getint('averaged peaks parameters', 'averaging window',
                                                                     fallback=0)
        self.averaged_peaks_parameters_file_name: Final[str] = c.get('averaged peaks parameters',
                                                                     'parameters file name',
                                                                     fallback='!av_ALT_PAR')

        self.averaged_peak_plots: Final[bool] = c.getboolean('averaged peaks plots', 'save', fallback=False)
        self.averaged_peak_plots_window: Final[int] = c.getint('averaged peaks plots', 'averaging window', fallback=0)
        self.averaged_peak_plots_width: Final[float] = \
            self._unit_to_inches(c.get('averaged peaks plots', 'width', fallback='6.4″'))
        self.averaged_peak_plots_height: Final[float] = \
            self._unit_to_inches(c.get('averaged peaks plots', 'height', fallback='4.8″'))
        self.averaged_peak_plots_dpi: Final[float] = c.getfloat('averaged peaks plots', 'dpi', fallback=100.0)
        self.averaged_peak_plots_x_grid: Final[bool] = c.getboolean('averaged peaks plots', 'x grid', fallback=False)
        self.averaged_peak_plots_y_grid: Final[bool] = c.getboolean('averaged peaks plots', 'y grid', fallback=False)
        self.averaged_peak_plots_line_color: Final[str] = c.get('averaged peaks plots', 'line color', fallback='C0')
        self.averaged_peak_plots_file_name_prefix: Final[str] = c.get('averaged peaks plots', 'file name prefix',
                                                                      fallback='averaged peaks plot ')
        self.averaged_peak_plots_file_format: Final[str] = c.get('averaged peaks plots', 'file format', fallback='png')

        self.psd: Final[bool] = c.getboolean('psd', 'save', fallback=True)
        self.psd_window: Final[str] = c.get('psd', 'window', fallback='hann').lower()
        self.psd_window_parameters: Final[Tuple[float, ...]] = \
            tuple(map(float, c.get('psd', 'window parameters', fallback='').split()))
        self.psd_averaging_mode: Final[str] = c.get('psd', 'averaging mode', fallback='mean')
        self.psd_max_frequency: Final[float] = c.getfloat('psd', 'max frequency', fallback=1.0)
        self.psd_file_name: Final[str] = c.get('psd', 'file name', fallback='Spectrum5_1')

        self.psd_plot: Final[bool] = c.getboolean('psd plot', 'save', fallback=False)
        self.psd_plot_width: Final[float] = self._unit_to_inches(c.get('psd plot', 'width', fallback='6.4″'))
        self.psd_plot_height: Final[float] = self._unit_to_inches(c.get('psd plot', 'height', fallback='4.8″'))
        self.psd_plot_dpi: Final[float] = c.getfloat('psd plot', 'dpi', fallback=100.0)
        self.psd_plot_x_grid: Final[bool] = c.getboolean('psd plot', 'x grid', fallback=False)
        self.psd_plot_y_grid: Final[bool] = c.getboolean('psd plot', 'y grid', fallback=False)
        self.psd_plot_line_color: Final[str] = c.get('psd plot', 'line color', fallback='C0')
        self.psd_plot_file_name: Final[str] = c.get('psd plot', 'file name', fallback='psd plot')
        self.psd_plot_file_format: Final[str] = c.get('psd plot', 'file format', fallback='png')

        self.rolling_psd: Final[bool] = c.getboolean('rolling psd', 'save', fallback=True)
        self.rolling_psd_files_prefix: Final[str] = c.get('rolling psd', 'files prefix',
                                                          fallback='RollingSpectrum_')
        self.rolling_psd_window: Final[str] = c.get('rolling psd', 'window', fallback='100%')
        self.rolling_psd_window_shift: Final[str] = c.get('rolling psd', 'window shift', fallback='100%')
        self.rolling_psd_averaged_file_name: Final[str] = c.get('rolling psd', 'averaged file name',
                                                                fallback='AverRollingSpectrum')

        self.psd_statistics: Final[bool] = c.getboolean('psd statistics', 'save', fallback=True)
        self.psd_statistics_file_name: Final[str] = c.get('psd statistics', 'file name',
                                                          fallback='SrSpectr5_1')

        self.integrals: Final[bool] = c.getboolean('integrals', 'save', fallback=True)
        self.integrals_file_name: Final[str] = c.get('integrals', 'file name', fallback='0Integrali_obrezka5100')

        self.statistics: Final[bool] = c.getboolean('statistics', 'save', fallback=False)
        self.statistics_file_path: Final[Path] = Path(c.get('statistics', 'file path', fallback='statistics'))

    def __str__(self) -> str:
        ss: List[Tuple[str, Any]] = [
            ('sync channel', self.sync_channel + 1),
            ('signal channel', self.signal_channel + 1),
            ('sync threshold', self.sync_threshold),
            ('calibration factor', self.calibration_factor),
        ]

        if self.averaged_peaks:
            ss.append(('averaged peaks window width', self.averaged_peaks_window))

        if self.averaged_peaks_parameters:
            ss.append(('averaged peaks parameters window width', self.averaged_peaks_parameters_window))

        if self.averaged_peak_plots:
            ss.append(('averaged peak plots window width', self.averaged_peak_plots_window))

        if self.psd:
            ss.append(('psd window shape', self.psd_window))
            if self.psd_window_parameters:
                ss.append(('psd window parameters', self.psd_window_parameters))
            ss.extend([
                ('psd averaging mode', self.psd_averaging_mode),
                ('psd max frequency', self.psd_max_frequency),
            ])

        if self.rolling_psd:
            ss.extend([
                ('rolling psd window width', self.rolling_psd_window),
                ('rolling psd window shift', self.rolling_psd_window_shift),
            ])
        return '\n'.join(f'{s[0]}: {repr(s[1])}' for s in ss)


# ------------------------------------------------------------------------
# Основная программа
# ------------------------------------------------------------------------
def main(c: Config) -> None:
    def list_files(path: Path, *, suffix: str = '', recursive: bool = True) -> List[Path]:
        matching_files: List[Path] = []
        if recursive and path.is_dir():
            for file in path.iterdir():
                matching_files.extend(list_files(file, suffix=suffix))
        elif path.is_file() and (not suffix or path.suffix == suffix):
            matching_files.append(path)
        return matching_files

    files: List[Path] = []
    if Path(c.filename_mask).is_absolute():  # not implemented in Path
        for glob_path in Path(c.filename_mask).parent.glob(Path(c.filename_mask).name):
            files.extend(list_files(glob_path, suffix=DAT_SUFFIX, recursive=c.recursive_search))
    else:
        for glob_path in Path.cwd().glob(c.filename_mask):
            files.extend(list_files(glob_path, suffix=DAT_SUFFIX, recursive=c.recursive_search))
    for fn in files:
        if fn.exists():
            process(fn, c)
    if c.verbosity > 1:
        print('\r' + ' ' * 6 + '\r', end='', flush=True)
    if c.verbosity:
        print('done')


def process(filename: Path, c: Config) -> None:
    def parameters_str() -> str:
        s: str = ''
        for parameter_name in parameters.parameter_names:
            parameter_value: Any = getattr(parameters, parameter_name)
            if isinstance(parameter_value, bytes):
                parameter_value = parameter_value.decode()
            s += f'{parameter_name.replace("_", " ")}: {parameter_value}\n'
        return s

    def save_initial_signal() -> None:
        if c.verbosity:
            print('saving text ...', end='', flush=True)
        utils.save_txt(saving_path / (c.save_initial_signal_file_name + c.saving_extension),
                       np.column_stack((t.astype(np.single), sn.astype(np.single), sc.astype(np.single))),
                       header=c.delimiter.join(('Time [s]', 'Signal [V]', 'Sync [V]')) if c.header else '',
                       fmt=c.delimiter.join(('%.7f', '%.8f', '%.8f')),
                       newline=os.linesep, encoding='utf-8')
        if c.verbosity:
            print(' done')

    def save_parameters() -> None:
        with (saving_path / 'parameters.txt').open('w', encoding='utf-8') as f_out:
            f_out.write(parameters_str())
            f_out.write(str(c))

    def save_statistics() -> None:
        statistics_file_path: Path
        if c.statistics_file_path.suffix == c.saving_extension:
            statistics_file_path = c.statistics_file_path
        else:
            statistics_file_path = c.statistics_file_path.with_name(c.statistics_file_path.name + c.saving_extension)
        if c.verbosity > 1:
            print('saving statistics to', statistics_file_path)
        if not statistics_file_path.exists():
            with statistics_file_path.open('w', encoding='utf-8') as f_out:
                f_out.write(c.delimiter.join(stat.keys()) + '\n')
        with statistics_file_path.open('a', encoding='utf-8') as f_out:
            f_out.write(c.delimiter.join(map(str, stat.values())) + '\n')

    def alt() -> Dict[str, Union[float, int, NDArray[np.float_]]]:
        h_t: Final[float] = c.min_depth * 2. / SOUND_SPEED
        h_max_t: Final[float] = c.max_depth * 2. / SOUND_SPEED

        na: int

        def save_peaks() -> None:
            def save_peak() -> None:
                if c.verbosity > 1:
                    # display progress
                    print(f'\r{_first_signal_index / first_signal_indexes[-1]:7.2%}', end='', flush=True)
                np.savetxt(saving_path / f'{c.peak_files_prefix}{len(ts):0{index_width}}{c.saving_extension}',
                           np.column_stack((ts[-1], sns[-1])),
                           header=os.linesep.join((c.delimiter.join(('dt', 'sn')), f't0 = {_t0}')) if c.header else '',
                           fmt=('%.6f', '%.6f'),
                           delimiter=c.delimiter, newline=os.linesep, encoding='utf-8')

            def save_peak_plot() -> None:
                if c.verbosity > 1:
                    print('saving plot for peak', c.peak_plot_number)
                fig, ax = plt.subplots()
                fig.set_size_inches(c.peak_plot_width, c.peak_plot_height, forward=True)
                fig.set_dpi(c.peak_plot_dpi)
                ax.plot(ts[-1], sns[-1], color=c.peak_plot_line_color)
                if c.peak_plot_x_grid:
                    ax.grid(axis='x')
                if c.peak_plot_y_grid:
                    ax.grid(axis='y')
                ax.set_xlabel('Время, с')
                ax.set_ylabel('Сигнал, В')
                fig.tight_layout()
                (saving_path / c.img_directory).mkdir(exist_ok=True, parents=True)
                fig.savefig(saving_path / c.img_directory / f'{c.peak_plot_file_name}.{c.peak_plot_file_format}',
                            bbox_inches='tight')
                plt.close(fig)

            index_width: int = len(str(init_start_indices_size))
            max_signal_duration: int = max(signal_durations)
            if c.peak_files and c.verbosity > 1:
                print('saving peaks')
            for _first_signal_index, _t0 in zip(first_signal_indexes, t0s):
                _last_signal_index: int = _first_signal_index + max_signal_duration
                ts.append(t[_first_signal_index:_last_signal_index] - _t0)
                sns.append(sn[_first_signal_index:_last_signal_index])

                if c.peak_files:
                    save_peak()

                if c.peak_plot and len(sns) == c.peak_plot_number:
                    save_peak_plot()

            if c.verbosity > 1:
                print('\r' + ' ' * 8 + '\r', end='', flush=True)

        def save_peaks_parameters() -> None:
            if c.verbosity > 1:
                print('saving peaks parameters')
            np.savetxt(saving_path / (c.peak_parameters_file_name + c.saving_extension),
                       np.column_stack((times,
                                        norms,
                                        mws,
                                        disps,
                                        m4s,
                                        maximum_sns,
                                        maximum_time_to_hs)),
                       header=c.delimiter.join('TIME NORM MW DISP M4 MAXSN TIMEMAXS'.split()) if c.header else '',
                       fmt='%s',
                       delimiter=c.delimiter, newline=os.linesep, encoding='utf-8')

        def save_peaks_parameters_plots() -> None:
            if c.verbosity > 1:
                print('saving peaks parameters plots')
            for peak_parameter, ax_label in zip(
                    (norms,
                     mws,
                     disps,
                     m4s,
                     maximum_sns,
                     maximum_time_to_hs),
                    'NORM MW DISP M4 MAXSN TIMEMAXS'.split()
            ):
                fig, ax = plt.subplots()
                fig.set_size_inches(c.peak_parameters_plots_width, c.peak_parameters_plots_height, forward=True)
                fig.set_dpi(c.peak_parameters_plots_dpi)
                ax.plot(times, peak_parameter, color=c.peak_parameters_plots_line_color)
                if c.peak_parameters_plots_x_grid:
                    ax.grid(axis='x')
                if c.peak_parameters_plots_y_grid:
                    ax.grid(axis='y')
                ax.set_xlabel('Время, с')
                # TODO: set y label
                fig.tight_layout()
                (saving_path / c.img_directory).mkdir(exist_ok=True, parents=True)
                fig.savefig(saving_path / c.img_directory / f'{ax_label}.{c.peak_parameters_plots_file_format}',
                            bbox_inches='tight')
                plt.close(fig)

        def save_averaged_peaks() -> None:
            index_width: int = len(str(t0s.size // na))
            if c.verbosity > 1:
                print('saving averaged peaks')
            for i, n in enumerate(range(0, t0s.size, na)):
                _signal_time = np.mean(ts[n:na + n], axis=0)
                _signal = np.mean(sns[n:na + n], axis=0)
                np.savetxt(saving_path / f'{c.averaged_peaks_files_prefix}{i + 1:0{index_width}}{c.saving_extension}',
                           np.column_stack((_signal_time, _signal)),
                           header=(os.linesep.join((c.delimiter.join(('dt', 'sn')), f'time = {np.mean(_signal_time)}'))
                                   if c.header else ''),
                           fmt=('%.6f', '%.6f'),
                           delimiter=c.delimiter, newline=os.linesep, encoding='utf-8')

        def save_averaged_peaks_parameters() -> None:
            if c.verbosity > 1:
                print('saving averaged peaks parameters')
            with ((saving_path / (c.averaged_peaks_parameters_file_name + c.saving_extension))
                    .open('w', encoding='utf-8')) as f_out:
                if c.header:
                    f_out.write(c.delimiter.join('TIME NORM MW DISP M4 MAXSN TIMEMAXS'.split()) + '\n')
                for i, n in enumerate(range(0, t0s.size, na)):
                    av_pars_dict = av_pars(n, na + n)
                    f_out.write(c.delimiter.join(map(str, av_pars_dict.values())) + '\n')

        def save_averaged_peaks_plots() -> None:
            if c.verbosity > 1:
                print('saving averaged peaks plots')
            for i, n in enumerate(range(0, t0s.size, na)):
                _signal_time = np.mean(ts[n:na + n], axis=0)
                _signal = np.mean(sns[n:na + n], axis=0)
                fig, ax = plt.subplots()
                fig.set_size_inches(c.averaged_peak_plots_width, c.averaged_peak_plots_height, forward=True)
                fig.set_dpi(c.averaged_peak_plots_dpi)
                ax.plot(_signal_time, _signal, color=c.averaged_peak_plots_line_color)
                if c.averaged_peak_plots_x_grid:
                    ax.grid(axis='x')
                if c.averaged_peak_plots_y_grid:
                    ax.grid(axis='y')
                ax.set_xlabel('Время, с')
                ax.set_ylabel('Сигнал, В')
                fig.tight_layout()
                (saving_path / c.img_directory).mkdir(exist_ok=True, parents=True)
                fig.savefig(saving_path / c.img_directory
                            / f'{c.averaged_peak_plots_file_name_prefix}{i + 1}.{c.psd_plot_file_format}',
                            bbox_inches='tight')
                plt.close(fig)

        def rls(time: NDArray[np.float_], ht: NDArray[np.float_]) -> Mapping[str, float]:
            from scipy import signal  # type: ignore

            t_r2: float = time[-1] - time[-2]
            ht -= np.mean(ht)
            ht *= c.calibration_factor

            f: NDArray[np.float_]
            pn_xx: NDArray[np.float_]
            f, pn_xx = signal.welch(ht, 1. / t_r2, window=(c.psd_window, *c.psd_window_parameters),
                                    # nperseg=ht.size,
                                    average=c.psd_averaging_mode
                                    )
            pn_xx = pn_xx[f <= c.psd_max_frequency]
            f = f[f <= c.psd_max_frequency]
            sp: Final[NDArray[np.float_]] = np.multiply(pn_xx, 2.0 * t_r2)

            def save_psd() -> None:
                if c.verbosity > 1:
                    print('saving spectrum')
                np.savetxt(saving_path / (c.psd_file_name + c.saving_extension), np.column_stack((f, sp)),
                           header=c.delimiter.join(('Frequency [Hz]', 'PSD [V²/Hz]')) if c.header else '',
                           delimiter=c.delimiter, newline=os.linesep, encoding='utf-8')

            def save_psd_plot() -> None:
                if c.verbosity > 1:
                    print('saving spectrum plot')
                fig, ax = plt.subplots()
                fig.set_size_inches(c.psd_plot_width, c.psd_plot_height, forward=True)
                fig.set_dpi(c.psd_plot_dpi)
                ax.plot(f, sp, color=c.psd_plot_line_color)
                if c.psd_plot_x_grid:
                    ax.grid(axis='x')
                if c.psd_plot_y_grid:
                    ax.grid(axis='y')
                ax.set_xlabel('Частота, Гц')
                ax.set_ylabel('Спектральная плотность мощности, В²/Гц')
                fig.tight_layout()
                (saving_path / c.img_directory).mkdir(exist_ok=True, parents=True)
                fig.savefig(saving_path / c.img_directory / f'{c.psd_plot_file_name}.{c.psd_plot_file_format}',
                            bbox_inches='tight')
                plt.close(fig)

            def save_rolling_psd() -> None:
                def save_averaged_rolling_psd() -> None:
                    if c.verbosity > 1:
                        print('saving averaged rolling spectra')
                    if rolling_pn_xxs is not None:
                        np.savetxt(saving_path / f'{c.rolling_psd_averaged_file_name}{c.saving_extension}',
                                   np.column_stack((rolling_f, 2.0 * rolling_pn_xxs * t_r2 / i)),
                                   header=c.delimiter.join(('Frequency [Hz]', 'PSD [V²/Hz]')) if c.header else '',
                                   delimiter=c.delimiter, newline=os.linesep, encoding='utf-8')
                    else:
                        print('ERROR: no rolling spectra has been calculated yet')

                if c.verbosity > 1:
                    print('saving rolling spectra')

                rolling_psd_window: float
                if c.rolling_psd_window.endswith('%'):
                    rolling_psd_window = float(c.rolling_psd_window.rstrip('%')) / 100. / f[np.argmax(pn_xx)]
                else:
                    rolling_psd_window = float(c.rolling_psd_window)
                if rolling_psd_window < t_r2:
                    rolling_psd_window = 1. / f[np.argmax(pn_xx)]

                rolling_psd_window_shift: float
                if c.rolling_psd_window_shift.endswith('%'):
                    rolling_psd_window_shift = float(
                        c.rolling_psd_window_shift.rstrip('%')) / 100. * rolling_psd_window
                else:
                    rolling_psd_window_shift = float(c.rolling_psd_window_shift)

                index_width: int = len(str(int((time[-1] - time[0] - rolling_psd_window) / rolling_psd_window_shift)))
                i: int = 1
                rolling_psd_window_start: float = time[0]
                rolling_psd_window_end: float = rolling_psd_window_start + rolling_psd_window
                rolling_pn_xxs: Union[None, NDArray[np.float_]] = None
                rolling_f: Union[None, NDArray[np.float_]] = None
                rolling_psd_averaging_failed: bool = False
                while time[-1] >= rolling_psd_window_end:
                    rolling_ht: NDArray[np.float_] = \
                        ht[np.argwhere((time >= rolling_psd_window_start) & (time <= rolling_psd_window_end)).ravel()]
                    rolling_f, rolling_pn_xx = signal.welch(rolling_ht, 1. / t_r2,
                                                            window=(c.psd_window, *c.psd_window_parameters),
                                                            nperseg=rolling_ht.size,
                                                            average=c.psd_averaging_mode)
                    if rolling_pn_xxs is None:
                        rolling_pn_xxs = rolling_pn_xx
                    elif rolling_pn_xxs.shape == rolling_pn_xx.shape:
                        rolling_pn_xxs += rolling_pn_xx
                    else:
                        rolling_psd_averaging_failed = True
                    np.savetxt(saving_path / f'{c.rolling_psd_files_prefix}{i:0{index_width}}{c.saving_extension}',
                               np.column_stack((rolling_f, 2 * rolling_pn_xx * t_r2)),
                               header=os.linesep.join(
                                   (c.delimiter.join(('Frequency [Hz]', 'PSD [V²/Hz]')),
                                    f'time from {rolling_psd_window_start} to {rolling_psd_window_end} [s]')
                               ) if c.header else '',
                               delimiter=c.delimiter, newline=os.linesep, encoding='utf-8')
                    rolling_psd_window_start += rolling_psd_window_shift
                    rolling_psd_window_end = rolling_psd_window_start + rolling_psd_window
                    i += 1
                if not rolling_psd_averaging_failed and rolling_f is not None and rolling_pn_xxs is not None:
                    save_averaged_rolling_psd()

            def save_psd_statistics() -> None:
                if c.verbosity > 1:
                    print('saving spectrum statistics')
                np.savetxt(saving_path / (c.psd_statistics_file_name + c.saving_extension),
                           np.column_stack((f, sp,
                                            np.square(2. * np.pi * f) * sp,
                                            np.square(k) * sp,
                                            np.power(k, 4) * sp)),
                           header=c.delimiter.join(('Частота', 'Спектр_возвышений', 'Спектр_скоростей',
                                                    'Спектр_наклонов', 'Спектр_кривизн')) if c.header else '',
                           delimiter=c.delimiter, newline=os.linesep, encoding='utf-8')

            def k_f(f15: NDArray[np.float_]) -> NDArray[np.float_]:
                water_surface_tension: Final[float] = 0.000074  # [N/mm]
                g: Final[float] = 9.8  # [m/s²]

                _omega_squared: Final[NDArray[np.float_]] = np.square(2.0 * np.pi * f15)  # [1/s²]
                q15: Final[NDArray[np.float_]] = ((g / water_surface_tension / 3) ** 3
                                                  + np.square(_omega_squared / water_surface_tension / 2))
                k15: Final[NDArray[np.float_]] = (np.cbrt(np.sqrt(q15) + _omega_squared / water_surface_tension / 2)
                                                  - np.cbrt(np.sqrt(q15) - _omega_squared / water_surface_tension / 2))
                return k15

            def integrals() -> Dict[str, float]:
                sigma_hs: float = cast(float, np.sum(sp[1:] * df))
                sigma_h: float = cast(float, np.square(np.std(ht)))
                integral: float = cast(float, np.sum(sp[1:] * np.square(2. * np.pi * f[1:]) * df))
                disp_vtt: float = cast(float, np.square(np.std(dvtt)))
                integral1: float = cast(float, np.sum(sp[1:] * np.square(k[1:]) * df))
                integral2: float = cast(float, np.sum(sp[1:] * k[1:] * 2. * np.pi * f[1:] * df))

                f_shift: float = cast(float, np.sum(f[1:] * sp[1:] * df) / sigma_hs)
                delta_f: float = cast(float, 2.0 * np.sqrt(np.sum(np.square(f[1:]) * sp[1:] * df)
                                                           / sigma_hs - f_shift ** 2))
                mu_2: float = cast(float, np.sum((f[1:] - f[1:].mean()) ** 2 * sp[1:] * df))
                mu_3: float = cast(float, np.sum((f[1:] - f[1:].mean()) ** 3 * sp[1:] * df))
                mu_4: float = cast(float, np.sum((f[1:] - f[1:].mean()) ** 4 * sp[1:] * df))
                delta_f_42: float = cast(float, 2.0 * np.sqrt(mu_4 / mu_2))
                a: float = mu_3 / mu_2
                e: float = mu_4 / mu_2 ** 2 - 3.0
                return dict(zip(
                    (
                        'Время_записи',
                        'ДиспВысСпектр', '4*корень(ДиспВысСпектр)',
                        'ДиспВысПроц', '4*корень(ДиспВысПроц)',
                        'ДиспСкорСпектр',
                        'ДиспСкорПроц',
                        'Наклоны',
                        'ВзКор',

                        'F_shift',
                        'ΔF',
                        'ΔF_42',
                        'A',
                        'E',
                    ),
                    (
                        cast(float, np.max(time)),
                        sigma_hs, 4.0 * cast(float, np.sqrt(sigma_hs)),
                        sigma_h, 4.0 * cast(float, np.sqrt(sigma_h)),
                        integral,
                        disp_vtt,
                        integral1,
                        integral2,

                        f_shift,
                        delta_f,
                        delta_f_42,
                        a,
                        e,
                    )
                ))

            def save_integrals() -> None:
                if c.verbosity > 1:
                    print('saving integrals')
                np.savetxt(saving_path / (c.integrals_file_name + c.saving_extension),
                           np.column_stack(integrals_dict.values()),
                           header=c.delimiter.join(integrals_dict.keys()) if c.header else '',
                           delimiter=c.delimiter, newline=os.linesep, encoding='utf-8')

            if c.psd:
                save_psd()

            if c.psd_plot:
                save_psd_plot()

            if c.rolling_psd:
                save_rolling_psd()

            dvtt: NDArray[np.float_] = (ht[1:] - ht[:-1]) / (time[1:] - time[:-1])

            df: NDArray[np.float_] = f[1:] - f[:-1]

            k = k_f(f)
            if c.psd_statistics:
                save_psd_statistics()

            integrals_dict: Dict[str, float] = integrals()
            if c.integrals:
                save_integrals()
            return integrals_dict

        def av_pars(n_init: int = 0, n_fin: Union[None, int] = None) -> Dict[str, float]:
            _signal_time = np.mean(ts[n_init:n_fin], axis=0)
            _signal = np.mean(sns[n_init:n_fin], axis=0)
            _dt = abs(_signal_time[-1] - h_t)
            _norm = _dt * np.mean(_signal)
            _mw = _dt * np.mean(_signal_time * _signal) / _norm
            _disp = _dt * np.mean(np.square(_signal_time) * _signal) / _norm - np.square(_mw)
            _m4 = _dt * np.mean(np.power(_signal_time, 4) * _signal) / _norm
            _m4 = np.sqrt(np.square(12. * np.square(_mw)) - (np.power(_mw, 4) - _m4) / 3.)
            _arg_max_sn = np.argmax(_signal)
            _maximum_sn = _signal[_arg_max_sn]
            _maximum_time = _signal_time[_arg_max_sn]
            return dict(zip(
                'TIME NORM MW DISP M4 MAXSN TIMEMAXS'.split(),
                (
                    np.mean(_signal_time) + t0s[na + n_init - 1],
                    _norm,
                    _mw,
                    _disp,
                    _m4,
                    _maximum_sn,
                    SOUND_SPEED * _maximum_time / 2.,
                )
            ))

        start_indices = np.argwhere((sc[1:] > c.sync_threshold) & (sc[:-1] <= c.sync_threshold)).ravel()
        init_start_indices_size: int = start_indices.size
        t0s: NDArray[np.float_] = np.empty(0)
        ts: List[NDArray[np.float_]] = []
        sns: List[NDArray[np.float_]] = []
        first_signal_indexes: List[int] = []
        signal_durations: List[int] = []
        while start_indices.size:
            t0 = t[start_indices[0] + 1]
            if t[-1] < h_max_t + t0:
                break
            t0s = np.append(t0s, t0)
            first_signal_index = np.searchsorted(t, h_t + t0, side='right')
            last_signal_index = np.searchsorted(t, h_max_t + t0, side='right')
            first_signal_indexes.append(first_signal_index)
            signal_durations.append(last_signal_index - first_signal_index)
            start_indices = start_indices[start_indices > last_signal_index]

        save_peaks()

        index: int
        start_time: float
        signal_time: NDArray[np.float_]
        signal: NDArray[np.float_]

        arg_max_sns: List[int] = []
        arg_max_sn: int
        for signal_time, signal in zip(ts, sns):
            arg_max_sn = cast(int, np.argmax(signal))
            arg_max_sns.append(arg_max_sn)
        times: NDArray[np.float_] = np.empty(len(arg_max_sns))
        maximum_sns: NDArray[np.float_] = np.empty(len(arg_max_sns))
        maximum_times: NDArray[np.float_] = np.empty(len(arg_max_sns))
        norms: NDArray[np.float_] = np.empty(len(arg_max_sns))
        mws: NDArray[np.float_] = np.empty(len(arg_max_sns))
        disps: NDArray[np.float_] = np.empty(len(arg_max_sns))
        m4s: NDArray[np.float_] = np.empty(len(arg_max_sns))

        for index, (start_time, signal_time, signal, arg_max_sn) in enumerate(zip(t0s, ts, sns, arg_max_sns)):
            times[index] = np.mean(signal_time) + start_time
            maximum_sns[index] = signal[arg_max_sn]
            maximum_times[index] = signal_time[arg_max_sn]
            dt = abs(signal_time[-1] - h_t)
            norms[index] = dt * np.mean(signal)
            mws[index] = dt * np.mean(signal_time * signal) / norms[index]
            disps[index] = dt * np.mean(np.square(signal_time) * signal) / norms[index] - np.square(mws[index])
            m4s[index] = dt * np.mean(np.power(signal_time, 4) * signal) / norms[index]
            m4s[index] = np.sqrt((12. * np.square(mws[index])) ** 2 - (np.power(mws[index], 4) - m4s[index]) / 3.)
        maximum_time_to_hs: NDArray[np.float_] = np.multiply(maximum_times, SOUND_SPEED / 2.)

        # # TODO: apply filtering to `maximum_time_to_hs` to reject too deviant values
        # values, counts = np.unique(maximum_time_to_hs, return_counts=True)
        # counts = counts.astype(np.float_) / maximum_time_to_hs.size
        # threshold: float = 1e-3
        # values = values[counts > threshold]
        # good = (maximum_time_to_hs >= np.min(values)) & (maximum_time_to_hs <= np.max(values))
        # t0s = t0s[:maximum_time_to_hs.size][good]
        # times = times[good]
        # maximum_sns = maximum_sns[good]
        # maximum_time_to_hs = maximum_time_to_hs[good]
        # norms = norms[good]
        # mws = mws[good]
        # disps = disps[good]
        # m4s = m4s[good]

        if c.peak_parameters:
            save_peaks_parameters()
        if c.peak_parameters_plots:
            save_peaks_parameters_plots()

        if c.averaged_peaks_window < 1:
            na = t0s.size
        else:
            na = c.averaged_peaks_window
        if na != 1 and c.averaged_peaks:
            save_averaged_peaks()

        if c.averaged_peaks_parameters_window < 1:
            na = t0s.size
        else:
            na = c.averaged_peaks_parameters_window
        if na != 1 and c.averaged_peaks_parameters:
            save_averaged_peaks_parameters()

        if c.averaged_peak_plots_window < 1:
            na = t0s.size
        else:
            na = c.averaged_peak_plots_window
        if na != 1 and c.averaged_peak_plots:
            save_averaged_peaks_plots()

        return {'pulse count': t0s.size,
                **av_pars(),
                **rls(t0s, maximum_time_to_hs)}

    saving_path: Path
    source_file_name: Path
    pars_file_name: Path
    if filename.suffix.lower() == DAT_SUFFIX:
        saving_path = filename.with_suffix('') / c.saving_directory
        # файл исходных бинарных данных
        source_file_name = filename
        # файл параметров файла данных
        pars_file_name = filename.with_suffix(PAR_SUFFIX)
    else:
        saving_path = filename.with_name(filename.name + '_result') / c.saving_directory
        # файл исходных бинарных данных
        source_file_name = filename.with_name(filename.name + DAT_SUFFIX)
        # файл параметров файла данных
        pars_file_name = filename.with_name(filename.name + PAR_SUFFIX)
    if not source_file_name.exists() or not pars_file_name.exists():
        return
    saving_path.mkdir(exist_ok=True, parents=True)

    fp_pars_file: BinaryIO
    with pars_file_name.open('rb') as fp_pars_file:
        # пробуем зачитать файл параметров
        parameters = utils.LCardParameters(fp_pars_file)

    fp_source_file: BinaryIO
    t: NDArray[np.float_] = np.full(parameters.frames_count, np.nan)
    sn: NDArray[np.float_] = np.full(parameters.frames_count, np.nan)
    sc: NDArray[np.float_] = np.full(parameters.frames_count, np.nan)
    samples_count_portion: int = round(psutil.virtual_memory().available / parameters.DATA_TYPE_SIZE / 16)

    # read data frames
    with source_file_name.open('rb', buffering=samples_count_portion) as fp_source_file:
        if c.verbosity:
            print('Processing', source_file_name)
            if c.verbosity > 1:
                print(parameters_str())
        sc_frame_number: int = 0
        sn_frame_number: int = 0
        remaining_samples_count: int = parameters.samples_count
        ch: int = 0
        while remaining_samples_count > 0:
            # прочитаем из файла очередную порция бинарных данных
            data_buffer = utils.read_numpy_array(fp_source_file, parameters.DATA_NUMPY_TYPE,
                                                 parameters.samples_count % samples_count_portion)
            actual_samples_count: int = data_buffer.size
            sc_frame_count: int = ((actual_samples_count + ch) // parameters.channels_count
                                   - (ch > c.sync_channel)
                                   + ((actual_samples_count + ch) % parameters.channels_count > c.sync_channel)
                                   )
            sn_frame_count: int = ((actual_samples_count + ch) // parameters.channels_count
                                   - (ch > c.signal_channel)
                                   + ((actual_samples_count + ch) % parameters.channels_count > c.signal_channel)
                                   )

            sc[sc_frame_number:sc_frame_number + sc_frame_count] = \
                data_buffer[(parameters.channels_count - ch + c.sync_channel) % parameters.channels_count::
                            parameters.channels_count]
            sn[sn_frame_number:sn_frame_number + sn_frame_count] = \
                data_buffer[(parameters.channels_count - ch + c.signal_channel) % parameters.channels_count::
                            parameters.channels_count]

            sc_frame_number += sc_frame_count
            sn_frame_number += sn_frame_count

            ch = (ch + actual_samples_count) % parameters.channels_count

            if not actual_samples_count:
                print('Unexpected end of file', source_file_name)
                break

            remaining_samples_count -= actual_samples_count
        t[:sc_frame_number] = np.arange(sc_frame_number) * 1e-3 / parameters.channel_rate
    sc += parameters.correction_offset[0]
    sn += parameters.correction_offset[0]
    if not parameters.RESCALING_REQUIRED:
        sc *= parameters.correction_factor[0]
        sn *= parameters.correction_factor[0]
    elif parameters.MAX_ADC_CODE:
        sc *= (parameters.correction_factor[0] / parameters.MAX_ADC_CODE * parameters.MAX_ADC_VOLTAGE)
        sn *= (parameters.correction_factor[0] / parameters.MAX_ADC_CODE * parameters.MAX_ADC_VOLTAGE)

    if c.save_initial_signal:
        save_initial_signal()

    save_parameters()

    if not np.isnan(t[0]) and not np.isnan(sn[0]) and not np.isnan(sc[0]):
        stat = {'filename': source_file_name.with_suffix(''),
                'date&time': parameters.time.decode(),
                **alt()}
        if c.statistics:
            save_statistics()


if __name__ == '__main__':
    # # выведем информационную строку
    # print("File converter from 'bin' format to 'text' for L-Graph I data files.")
    # print('Copyright © 2008 L-Card Ltd.')
    if len(sys.argv) > 2:
        print(f'Usage: {Path(sys.argv[0]).name} [CONFIG]')
    elif len(sys.argv) == 2 and Path(sys.argv[1]).exists():
        main(Config(sys.argv[1]))
    elif Path(sys.argv[0]).with_suffix(INI_SUFFIX).exists():
        main(Config(Path(sys.argv[0]).with_suffix(INI_SUFFIX)))
    else:
        print('No configuration file found')
