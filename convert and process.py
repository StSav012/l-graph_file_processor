# coding: utf-8
import configparser
import os
import sys
from pathlib import Path
from typing import List, Tuple, Union, Any, Dict, BinaryIO, cast, Mapping, Optional

import numpy as np
import psutil  # type: ignore
import scipy.signal  # type: ignore
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
        self.save_initial_signal_file_name: Final[str] = c.get('initial signal', 'file name', fallback='IMPV')

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


class Reader:
    def __init__(self, filename: Path, c: Config) -> None:
        self.c: Config = c

        self.saving_path: Path
        self.source_file_name: Path
        pars_file_name: Path
        if filename.suffix.lower() == DAT_SUFFIX:
            self.saving_path = filename.with_suffix('') / self.c.saving_directory
            # файл исходных бинарных данных
            self.source_file_name = filename
            # файл параметров файла данных
            pars_file_name = filename.with_suffix(PAR_SUFFIX)
        else:
            self.saving_path = filename.with_name(filename.name + '_result') / self.c.saving_directory
            # файл исходных бинарных данных
            self.source_file_name = filename.with_name(filename.name + DAT_SUFFIX)
            # файл параметров файла данных
            pars_file_name = filename.with_name(filename.name + PAR_SUFFIX)
        if not self.source_file_name.exists() or not pars_file_name.exists():
            return
        self.saving_path.mkdir(exist_ok=True, parents=True)

        fp_pars_file: BinaryIO
        with pars_file_name.open('rb') as fp_pars_file:
            # пробуем зачитать файл параметров
            self.parameters: utils.LCardParameters = utils.LCardParameters(fp_pars_file)

        fp_source_file: BinaryIO
        self.t: NDArray[np.float_] = np.full(self.parameters.frames_count, np.nan)
        self.sn: NDArray[np.float_] = np.full(self.parameters.frames_count, np.nan)
        self.sc: NDArray[np.float_] = np.full(self.parameters.frames_count, np.nan)
        self.samples_count_portion: int = round(psutil.virtual_memory().available / self.parameters.DATA_TYPE_SIZE / 16)

        # read data frames
        with self.source_file_name.open('rb', buffering=self.samples_count_portion) as fp_source_file:
            if self.c.verbosity:
                print('Processing', self.source_file_name)
                if self.c.verbosity > 1:
                    print(str(self.parameters))
            sc_frame_number: int = 0
            sn_frame_number: int = 0
            remaining_samples_count: int = self.parameters.samples_count
            ch: int = 0
            while remaining_samples_count > 0:
                # прочитаем из файла очередную порция бинарных данных
                data_buffer = utils.read_numpy_array(fp_source_file, self.parameters.DATA_NUMPY_TYPE,
                                                     self.parameters.samples_count % self.samples_count_portion)
                actual_samples_count: int = data_buffer.size
                sc_frame_count: int = ((actual_samples_count + ch) // self.parameters.channels_count
                                       - (ch > self.c.sync_channel)
                                       + ((actual_samples_count + ch)
                                          % self.parameters.channels_count > self.c.sync_channel)
                                       )
                sn_frame_count: int = ((actual_samples_count + ch) // self.parameters.channels_count
                                       - (ch > self.c.signal_channel)
                                       + ((actual_samples_count + ch)
                                          % self.parameters.channels_count > self.c.signal_channel)
                                       )

                self.sc[sc_frame_number:sc_frame_number + sc_frame_count] = \
                    data_buffer[(self.parameters.channels_count - ch + self.c.sync_channel)
                                % self.parameters.channels_count::
                                self.parameters.channels_count]
                self.sn[sn_frame_number:sn_frame_number + sn_frame_count] = \
                    data_buffer[(self.parameters.channels_count - ch + self.c.signal_channel)
                                % self.parameters.channels_count::
                                self.parameters.channels_count]

                sc_frame_number += sc_frame_count
                sn_frame_number += sn_frame_count

                ch = (ch + actual_samples_count) % self.parameters.channels_count

                if not actual_samples_count:
                    print('Unexpected end of file', self.source_file_name)
                    break

                remaining_samples_count -= actual_samples_count
            self.t[:sc_frame_number] = np.arange(sc_frame_number) * 1e-3 / self.parameters.channel_rate
        self.sc += self.parameters.correction_offset[0]
        self.sn += self.parameters.correction_offset[0]
        if not self.parameters.RESCALING_REQUIRED:
            self.sc *= self.parameters.correction_factor[0]
            self.sn *= self.parameters.correction_factor[0]
        elif self.parameters.MAX_ADC_CODE:
            self.sc *= (self.parameters.correction_factor[0]
                        / self.parameters.MAX_ADC_CODE * self.parameters.MAX_ADC_VOLTAGE)
            self.sn *= (self.parameters.correction_factor[0]
                        / self.parameters.MAX_ADC_CODE * self.parameters.MAX_ADC_VOLTAGE)


def k_f(f15: NDArray[np.float_]) -> NDArray[np.float_]:
    water_surface_tension: Final[float] = 0.000074  # [N/mm]
    g: Final[float] = 9.8  # [m/s²]

    omega_squared: Final[NDArray[np.float_]] = np.square(2.0 * np.pi * f15)  # [1/s²]
    q15: Final[NDArray[np.float_]] = ((g / water_surface_tension / 3) ** 3
                                      + np.square(omega_squared / water_surface_tension / 2))
    k15: Final[NDArray[np.float_]] = (np.cbrt(np.sqrt(q15) + omega_squared / water_surface_tension / 2)
                                      - np.cbrt(np.sqrt(q15) - omega_squared / water_surface_tension / 2))
    return k15


class RLS:
    def __init__(self, c: Config, time: NDArray[np.float_], ht: NDArray[np.float_]) -> None:
        self.time: NDArray[np.float_] = time
        self.ht: NDArray[np.float_] = ht

        self.t_r2: float = self.time[-1] - self.time[-2]
        self.ht -= np.mean(self.ht)
        self.ht *= c.calibration_factor

        self.f: NDArray[np.float_]
        self.pn_xx: NDArray[np.float_]
        self.f, self.pn_xx = scipy.signal.welch(self.ht, 1. / self.t_r2,
                                                window=(c.psd_window, *c.psd_window_parameters),
                                                # nperseg=ht.size,
                                                average=c.psd_averaging_mode
                                                )
        self.pn_xx = self.pn_xx[self.f <= c.psd_max_frequency]
        self.f = self.f[self.f <= c.psd_max_frequency]
        self.sp: Final[NDArray[np.float_]] = np.multiply(self.pn_xx, 2.0 * self.t_r2)

        self.dvtt: NDArray[np.float_] = (self.ht[1:] - self.ht[:-1]) / (self.time[1:] - self.time[:-1])

        self.df: NDArray[np.float_] = self.f[1:] - self.f[:-1]

        self.k = k_f(self.f)
        self.integrals_dict: Dict[str, float] = dict()

    def integrals(self) -> Dict[str, float]:
        sigma_hs: float = cast(float, np.sum(self.sp[1:] * self.df))
        sigma_h: float = cast(float, np.square(np.std(self.ht)))
        integral: float = cast(float, np.sum(self.sp[1:] * np.square(2. * np.pi * self.f[1:]) * self.df))
        disp_vtt: float = cast(float, np.square(np.std(self.dvtt)))
        integral1: float = cast(float, np.sum(self.sp[1:] * np.square(self.k[1:]) * self.df))
        integral2: float = cast(float, np.sum(self.sp[1:] * self.k[1:] * 2. * np.pi * self.f[1:] * self.df))

        f_shift: float = cast(float, np.sum(self.f[1:] * self.sp[1:] * self.df) / sigma_hs)
        delta_f: float = cast(float, 2.0 * np.sqrt(np.sum(np.square(self.f[1:]) * self.sp[1:] * self.df)
                                                   / sigma_hs - f_shift ** 2))
        mu_2: float = cast(float, np.sum((self.f[1:] - self.f[1:].mean()) ** 2 * self.sp[1:] * self.df))
        mu_3: float = cast(float, np.sum((self.f[1:] - self.f[1:].mean()) ** 3 * self.sp[1:] * self.df))
        mu_4: float = cast(float, np.sum((self.f[1:] - self.f[1:].mean()) ** 4 * self.sp[1:] * self.df))
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
                cast(float, np.max(self.time)),
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

    def rls(self) -> Mapping[str, float]:
        if not self.integrals_dict:
            self.integrals_dict = self.integrals()
        return self.integrals_dict


class Altimeter(Reader):
    def __init__(self, filename: Path, c: Config) -> None:
        super().__init__(filename, c)

        self.h_t: Final[float] = self.c.min_depth * 2. / SOUND_SPEED
        h_max_t: Final[float] = self.c.max_depth * 2. / SOUND_SPEED

        start_indices: NDArray[np.int_] = np.argwhere((self.sc[1:] > self.c.sync_threshold)
                                                      & (self.sc[:-1] <= self.c.sync_threshold)).ravel()
        self.init_start_indices_size: int = start_indices.size
        self.t0s: NDArray[np.float_] = np.empty(0)
        self.ts: List[NDArray[np.float_]] = []
        self.sns: List[NDArray[np.float_]] = []
        self.first_signal_indexes: List[int] = []
        self.signal_durations: List[int] = []
        while start_indices.size:
            t0 = self.t[start_indices[0] + 1]
            if self.t[-1] < h_max_t + t0:
                break
            self.t0s = np.append(self.t0s, t0)
            first_signal_index: int = np.searchsorted(self.t, self.h_t + t0, side='right')
            last_signal_index: int = np.searchsorted(self.t, h_max_t + t0, side='right')
            self.first_signal_indexes.append(first_signal_index)
            self.signal_durations.append(last_signal_index - first_signal_index)
            start_indices = start_indices[start_indices > last_signal_index]

        index: int
        start_time: float
        signal_time: NDArray[np.float_]
        signal: NDArray[np.float_]

        max_signal_duration: int = max(self.signal_durations)
        if self.c.peak_files and self.c.verbosity > 1:
            print('saving peaks')
        for index, (first_signal_index, t0) in enumerate(zip(self.first_signal_indexes, self.t0s)):
            _last_signal_index: int = first_signal_index + max_signal_duration
            self.ts.append(self.t[first_signal_index:_last_signal_index] - t0)
            self.sns.append(self.sn[first_signal_index:_last_signal_index])

        arg_max_sns: List[int] = []
        arg_max_sn: int
        for signal_time, signal in zip(self.ts, self.sns):
            arg_max_sn = cast(int, np.argmax(signal))
            arg_max_sns.append(arg_max_sn)
        self.times: NDArray[np.float_] = np.empty(len(arg_max_sns))
        self.maximum_sns: NDArray[np.float_] = np.empty(len(arg_max_sns))
        maximum_times: NDArray[np.float_] = np.empty(len(arg_max_sns))
        self.norms: NDArray[np.float_] = np.empty(len(arg_max_sns))
        self.mws: NDArray[np.float_] = np.empty(len(arg_max_sns))
        self.disps: NDArray[np.float_] = np.empty(len(arg_max_sns))
        self.m4s: NDArray[np.float_] = np.empty(len(arg_max_sns))

        for index, (start_time, signal_time, signal, arg_max_sn) \
                in enumerate(zip(self.t0s, self.ts, self.sns, arg_max_sns)):
            self.times[index] = np.mean(signal_time) + start_time
            self.maximum_sns[index] = signal[arg_max_sn]
            maximum_times[index] = signal_time[arg_max_sn]
            dt = abs(signal_time[-1] - self.h_t)
            self.norms[index] = dt * np.mean(signal)
            self.mws[index] = dt * np.mean(signal_time * signal) / self.norms[index]
            self.disps[index] = dt * (np.mean(np.square(signal_time) * signal) / self.norms[index]
                                      - np.square(self.mws[index]))
            self.m4s[index] = dt * np.mean(np.power(signal_time, 4) * signal) / self.norms[index]
            self.m4s[index] = np.sqrt((12. * np.square(self.mws[index])) ** 2
                                      - (np.power(self.mws[index], 4) - self.m4s[index]) / 3.)
        self.maximum_time_to_hs: NDArray[np.float_] = np.multiply(maximum_times, SOUND_SPEED / 2.)

        # # TODO: apply filtering to `self.maximum_time_to_hs` to reject too deviant values
        # values, counts = np.unique(self.maximum_time_to_hs, return_counts=True)
        # counts = counts.astype(np.float_) / self.maximum_time_to_hs.size
        # threshold: float = 1e-3
        # values = values[counts > threshold]
        # good = (self.maximum_time_to_hs >= np.min(values)) & (self.maximum_time_to_hs <= np.max(values))
        # self.t0s = self.t0s[:self.maximum_time_to_hs.size][good]
        # self.times = self.times[good]
        # self.maximum_sns = self.maximum_sns[good]
        # self.maximum_time_to_hs = self.maximum_time_to_hs[good]
        # self.norms = self.norms[good]
        # self.mws = self.mws[good]
        # self.disps = self.disps[good]
        # self.m4s = self.m4s[good]

        self.rls: RLS = RLS(c, self.t0s, self.maximum_time_to_hs)

    def av_pars(self, na: int, n_init: int = 0, n_fin: Optional[int] = None) -> Mapping[str, float]:
        signal_time = np.mean(self.ts[n_init:n_fin], axis=0)
        signal = np.mean(self.sns[n_init:n_fin], axis=0)
        dt = abs(signal_time[-1] - self.h_t)
        norm = dt * np.mean(signal)
        mw = dt * np.mean(signal_time * signal) / norm
        disp = dt * np.mean(np.square(signal_time) * signal) / norm - np.square(mw)
        m4 = dt * np.mean(np.power(signal_time, 4) * signal) / norm
        m4 = np.sqrt(np.square(12. * np.square(mw)) - (np.power(mw, 4) - m4) / 3.)
        arg_max_sn = np.argmax(signal)
        maximum_sn = signal[arg_max_sn]
        maximum_time = signal_time[arg_max_sn]
        return dict(zip(
            'TIME NORM MW DISP M4 MAXSN TIMEMAXS'.split(),
            (
                np.mean(signal_time) + self.t0s[na + n_init - 1],
                norm,
                mw,
                disp,
                m4,
                maximum_sn,
                SOUND_SPEED * maximum_time / 2.,
            )
        ))

    def alt(self) -> Dict[str, Union[float, int, NDArray[np.float_]]]:
        na: int
        if self.c.averaged_peaks_parameters_window < 1:
            na = self.t0s.size
        else:
            na = self.c.averaged_peaks_parameters_window
        return {'pulse count': self.t0s.size,
                **self.av_pars(na),
                **self.rls.rls()}


class Saviour(Altimeter):
    def save_initial_signal(self) -> None:
        if self.c.verbosity:
            print('saving text to '
                  f'{self.saving_path / (self.c.save_initial_signal_file_name + self.c.saving_extension)}...',
                  end='', flush=True)
        utils.save_txt(
            self.saving_path / (self.c.save_initial_signal_file_name + self.c.saving_extension),
            np.column_stack((self.t.astype(np.single), self.sn.astype(np.single), self.sc.astype(np.single))),
            header=self.c.delimiter.join(('Time [s]', 'Signal [V]', 'Sync [V]')) if self.c.header else '',
            fmt=self.c.delimiter.join(('%.7f', '%.8f', '%.8f')),
            newline=os.linesep, encoding='utf-8')
        if self.c.verbosity:
            print(' done')

    def save_parameters(self) -> None:
        with (self.saving_path / 'parameters.txt').open('w', encoding='utf-8') as f_out:
            f_out.write(str(self.parameters))
            f_out.write(str(self.c))

    def save_peak(self, i: int, t0: np.float64) -> None:
        index_width: int = len(str(self.init_start_indices_size))
        utils.save_txt(
            self.saving_path / f'{self.c.peak_files_prefix}{i + 1:0{index_width}}{self.c.saving_extension}',
            np.column_stack((self.ts[i], self.sns[i])),
            header='\n'.join((self.c.delimiter.join(('dt', 'self.sn')), f't0 = {t0}'))
            if self.c.header else '',
            fmt=('%.6f', '%.6f'),
            delimiter=self.c.delimiter, newline=os.linesep, encoding='utf-8')

    def save_peak_plot(self, i: int) -> None:
        if self.c.verbosity > 1:
            print('saving plot for peak', self.c.peak_plot_number)
        fig, ax = plt.subplots()
        fig.set_size_inches(self.c.peak_plot_width, self.c.peak_plot_height, forward=True)
        fig.set_dpi(self.c.peak_plot_dpi)
        ax.plot(self.ts[i], self.sns[i], color=self.c.peak_plot_line_color)
        if self.c.peak_plot_x_grid:
            ax.grid(axis='x')
        if self.c.peak_plot_y_grid:
            ax.grid(axis='y')
        ax.set_xlabel('Время, с')
        ax.set_ylabel('Сигнал, В')
        fig.tight_layout()
        (self.saving_path / self.c.img_directory).mkdir(exist_ok=True, parents=True)
        fig.savefig(self.saving_path / self.c.img_directory
                    / f'{self.c.peak_plot_file_name}.{self.c.peak_plot_file_format}',
                    bbox_inches='tight')
        plt.close(fig)

    def save_peaks(self) -> None:
        if self.c.peak_files and self.c.verbosity > 1:
            print('saving peaks')
        for index, (first_signal_index, t0) in enumerate(zip(self.first_signal_indexes, self.t0s)):
            if self.c.peak_files:
                if self.c.verbosity > 1:
                    # display progress
                    print(f'\r{first_signal_index / self.first_signal_indexes[-1]:7.2%}', end='', flush=True)
                self.save_peak(index, t0)

            if self.c.peak_plot and index + 1 == self.c.peak_plot_number:
                self.save_peak_plot(index)

        if self.c.verbosity > 1:
            print('\r' + ' ' * 8 + '\r', end='', flush=True)

    def save_peaks_parameters(self) -> None:
        if self.c.verbosity > 1:
            print('saving peaks parameters')
        utils.save_txt(self.saving_path / (self.c.peak_parameters_file_name + self.c.saving_extension),
                       np.column_stack((self.times,
                                        self.norms,
                                        self.mws,
                                        self.disps,
                                        self.m4s,
                                        self.maximum_sns,
                                        self.maximum_time_to_hs)),
                       header=self.c.delimiter.join('TIME NORM MW DISP M4 MAXSN TIMEMAXS'.split())
                       if self.c.header else '',
                       fmt='%s',
                       delimiter=self.c.delimiter, newline=os.linesep, encoding='utf-8')

    def save_peaks_parameters_plots(self) -> None:
        if self.c.verbosity > 1:
            print('saving peaks parameters plots')
        for peak_parameter, ax_label in zip(
                (self.norms,
                 self.mws,
                 self.disps,
                 self.m4s,
                 self.maximum_sns,
                 self.maximum_time_to_hs),
                'NORM MW DISP M4 MAXSN TIMEMAXS'.split()
        ):
            fig, ax = plt.subplots()
            fig.set_size_inches(self.c.peak_parameters_plots_width, self.c.peak_parameters_plots_height,
                                forward=True)
            fig.set_dpi(self.c.peak_parameters_plots_dpi)
            ax.plot(self.times, peak_parameter, color=self.c.peak_parameters_plots_line_color)
            if self.c.peak_parameters_plots_x_grid:
                ax.grid(axis='x')
            if self.c.peak_parameters_plots_y_grid:
                ax.grid(axis='y')
            ax.set_xlabel('Время, с')
            # TODO: set y label
            fig.tight_layout()
            (self.saving_path / self.c.img_directory).mkdir(exist_ok=True, parents=True)
            fig.savefig(self.saving_path / self.c.img_directory
                        / f'{ax_label}.{self.c.peak_parameters_plots_file_format}',
                        bbox_inches='tight')
            plt.close(fig)

    def save_averaged_peaks(self, na: int) -> None:
        index_width: int = len(str(self.t0s.size // na))
        if self.c.verbosity > 1:
            print('saving averaged peaks')
        for i, n in enumerate(range(0, self.t0s.size, na), start=1):
            _signal_time = np.mean(self.ts[n:na + n], axis=0)
            _signal = np.mean(self.sns[n:na + n], axis=0)
            utils.save_txt(self.saving_path
                           / f'{self.c.averaged_peaks_files_prefix}{i:0{index_width}}{self.c.saving_extension}',
                           np.column_stack((_signal_time, _signal)),
                           header=('\n'.join((self.c.delimiter.join(('dt', 'self.sn')),
                                              f'time = {np.mean(_signal_time)}'))
                                   if self.c.header else ''),
                           fmt=('%.6f', '%.6f'),
                           delimiter=self.c.delimiter, newline=os.linesep, encoding='utf-8')

    def save_averaged_peaks_parameters(self, na: int) -> None:
        if self.c.verbosity > 1:
            print('saving averaged peaks parameters')
        with ((self.saving_path / (self.c.averaged_peaks_parameters_file_name + self.c.saving_extension))
                .open('w', encoding='utf-8')) as f_out:
            if self.c.header:
                f_out.write(self.c.delimiter.join('TIME NORM MW DISP M4 MAXSN TIMEMAXS'.split()) + '\n')
            for i, n in enumerate(range(0, self.t0s.size, na)):
                av_pars_dict = self.av_pars(na, n, na + n)
                f_out.write(self.c.delimiter.join(map(str, av_pars_dict.values())) + '\n')

    def save_averaged_peaks_plots(self, na: int) -> None:
        if self.c.verbosity > 1:
            print('saving averaged peaks plots')
        for i, n in enumerate(range(0, self.t0s.size, na), start=1):
            _signal_time = np.mean(self.ts[n:na + n], axis=0)
            _signal = np.mean(self.sns[n:na + n], axis=0)
            fig, ax = plt.subplots()
            fig.set_size_inches(self.c.averaged_peak_plots_width, self.c.averaged_peak_plots_height, forward=True)
            fig.set_dpi(self.c.averaged_peak_plots_dpi)
            ax.plot(_signal_time, _signal, color=self.c.averaged_peak_plots_line_color)
            if self.c.averaged_peak_plots_x_grid:
                ax.grid(axis='x')
            if self.c.averaged_peak_plots_y_grid:
                ax.grid(axis='y')
            ax.set_xlabel('Время, с')
            ax.set_ylabel('Сигнал, В')
            fig.tight_layout()
            (self.saving_path / self.c.img_directory).mkdir(exist_ok=True, parents=True)
            fig.savefig(self.saving_path / self.c.img_directory
                        / f'{self.c.averaged_peak_plots_file_name_prefix}{i}.{self.c.psd_plot_file_format}',
                        bbox_inches='tight')
            plt.close(fig)

    def save_statistics(self) -> None:
        statistics_file_path: Path
        if self.c.statistics_file_path.suffix == self.c.saving_extension:
            statistics_file_path = self.c.statistics_file_path
        else:
            statistics_file_path = self.c.statistics_file_path.with_name(self.c.statistics_file_path.name
                                                                         + self.c.saving_extension)
        if self.c.verbosity > 1:
            print('saving statistics to', statistics_file_path)
        stat = {'filename': self.source_file_name.with_suffix(''),
                'date&time': self.parameters.time.decode(),
                **self.alt()}
        if not statistics_file_path.exists():
            with statistics_file_path.open('w', encoding='utf-8') as f_out:
                f_out.write(self.c.delimiter.join(stat.keys()) + '\n')
        with statistics_file_path.open('a', encoding='utf-8') as f_out:
            f_out.write(self.c.delimiter.join(map(str, stat.values())) + '\n')

    def save_psd(self) -> None:
        if self.c.verbosity > 1:
            print('saving spectrum')
        utils.save_txt(
            self.saving_path / (self.c.psd_file_name + self.c.saving_extension),
            np.column_stack((self.rls.f, self.rls.sp)),
            header=self.c.delimiter.join(('Frequency [Hz]', 'PSD [V²/Hz]')) if self.c.header else '',
            delimiter=self.c.delimiter, newline=os.linesep, encoding='utf-8')

    def save_psd_plot(self) -> None:
        if self.c.verbosity > 1:
            print('saving spectrum plot')
        fig, ax = plt.subplots()
        fig.set_size_inches(self.c.psd_plot_width, self.c.psd_plot_height, forward=True)
        fig.set_dpi(self.c.psd_plot_dpi)
        ax.plot(self.rls.f, self.rls.sp, color=self.c.psd_plot_line_color)
        if self.c.psd_plot_x_grid:
            ax.grid(axis='x')
        if self.c.psd_plot_y_grid:
            ax.grid(axis='y')
        ax.set_xlabel('Частота, Гц')
        ax.set_ylabel('Спектральная плотность мощности, В²/Гц')
        fig.tight_layout()
        (self.saving_path / self.c.img_directory).mkdir(exist_ok=True, parents=True)
        fig.savefig(self.saving_path / self.c.img_directory
                    / f'{self.c.psd_plot_file_name}.{self.c.psd_plot_file_format}',
                    bbox_inches='tight')
        plt.close(fig)

    def save_rolling_psd(self) -> None:
        def save_averaged_rolling_psd() -> None:
            if self.c.verbosity > 1:
                print('saving averaged rolling spectra')
            if rolling_pn_xxs is not None:
                utils.save_txt(
                    self.saving_path / f'{self.c.rolling_psd_averaged_file_name}{self.c.saving_extension}',
                    np.column_stack((rolling_f, 2.0 * rolling_pn_xxs * self.rls.t_r2 / i)),
                    header=self.c.delimiter.join(('Frequency [Hz]', 'PSD [V²/Hz]')) if self.c.header else '',
                    delimiter=self.c.delimiter, newline=os.linesep, encoding='utf-8')
            else:
                print('ERROR: no rolling spectra has been calculated yet')

        if self.c.verbosity > 1:
            print('saving rolling spectra')

        rolling_psd_window: float
        if self.c.rolling_psd_window.endswith('%'):
            rolling_psd_window = float(self.c.rolling_psd_window.rstrip('%')) / 100. \
                                 / self.rls.f[np.argmax(self.rls.pn_xx)]
        else:
            rolling_psd_window = float(self.c.rolling_psd_window)
        if rolling_psd_window < self.rls.t_r2:
            rolling_psd_window = 1. / self.rls.f[np.argmax(self.rls.pn_xx)]

        rolling_psd_window_shift: float
        if self.c.rolling_psd_window_shift.endswith('%'):
            rolling_psd_window_shift = float(
                self.c.rolling_psd_window_shift.rstrip('%')) / 100. * rolling_psd_window
        else:
            rolling_psd_window_shift = float(self.c.rolling_psd_window_shift)

        index_width: int = len(str(int((self.rls.time[-1] - self.rls.time[0] - rolling_psd_window)
                                       / rolling_psd_window_shift)))
        i: int = 1
        rolling_psd_window_start: float = self.rls.time[0]
        rolling_psd_window_end: float = rolling_psd_window_start + rolling_psd_window
        rolling_pn_xxs: Optional[NDArray[np.float_]] = None
        rolling_f: Optional[NDArray[np.float_]] = None
        rolling_psd_averaging_failed: bool = False
        while self.rls.time[-1] >= rolling_psd_window_end:
            rolling_ht: NDArray[np.float_] = \
                self.rls.ht[np.argwhere((self.rls.time >= rolling_psd_window_start)
                                        & (self.rls.time <= rolling_psd_window_end)).ravel()]
            rolling_f, rolling_pn_xx = scipy.signal.welch(rolling_ht, 1. / self.rls.t_r2,
                                                          window=(self.c.psd_window, *self.c.psd_window_parameters),
                                                          nperseg=rolling_ht.size,
                                                          average=self.c.psd_averaging_mode)
            if rolling_pn_xxs is None:
                rolling_pn_xxs = rolling_pn_xx
            elif rolling_pn_xxs.shape == rolling_pn_xx.shape:
                rolling_pn_xxs += rolling_pn_xx
            else:
                rolling_psd_averaging_failed = True
            utils.save_txt(self.saving_path
                           / f'{self.c.rolling_psd_files_prefix}{i:0{index_width}}{self.c.saving_extension}',
                           np.column_stack((rolling_f, 2 * rolling_pn_xx * self.rls.t_r2)),
                           header='\n'.join(
                               (self.c.delimiter.join(('Frequency [Hz]', 'PSD [V²/Hz]')),
                                f'time from {rolling_psd_window_start} to {rolling_psd_window_end} [s]')
                           ) if self.c.header else '',
                           delimiter=self.c.delimiter, newline=os.linesep, encoding='utf-8')
            rolling_psd_window_start += rolling_psd_window_shift
            rolling_psd_window_end = rolling_psd_window_start + rolling_psd_window
            i += 1
        if not rolling_psd_averaging_failed and rolling_f is not None and rolling_pn_xxs is not None:
            save_averaged_rolling_psd()

    def save_psd_statistics(self) -> None:
        if self.c.verbosity > 1:
            print('saving spectrum statistics')
        utils.save_txt(self.saving_path / (self.c.psd_statistics_file_name + self.c.saving_extension),
                       np.column_stack((self.rls.f, self.rls.sp,
                                        np.square(2. * np.pi * self.rls.f) * self.rls.sp,
                                        np.square(self.rls.k) * self.rls.sp,
                                        np.power(self.rls.k, 4) * self.rls.sp)),
                       header=self.c.delimiter.join(('Частота', 'Спектр_возвышений', 'Спектр_скоростей',
                                                     'Спектр_наклонов', 'Спектр_кривизн')) if self.c.header else '',
                       delimiter=self.c.delimiter, newline=os.linesep, encoding='utf-8')

    def save_integrals(self) -> None:
        if self.c.verbosity > 1:
            print('saving integrals')
        utils.save_txt(self.saving_path / (self.c.integrals_file_name + self.c.saving_extension),
                       np.column_stack(tuple(self.rls.rls().values())),
                       header=self.c.delimiter.join(self.rls.rls().keys()) if self.c.header else '',
                       delimiter=self.c.delimiter, newline=os.linesep, encoding='utf-8')

    def save_all_requested(self) -> None:
        if self.c.save_initial_signal:
            self.save_initial_signal()

        self.save_parameters()

        self.save_peaks()

        if self.c.peak_parameters:
            self.save_peaks_parameters()
        if self.c.peak_parameters_plots:
            self.save_peaks_parameters_plots()

        na: int

        if self.c.averaged_peaks_window < 1:
            na = self.t0s.size
        else:
            na = self.c.averaged_peaks_window
        if na != 1 and self.c.averaged_peaks:
            self.save_averaged_peaks(na)

        if self.c.averaged_peaks_parameters_window < 1:
            na = self.t0s.size
        else:
            na = self.c.averaged_peaks_parameters_window
        if na != 1 and self.c.averaged_peaks_parameters:
            self.save_averaged_peaks_parameters(na)

        if self.c.averaged_peak_plots_window < 1:
            na = self.t0s.size
        else:
            na = self.c.averaged_peak_plots_window
        if na != 1 and self.c.averaged_peak_plots:
            self.save_averaged_peaks_plots(na)

        if self.c.psd:
            self.save_psd()

        if self.c.psd_plot:
            self.save_psd_plot()

        if self.c.rolling_psd:
            self.save_rolling_psd()

        if self.c.integrals:
            self.save_integrals()

        if self.c.psd_statistics:
            self.save_psd_statistics()

        if not np.isnan(self.t[0]) and not np.isnan(self.sn[0]) and not np.isnan(self.sc[0]):
            if self.c.statistics:
                self.save_statistics()


def process(filename: Path, c: Config) -> None:
    saviour: Saviour = Saviour(filename, c)
    saviour.save_all_requested()


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
