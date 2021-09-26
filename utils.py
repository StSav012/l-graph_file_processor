# coding: utf-8
import os
import struct
import sys
from io import BufferedReader
from pathlib import Path
from typing import Tuple, Union, Dict, List, Type, BinaryIO, Optional

import numpy as np
from numpy.typing import NDArray, ArrayLike

if sys.version_info >= (3, 8):
    from typing import Final
else:
    # stub
    class Final:
        @staticmethod
        def __getitem__(item):
            return item


class LCardParameters:
    _struct: Final[str] = '<20s 17s 26s 2x h I I d f f f 32x 32x 32x 32x'
    _max_adc_codes: Final[Dict[bytes, int]] = {
        b'E140': 8000,
        b'E502': 6000000,
    }
    _adc_ranges: Final[Dict[bytes, List[float]]] = {
        b'E140': [10., 2.5, .625, .15625],
        b'E502': [10., 5., 2., 1., .5, .2],
    }
    _data_type: Final[Dict[bytes, str]] = {
        b'E140': 'h',
        b'E502': 'd',
    }
    _data_numpy_type: Final[Dict[bytes, Type[np.generic]]] = {
        b'E140': np.int16,
        b'E502': np.float64,
    }
    _rescaling_required: Final[Dict[bytes, bool]] = {
        b'E140': True,
        b'E502': False,
    }

    parameter_names: Final[Tuple[str, ...]] = (
        'code',
        'device_name',
        'time',
        'channels_count',
        'frames_count',
        'samples_count',
        'duration',
        'adc_rate',
        'frame_delay',
        'channel_rate',
    )

    def __init__(self, source: Union[None, bytes, BinaryIO] = None) -> None:
        self.code: bytes = b''
        self.device_name: bytes = b''
        self.time: bytes = b''
        self.channels_count: int = 0
        self.frames_count: int = 0
        self.samples_count: int = 0
        self.duration: float = 0.
        self.adc_rate: float = 0.
        self.frame_delay: float = 0.
        self.channel_rate: float = 0.
        self.serial_number: bytes = b''
        raw_data: bytes
        if source is None:
            return
        elif isinstance(source, bytes):
            raw_data = source
        elif isinstance(source, BufferedReader):
            init_pos: int = 0
            if source.seekable():
                init_pos = source.tell()
            raw_data = source.read(max(struct.calcsize(self._struct), 0x439c + 8))
            if source.seekable():
                source.seek(init_pos)
        else:
            raise TypeError(f'Unknown source type: {type(source)}')
        (
            self.code,
            self.device_name,
            self.time,
            self.channels_count,
            self.frames_count,
            self.samples_count,
            self.duration,
            self.adc_rate,
            self.frame_delay,
            self.channel_rate,
        ) = struct.unpack_from(self._struct, raw_data)
        if len(raw_data) >= 0x439c + 8:
            self.serial_number = raw_data[0x439c:0x439c + 8]
        self.code = self.code.strip(b' \0')
        self.device_name = self.device_name.strip(b'\0')
        self.time = self.time.strip(b' \0')
        # hardware corrections
        self.correction_factor: Tuple[float, ...] = tuple()
        self.correction_offset: Tuple[float, ...] = tuple()
        self.MAX_ADC_CODE: int = 0
        if self.device_name in self._max_adc_codes:
            self.MAX_ADC_CODE = self._max_adc_codes[self.device_name]
        else:
            raise NotImplementedError(f'Unknown device type: {self.device_name!r}')
        self.MAX_ADC_VOLTAGE: float = 10.
        if self.device_name in self._adc_ranges:
            self.MAX_ADC_VOLTAGE = 10.  # FIXME: get the value from parameters
        else:
            raise NotImplementedError(f'Unknown device type: {self.device_name!r}')
        if self.device_name in self._adc_ranges:
            self.correction_factor = struct.unpack_from('<' + 'd' * len(self._adc_ranges[self.device_name]),
                                                        raw_data, offset=0x00eb)
            self.correction_offset = struct.unpack_from('<' + 'd' * len(self._adc_ranges[self.device_name]),
                                                        raw_data, offset=0x01eb)
        else:
            raise NotImplementedError(f'Unknown device type: {self.device_name!r}')

        self.DATA_TYPE: str
        self.DATA_NUMPY_TYPE: Type[np.generic]
        self.DATA_TYPE_SIZE: int
        if self.device_name in self._data_type:
            self.DATA_TYPE = self._data_type[self.device_name]
            self.DATA_NUMPY_TYPE = self._data_numpy_type[self.device_name]
            self.DATA_TYPE_SIZE = len(self.DATA_NUMPY_TYPE().tobytes())
        else:
            raise NotImplementedError(f'Unknown device type: {self.device_name!r}')

        self.RESCALING_REQUIRED: bool
        if self.device_name in self._rescaling_required:
            self.RESCALING_REQUIRED = self._rescaling_required[self.device_name]
        else:
            raise NotImplementedError(f'Unknown device type: {self.device_name!r}')


def read_array(source: Union[bytes, BinaryIO], _type: str, count: int = 1) -> Tuple[Union[float, int, bool], ...]:
    if count < 1:
        raise ValueError(f'Invalid count: {count}')
    if _type not in ['c', 'b', 'B', '?', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q', 'n', 'N', 'e', 'f', 'd']:
        raise ValueError(f'Invalid data type: {_type}')
    _full_struct: str = _type * count
    _struct_size = struct.calcsize(_type)
    _full_struct_size = struct.calcsize(_full_struct)
    raw_data: bytes
    if isinstance(source, bytes):
        raw_data = source
    elif isinstance(source, BufferedReader):
        raw_data = source.read(_full_struct_size)
    else:
        raise TypeError(f'Unknown source type: {type(source)}')
    data: Tuple[Union[float, int, bool], ...]
    if len(raw_data) == _full_struct_size:
        data = struct.unpack('<' + _full_struct, raw_data)
    else:
        available_count: int = len(raw_data) // _struct_size
        data = struct.unpack('<' + _type * available_count, raw_data[:_struct_size * available_count])
    return data


def read_numpy_array(source: Union[bytes, BinaryIO], _type: Type[np.generic], count: int = 1) -> NDArray[np.generic]:
    if count < 1:
        raise ValueError(f'Invalid count: {count}')
    _struct_size: int = len(_type().tobytes())
    _full_struct_size: int = _struct_size * count
    raw_data: bytes
    if isinstance(source, bytes):
        raw_data = source
    elif isinstance(source, BufferedReader):
        raw_data = source.read(_full_struct_size)
    else:
        raise TypeError(f'Unknown source type: {type(source)}')
    data: NDArray[np.generic]
    dt: np.dtype = np.dtype(_type)
    dt.newbyteorder('<')
    if len(raw_data) == _full_struct_size:
        data = np.frombuffer(raw_data, dtype=dt)
    else:
        available_count: int = len(raw_data) // _struct_size
        data = np.frombuffer(raw_data[:_struct_size * available_count], dtype=dt)
    return data


def save_txt(filename: Path, x: ArrayLike, fmt: Union[str, Tuple[str]] = '%.18e',
             delimiter: str = ' ', newline: str = os.linesep,
             header: str = '', footer: str = '', comments: str = '# ', encoding: Optional[str] = None) -> None:
    """
    from `numpy.savetxt`

    Save an array to a text file.

    Parameters
    ----------
    filename : {pathlib.Path}
        If the filename ends in ``.gz``, the file is automatically saved in
        compressed gzip format.  `loadtxt` understands gzipped files
        transparently.
    x : {np.ndarray}
        1D or 2D array_like
        Data to be saved to a text file.
    fmt : str or sequence of strs, optional
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case `delimiter` is ignored. For complex `X`, the legal options
        for `fmt` are:

        * a single specifier, `fmt='%.4e'`, resulting in numbers formatted
          like `' (%s+%sj)' % (fmt, fmt)`
        * a full string specifying every real and imaginary part, e.g.
          `' %.4e %+.4ej %.4e %+.4ej %.4e %+.4ej'` for 3 columns
        * a list of specifiers, one per column - in this case, the real
          and imaginary part must have separate specifiers,
          e.g. `['%.3e + %.3ej', '(%.15e%+.15ej)']` for 2 columns
    delimiter : str, optional
        String or character separating columns.
    newline : str, optional
        String or character separating lines.
    header : str, optional
        String that will be written at the beginning of the file.
    footer : str, optional
        String that will be written at the end of the file.
    comments : str, optional
        String that will be prepended to the ``header`` and ``footer`` strings,
        to mark them as comments. Default: '# ',  as expected by e.g.
        ``numpy.loadtxt``.
    encoding : {None, str}, optional
        Encoding used to encode the outputfile. Does not apply to output
        streams. If the encoding is something other than 'bytes' or 'latin1'
        you will not be able to load the file in NumPy versions < 1.14. Default
        is 'latin1'.


    Notes
    -----
    Further explanation of the `fmt` parameter
    (``%[flag]width[.precision]specifier``):

    flags:
        ``-`` : left justify

        ``+`` : Forces to precede result with + or -.

        ``0`` : Left pad the number with zeros instead of space (see width).

    width:
        Minimum number of characters to be printed. The value is not truncated
        if it has more characters.

    precision:
        - For integer specifiers (eg. ``d,i,o,x``), the minimum number of
          digits.
        - For ``e, E`` and ``f`` specifiers, the number of digits to print
          after the decimal point.
        - For ``g`` and ``G``, the maximum number of significant digits.
        - For ``s``, the maximum number of characters.

    specifiers:
        ``c`` : character

        ``d`` or ``i`` : signed decimal integer

        ``e`` or ``E`` : scientific notation with ``e`` or ``E``.

        ``f`` : decimal floating point

        ``g,G`` : use the shorter of ``e,E`` or ``f``

        ``o`` : signed octal

        ``s`` : string of characters

        ``u`` : unsigned decimal integer

        ``x,X`` : unsigned hexadecimal integer

    This explanation of ``fmt`` is not complete, for an exhaustive
    specification see [1]_.

    References
    ----------
    .. [1] `Format Specification Mini-Language
           <https://docs.python.org/library/string.html#format-specification-mini-language>`_,
           Python Documentation.

    """

    try:
        x = np.asarray(x)

        # Handle 1-dimensional arrays
        if x.ndim == 0 or x.ndim > 2:
            raise ValueError(
                "Expected 1D or 2D array, got %dD array instead" % x.ndim)
        elif x.ndim == 1:
            # Common case -- 1d array of numbers
            if x.dtype.names is None:
                x = np.atleast_2d(x).T
                ncol = 1

            # Complex dtype -- each field indicates a separate column
            else:
                ncol = len(x.dtype.names)
        else:
            ncol = x.shape[1]

        # `fmt` can be a string with multiple insertion points or a
        # list of formats.  E.g. '%10.5f\t%10d' or ('%10.5f', '$10d')
        if type(fmt) in (list, tuple):
            if len(fmt) != ncol:
                raise AttributeError(f'fmt has wrong shape. {fmt}')
            fmt = delimiter.join(fmt)
        elif isinstance(fmt, str):
            n_fmt_chars = fmt.count('%')
            error = ValueError(f'fmt has wrong number of %% formats: {fmt}')
            if n_fmt_chars == 1:
                fmt = delimiter.join([fmt] * ncol)
            elif n_fmt_chars != ncol:
                raise error
        else:
            raise ValueError(f'invalid format: {fmt!r}')

        with filename.open('wt', encoding=encoding, newline=newline) as fh:
            if header:
                header = header.replace(newline, newline + comments)
                fh.write(comments + header + newline)
            row_pack: int = 100000
            row_pack_fmt = newline.join([fmt] * row_pack)
            for row in range(0, x.shape[0] - row_pack, row_pack):
                try:
                    fh.write(row_pack_fmt % tuple(x[row:row + row_pack, ...].ravel()) + newline)
                except TypeError:
                    raise TypeError('Mismatch between array data type and format specifier')
            row_pack = x.shape[0] % row_pack
            row_pack_fmt = newline.join([fmt] * row_pack)
            try:
                fh.write(row_pack_fmt % tuple(x[-row_pack:, ...].ravel()) + newline)
            except TypeError:
                raise TypeError('Mismatch between array data type and format specifier')

            if footer:
                footer = footer.replace(newline, newline + comments)
                fh.write(comments + footer + newline)
    finally:
        pass
